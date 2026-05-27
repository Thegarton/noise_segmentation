from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.data.bin_loader import H, W
from autolabeler.data.class_config import load_semantic_classes
from autolabeler.teachers.sam3_text_adapter import Sam3TextResult


IGNORE_ID = 255


@dataclass(frozen=True)
class FusedPointwiseLabels:
    semantic_mask: np.ndarray
    confidence_mask: np.ndarray
    sam3_candidate_mask: np.ndarray
    provenance_counts: dict[str, int]
    metadata: dict[str, Any]


def load_name_mapping(path: str | Path) -> dict[str, str]:
    section = None
    mapping: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            continue
        if section == "source_to_target" and indent == 2 and ":" in stripped:
            source, target = [x.strip() for x in stripped.split(":", 1)]
            mapping[_norm(source)] = target
    return mapping


def class_names_to_project_ids(
    class_names: list[str],
    *,
    source_to_target: dict[str, str],
    project_classes: dict[str, int],
) -> dict[int, int]:
    out: dict[int, int] = {}
    for idx, name in enumerate(class_names):
        target = source_to_target.get(_norm(name))
        if target is None:
            continue
        if target not in project_classes:
            raise ValueError(f"Mapped target class {target!r} is not in project class config")
        out[idx] = int(project_classes[target])
    return out


def lift_sam3_candidates_to_points(
    sam3: Sam3TextResult,
    *,
    point_to_pixel: np.ndarray,
    label_to_id: dict[str, int],
    min_points_per_mask: int,
    min_score: float,
    height: int = H,
    width: int = W,
) -> tuple[np.ndarray, np.ndarray]:
    point_to_pixel = np.asarray(point_to_pixel, dtype=np.int32)
    candidate = np.full((point_to_pixel.shape[0],), IGNORE_ID, dtype=np.uint16)
    candidate_conf = np.zeros((point_to_pixel.shape[0],), dtype=np.float32)
    valid_points = point_to_pixel[:, 0] >= 0

    for mask, score, label in zip(sam3.masks, sam3.scores, sam3.labels):
        target_id = label_to_id.get(str(label))
        if target_id is None or float(score) < min_score:
            continue
        pixels = point_to_pixel[valid_points]
        inside_valid = mask[pixels[:, 1], pixels[:, 0]]
        point_indices = np.where(valid_points)[0][inside_valid]
        if point_indices.size < min_points_per_mask:
            continue
        stronger = score > candidate_conf[point_indices]
        selected = point_indices[stronger]
        candidate[selected] = np.uint16(target_id)
        candidate_conf[selected] = float(score)

    return candidate.reshape(height, width), candidate_conf.reshape(height, width)


def fuse_litept_and_sam3(
    *,
    litept_mask: np.ndarray,
    litept_confidence: np.ndarray,
    litept_id_to_project_id: dict[int, int],
    sam3_candidate_mask: np.ndarray,
    sam3_candidate_confidence: np.ndarray,
    litept_accept_threshold: float,
    sam3_accept_threshold: float,
    ignore_id: int = IGNORE_ID,
) -> FusedPointwiseLabels:
    litept_mask = np.asarray(litept_mask)
    litept_confidence = np.asarray(litept_confidence, dtype=np.float32)
    sam3_candidate_mask = np.asarray(sam3_candidate_mask)
    sam3_candidate_confidence = np.asarray(sam3_candidate_confidence, dtype=np.float32)
    if litept_mask.shape != (H, W):
        raise ValueError(f"LitePT mask must have shape {(H, W)}, got {litept_mask.shape}")
    if litept_confidence.shape != (H, W) or sam3_candidate_mask.shape != (H, W):
        raise ValueError("LitePT confidence and SAM3 candidate mask must match organized mask shape")

    fused = np.zeros((H, W), dtype=np.uint16)
    confidence = np.zeros((H, W), dtype=np.float32)
    provenance = np.full((H, W), "background", dtype=object)

    for source_id, target_id in litept_id_to_project_id.items():
        accepted = (litept_mask == source_id) & (litept_confidence >= litept_accept_threshold)
        fused[accepted] = np.uint16(target_id)
        confidence[accepted] = litept_confidence[accepted]
        provenance[accepted] = "litept_waymo"

    sam3_valid = (sam3_candidate_mask != ignore_id) & (sam3_candidate_confidence >= sam3_accept_threshold)
    empty_or_weak = (fused == 0) | (confidence < litept_accept_threshold)
    accept_sam3 = sam3_valid & empty_or_weak
    conflict = sam3_valid & (fused != 0) & (sam3_candidate_mask != fused) & (confidence < litept_accept_threshold)

    fused[accept_sam3] = sam3_candidate_mask[accept_sam3].astype(np.uint16, copy=False)
    confidence[accept_sam3] = sam3_candidate_confidence[accept_sam3]
    provenance[accept_sam3] = "sam3_text"

    fused[conflict] = np.uint16(ignore_id)
    confidence[conflict] = 0.0
    provenance[conflict] = "needs_review"

    counts = {name: int(np.count_nonzero(provenance == name)) for name in np.unique(provenance)}
    return FusedPointwiseLabels(
        semantic_mask=fused,
        confidence_mask=confidence,
        sam3_candidate_mask=sam3_candidate_mask.astype(np.uint16, copy=False),
        provenance_counts=counts,
        metadata={
            "label_space": "project_pointwise_v1",
            "provenance_counts": counts,
            "fusion_policy": {
                "litept_accept_threshold": litept_accept_threshold,
                "sam3_accept_threshold": sam3_accept_threshold,
                "residual_is_not_noise": True,
            },
        },
    )


def read_class_names_from_metadata(path: str | Path) -> list[str]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    names = metadata.get("class_names")
    if not isinstance(names, list):
        raise ValueError(f"metadata.json has no class_names list: {path}")
    return [str(x) for x in names]


def labels_to_ids_from_prompt_config(project_classes_path: str | Path, prompt_config: dict[str, list[str]]) -> dict[str, int]:
    classes = load_semantic_classes(str(project_classes_path))
    out: dict[str, int] = {}
    for label in prompt_config:
        if label not in classes:
            raise ValueError(f"SAM3 prompt label {label!r} is not in project class config")
        out[label] = int(classes[label])
    return out


def _norm(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")

