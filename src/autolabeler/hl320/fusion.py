from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.data.class_config import load_noise_groups, load_semantic_classes
from autolabeler.hl320.csv_points import load_hl320_csv


IGNORE_ID = 255

PROVENANCE_CODES = {
    "ignore": 0,
    "litept": 1,
    "sam3_fill": 2,
    "sam3_override": 3,
    "noise_protected": 4,
}


@dataclass(frozen=True)
class FusionPolicy:
    min_sam3_confidence: float = 0.7
    litept_keep_threshold: float = 0.6
    noise_protect_threshold: float = 0.4
    sam3_override_margin: float = 0.2
    allow_sam3_background: bool = False
    ignore_id: int = IGNORE_ID


@dataclass(frozen=True)
class PointPrediction:
    labels: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class FusionResult:
    labels: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    sam3_lifted_labels: np.ndarray
    sam3_lifted_confidence: np.ndarray
    stats: dict[str, Any]


def fuse_hl320_predictions(
    *,
    csv_dir: str | Path,
    litept_dir: str | Path,
    sam3_dir: str | Path,
    classes_yaml: str | Path,
    out_dir: str | Path,
    policy: FusionPolicy = FusionPolicy(),
    overwrite: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    csv_root = Path(csv_dir).expanduser().resolve()
    litept_root = Path(litept_dir).expanduser().resolve()
    sam3_root = Path(sam3_dir).expanduser().resolve()
    classes_path = Path(classes_yaml).expanduser().resolve()
    output_root = Path(out_dir).expanduser().resolve()
    _require_dir(csv_root, "CSV directory")
    _require_dir(litept_root, "LitePT inference directory")
    _require_dir(sam3_root, "SAM3 directory")
    if not classes_path.is_file():
        raise FileNotFoundError(f"Classes YAML does not exist: {classes_path}")
    _prepare_output_dir(output_root, overwrite=overwrite)

    class_to_id = load_semantic_classes(str(classes_path))
    known_ids = {int(value) for value in class_to_id.values()}
    noise_ids = noise_source_ids(class_to_id, load_noise_groups(str(classes_path)))
    csv_paths = sorted(csv_root.glob("*.csv"))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        csv_paths = csv_paths[:max_frames]
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in {csv_root}")

    manifest: dict[str, Any] = {
        "version": 1,
        "mode": "hl320_point_fusion_v1",
        "csv_dir": str(csv_root),
        "litept_dir": str(litept_root),
        "sam3_dir": str(sam3_root),
        "classes_yaml": str(classes_path),
        "out_dir": str(output_root),
        "policy": asdict(policy),
        "provenance_codes": PROVENANCE_CODES,
        "noise_source_ids": sorted(noise_ids),
        "frames": [],
        "summary": init_fusion_stats(),
    }

    for csv_path in csv_paths:
        frame = load_hl320_csv(csv_path)
        litept = load_point_prediction(litept_root / frame.frame_id, expected_points=frame.point_count)
        sam3 = lift_sam3_to_points(
            sam3_frame_dir=sam3_root / frame.frame_id,
            cxd=frame.fields.get("cxd"),
            cyd=frame.fields.get("cyd"),
            known_ids=known_ids,
            policy=policy,
        )
        fused = fuse_point_predictions(
            litept=litept,
            sam3=sam3,
            noise_ids=noise_ids,
            policy=policy,
        )
        frame_out = output_root / frame.frame_id
        frame_out.mkdir(parents=True, exist_ok=True)
        np.save(frame_out / "semantic_mask.npy", fused.labels)
        np.save(frame_out / "confidence.npy", fused.confidence)
        np.save(frame_out / "provenance.npy", fused.provenance)
        np.save(frame_out / "sam3_lifted_mask.npy", fused.sam3_lifted_labels)
        np.save(frame_out / "sam3_lifted_confidence.npy", fused.sam3_lifted_confidence)
        metadata = {
            "frame_id": frame.frame_id,
            "source_csv": str(csv_path),
            "source_litept": str(litept_root / frame.frame_id),
            "source_sam3": str(sam3_root / frame.frame_id),
            "point_count": frame.point_count,
            "semantic_mask": "semantic_mask.npy",
            "confidence": "confidence.npy",
            "provenance": "provenance.npy",
            "sam3_lifted_mask": "sam3_lifted_mask.npy",
            "sam3_lifted_confidence": "sam3_lifted_confidence.npy",
            "stats": fused.stats,
        }
        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["frames"].append({**metadata, "path": str(frame_out)})
        accumulate_fusion_stats(manifest["summary"], fused.stats)

    manifest_path = output_root / "hl320_fusion_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_point_prediction(frame_dir: Path, *, expected_points: int) -> PointPrediction:
    semantic_path = frame_dir / "semantic_mask.npy"
    confidence_path = frame_dir / "confidence.npy"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Missing point semantic mask: {semantic_path}")
    labels = np.load(semantic_path, allow_pickle=False)
    if labels.shape != (expected_points,):
        raise ValueError(f"{semantic_path} has shape {labels.shape}, expected {(expected_points,)}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{semantic_path} dtype must be integer, got {labels.dtype}")
    if confidence_path.is_file():
        confidence = np.load(confidence_path, allow_pickle=False)
        if confidence.shape != labels.shape:
            raise ValueError(f"{confidence_path} has shape {confidence.shape}, expected {labels.shape}")
        if not np.issubdtype(confidence.dtype, np.number):
            raise ValueError(f"{confidence_path} dtype must be numeric, got {confidence.dtype}")
        confidence = np.asarray(confidence, dtype=np.float32)
    else:
        confidence = np.zeros(expected_points, dtype=np.float32)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    return PointPrediction(labels=np.asarray(labels, dtype=np.int32), confidence=confidence)


def lift_sam3_to_points(
    *,
    sam3_frame_dir: Path,
    cxd: np.ndarray | None,
    cyd: np.ndarray | None,
    known_ids: set[int],
    policy: FusionPolicy,
) -> PointPrediction:
    if cxd is None or cyd is None:
        raise ValueError("HL320 CSV must contain Cxd/Cyd columns to lift SAM3 masks")
    semantic_path = sam3_frame_dir / "semantic_mask.npy"
    confidence_path = sam3_frame_dir / "confidence.npy"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Missing SAM3 semantic mask: {semantic_path}")
    semantic = np.load(semantic_path, allow_pickle=False)
    if semantic.ndim != 2:
        raise ValueError(f"SAM3 semantic mask must be 2D, got {semantic.shape}: {semantic_path}")
    if not np.issubdtype(semantic.dtype, np.integer):
        raise ValueError(f"SAM3 semantic mask dtype must be integer, got {semantic.dtype}")
    if confidence_path.is_file():
        confidence = np.load(confidence_path, allow_pickle=False)
        if confidence.shape != semantic.shape:
            raise ValueError(f"SAM3 confidence shape {confidence.shape} does not match semantic {semantic.shape}")
        if not np.issubdtype(confidence.dtype, np.number):
            raise ValueError(f"SAM3 confidence dtype must be numeric, got {confidence.dtype}")
        confidence = np.nan_to_num(np.asarray(confidence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        confidence = np.zeros(semantic.shape, dtype=np.float32)

    cxd = np.asarray(cxd, dtype=np.float64)
    cyd = np.asarray(cyd, dtype=np.float64)
    if cxd.shape != cyd.shape:
        raise ValueError(f"Cxd/Cyd shapes differ: {cxd.shape} vs {cyd.shape}")
    point_count = cxd.size
    labels = np.full(point_count, policy.ignore_id, dtype=np.int32)
    point_confidence = np.zeros(point_count, dtype=np.float32)
    height, width = semantic.shape

    finite_projection = np.isfinite(cxd) & np.isfinite(cyd)
    u = np.full(point_count, -1, dtype=np.int64)
    v = np.full(point_count, -1, dtype=np.int64)
    u[finite_projection] = round_like_point_labeler(cxd[finite_projection])
    v[finite_projection] = round_like_point_labeler(cyd[finite_projection])
    inside = finite_projection & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    projected_labels = np.full(point_count, policy.ignore_id, dtype=np.int64)
    projected_confidence = np.zeros(point_count, dtype=np.float32)
    projected_labels[inside] = semantic[v[inside], u[inside]].astype(np.int64, copy=False)
    projected_confidence[inside] = confidence[v[inside], u[inside]].astype(np.float32, copy=False)
    high_confidence = inside & (projected_confidence >= policy.min_sam3_confidence)
    known = np.isin(projected_labels, np.asarray(sorted(known_ids), dtype=np.int64))
    non_background = policy.allow_sam3_background | (projected_labels != 0)
    accepted = high_confidence & known & (projected_labels != policy.ignore_id) & non_background
    labels[accepted] = projected_labels[accepted].astype(np.int32, copy=False)
    point_confidence[inside] = projected_confidence[inside]
    return PointPrediction(labels=labels, confidence=point_confidence)


def fuse_point_predictions(
    *,
    litept: PointPrediction,
    sam3: PointPrediction,
    noise_ids: set[int],
    policy: FusionPolicy,
) -> FusionResult:
    if litept.labels.shape != sam3.labels.shape:
        raise ValueError(f"LitePT and SAM3 label shapes differ: {litept.labels.shape} vs {sam3.labels.shape}")
    if litept.confidence.shape != litept.labels.shape or sam3.confidence.shape != sam3.labels.shape:
        raise ValueError("Prediction confidence arrays must match label shape")

    litept_labels = np.asarray(litept.labels, dtype=np.int32)
    litept_conf = np.asarray(litept.confidence, dtype=np.float32)
    sam3_labels = np.asarray(sam3.labels, dtype=np.int32)
    sam3_conf = np.asarray(sam3.confidence, dtype=np.float32)
    fused = litept_labels.copy()
    confidence = litept_conf.copy()
    provenance = np.full(litept_labels.shape, PROVENANCE_CODES["litept"], dtype=np.uint8)

    litept_ignored = litept_labels == policy.ignore_id
    litept_weak = litept_conf < policy.litept_keep_threshold
    sam3_valid = (sam3_labels != policy.ignore_id) & (sam3_conf >= policy.min_sam3_confidence)
    if not policy.allow_sam3_background:
        sam3_valid &= sam3_labels != 0

    noise_array = np.asarray(sorted(noise_ids), dtype=np.int32)
    litept_is_noise = np.isin(litept_labels, noise_array) if noise_array.size else np.zeros(litept_labels.shape, dtype=bool)
    noise_protected = litept_is_noise & (litept_conf >= policy.noise_protect_threshold)
    fill_from_sam3 = sam3_valid & (litept_ignored | litept_weak) & ~noise_protected
    strong_camera_conflict = (
        sam3_valid
        & ~fill_from_sam3
        & ~noise_protected
        & (sam3_labels != litept_labels)
        & (sam3_conf >= litept_conf + policy.sam3_override_margin)
    )

    fused[fill_from_sam3] = sam3_labels[fill_from_sam3]
    confidence[fill_from_sam3] = sam3_conf[fill_from_sam3]
    provenance[fill_from_sam3] = PROVENANCE_CODES["sam3_fill"]

    fused[strong_camera_conflict] = sam3_labels[strong_camera_conflict]
    confidence[strong_camera_conflict] = sam3_conf[strong_camera_conflict]
    provenance[strong_camera_conflict] = PROVENANCE_CODES["sam3_override"]

    provenance[noise_protected] = PROVENANCE_CODES["noise_protected"]
    ignored = fused == policy.ignore_id
    confidence[ignored] = 0.0
    provenance[ignored] = PROVENANCE_CODES["ignore"]

    stats = fusion_stats(
        fused=fused,
        provenance=provenance,
        sam3_valid=sam3_valid,
        noise_protected=noise_protected,
    )
    return FusionResult(
        labels=fused.astype(np.int32, copy=False),
        confidence=np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False),
        provenance=provenance,
        sam3_lifted_labels=sam3_labels,
        sam3_lifted_confidence=sam3_conf,
        stats=stats,
    )


def noise_source_ids(class_to_id: dict[str, int], noise_groups: dict[str, list[str]]) -> set[int]:
    names = {name for members in noise_groups.values() for name in members}
    names.update(name for name in class_to_id if "noise" in name.casefold())
    return {int(class_to_id[name]) for name in names if name in class_to_id}


def round_like_point_labeler(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return rounded.astype(np.int64)


def fusion_stats(
    *,
    fused: np.ndarray,
    provenance: np.ndarray,
    sam3_valid: np.ndarray,
    noise_protected: np.ndarray,
) -> dict[str, Any]:
    counts_by_label = {
        str(int(label_id)): int(count)
        for label_id, count in zip(*np.unique(fused, return_counts=True))
    }
    provenance_by_name = {
        name: int(np.count_nonzero(provenance == code))
        for name, code in PROVENANCE_CODES.items()
    }
    return {
        "points": int(fused.size),
        "sam3_valid_points": int(np.count_nonzero(sam3_valid)),
        "noise_protected_points": int(np.count_nonzero(noise_protected)),
        "class_point_counts": counts_by_label,
        "provenance_counts": provenance_by_name,
    }


def init_fusion_stats() -> dict[str, Any]:
    return {
        "points": 0,
        "sam3_valid_points": 0,
        "noise_protected_points": 0,
        "class_point_counts": {},
        "provenance_counts": {name: 0 for name in PROVENANCE_CODES},
    }


def accumulate_fusion_stats(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in ("points", "sam3_valid_points", "noise_protected_points"):
        total[key] += int(frame.get(key, 0))
    for label_id, count in frame.get("class_point_counts", {}).items():
        total["class_point_counts"][str(label_id)] = int(total["class_point_counts"].get(str(label_id), 0)) + int(count)
    for name, count in frame.get("provenance_counts", {}).items():
        total["provenance_counts"][str(name)] = int(total["provenance_counts"].get(str(name), 0)) + int(count)


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
            shutil.rmtree(path)
        else:
            path.rmdir()
    path.mkdir(parents=True)
