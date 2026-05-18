from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from ..data.dataset_indexer import FrameRecord, build_dataset_index
from ..data.frame_loader import load_frame
from ..data.schemas import SemanticSegmentationResult


@dataclass(frozen=True)
class LitePTInferencePlan:
    litept_root: str
    checkpoint: str
    input_dir: str
    output_dir: str
    frame_count: int
    frame_ids: list[str]


class LitePTUnavailableError(RuntimeError):
    pass


def build_litept_inference_plan(
    *,
    litept_root: str,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    input_format: str = "auto",
    validate_checkpoint: bool = True,
) -> LitePTInferencePlan:
    root = Path(litept_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"LitePT root does not exist or is not a directory: {root}")

    ckpt = Path(checkpoint).expanduser().resolve()
    if validate_checkpoint and not ckpt.exists():
        raise FileNotFoundError(f"LitePT checkpoint does not exist: {ckpt}")

    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input dir does not exist or is not a directory: {input_path}")

    records = build_dataset_index(str(input_path), input_format=input_format)
    return LitePTInferencePlan(
        litept_root=str(root),
        checkpoint=str(ckpt),
        input_dir=str(input_path),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        frame_count=len(records),
        frame_ids=[r.frame_id for r in records],
    )


def run_litept_inference(
    *,
    litept_root: str,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    input_format: str = "auto",
    config_path: str = "configs/classes.yaml",
    litept_config: str | None = None,
) -> list[SemanticSegmentationResult]:
    plan = build_litept_inference_plan(
        litept_root=litept_root,
        checkpoint=checkpoint,
        input_dir=input_dir,
        output_dir=output_dir,
        input_format=input_format,
        validate_checkpoint=True,
    )
    _add_litept_to_path(plan.litept_root)

    records = build_dataset_index(plan.input_dir, input_format=input_format)
    model = _load_litept_model(
        litept_root=plan.litept_root,
        checkpoint=plan.checkpoint,
        config_path=config_path,
        litept_config=litept_config,
    )

    results: list[SemanticSegmentationResult] = []
    for record in records:
        frame = load_frame(record.lidar_path, record.frame_id, input_format=input_format)
        semantic_mask, confidence_mask = _predict_frame(model, frame.points_range)
        results.append(
            SemanticSegmentationResult(
                frame_id=record.frame_id,
                semantic_mask=semantic_mask,
                confidence_mask=confidence_mask,
                pseudo_label_version="litept_pretrained_v0",
                provenance="student_predicted",
            )
        )
    return results


def _add_litept_to_path(litept_root: str) -> None:
    root = str(Path(litept_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_litept_model(*, litept_root: str, checkpoint: str, config_path: str, litept_config: str | None):
    candidates = _candidate_entrypoints(Path(litept_root))
    raise LitePTUnavailableError(
        "LitePT model adapter is not wired to this external repository layout yet. "
        "Run dry-run first, then inspect LitePT entrypoints and connect _load_litept_model/_predict_frame. "
        f"litept_root={litept_root} checkpoint={checkpoint} config={config_path} "
        f"litept_config={litept_config} candidate_files={candidates[:12]}"
    )


def _predict_frame(model, points_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise LitePTUnavailableError("LitePT prediction is unavailable until _load_litept_model is implemented")


def _candidate_entrypoints(root: Path) -> list[str]:
    names = []
    for pattern in ("*infer*.py", "*test*.py", "*demo*.py", "*eval*.py"):
        names.extend(str(p.relative_to(root)) for p in root.rglob(pattern) if p.is_file())
    return sorted(names)
