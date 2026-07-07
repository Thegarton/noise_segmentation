#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.teachers.litept_adapter import normalize_litept_strength  # noqa: E402
from autolabeler.teachers.litept_finetune import (  # noqa: E402
    build_training_taxonomy,
    read_labels_xml,
    remap_source_labels,
    valid_litept_points,
)


def main() -> None:
    args = parse_args()
    labeler_dir = Path(args.labeler_dir).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    frame_ids = [args.frame_id] if args.frame_id else discover_frame_ids(export_dir)

    taxonomy = load_taxonomy(labeler_dir=labeler_dir, output_dir=output_dir)
    source_to_training = {int(k): int(v) for k, v in taxonomy["source_id_to_training_id"].items()}
    ignore_source_ids = {int(value) for value in taxonomy.get("ignore_source_ids", [255])}
    id_to_name = {
        int(item["source_id"]): str(item["name"])
        for item in taxonomy.get("classes", [])
    }
    excluded_by_id = {
        int(item["source_id"]): str(item.get("reason", "excluded"))
        for item in taxonomy.get("excluded_classes", [])
    }

    summaries = []
    for frame_id in frame_ids:
        summaries.append(
            audit_frame(
                frame_id=frame_id,
                labeler_dir=labeler_dir,
                export_dir=export_dir,
                output_dir=output_dir,
                source_to_training=source_to_training,
                ignore_source_ids=ignore_source_ids,
                id_to_name=id_to_name,
                excluded_by_id=excluded_by_id,
                inspect_index=args.inspect_index,
            )
        )

    aggregate = aggregate_summaries(summaries)
    payload = {
        "labeler_dir": str(labeler_dir),
        "export_dir": str(export_dir),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "taxonomy": {
            "num_classes": int(taxonomy["num_classes"]),
            "source_id_to_training_id": source_to_training,
            "training_id_to_source_id": taxonomy["training_id_to_source_id"],
            "excluded_classes": taxonomy.get("excluded_classes", []),
            "ignore_source_ids": sorted(ignore_source_ids),
        },
        "aggregate": aggregate,
        "frames": summaries,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit point_labeler export -> LitePT fine-tune dataset conversion.")
    parser.add_argument("--labeler-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir", default=None, help="Fine-tune output dir containing taxonomy.json and dataset/.")
    parser.add_argument("--frame-id", default=None, help="Audit one frame, e.g. 000003.")
    parser.add_argument("--inspect-index", type=int, default=None, help="Inspect one index in prepared coord/segment arrays.")
    return parser.parse_args()


def discover_frame_ids(export_dir: Path) -> list[str]:
    frame_ids = sorted(
        path.name
        for path in export_dir.iterdir()
        if path.is_dir() and (path / "semantic_mask.npy").is_file()
    )
    if not frame_ids:
        raise FileNotFoundError(f"No exported semantic_mask.npy files found in {export_dir}")
    return frame_ids


def load_taxonomy(*, labeler_dir: Path, output_dir: Path | None) -> dict[str, Any]:
    taxonomy_path = output_dir / "taxonomy.json" if output_dir is not None else None
    if taxonomy_path is not None and taxonomy_path.is_file():
        return json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return build_training_taxonomy(read_labels_xml(labeler_dir / "labels.xml"))


def audit_frame(
    *,
    frame_id: str,
    labeler_dir: Path,
    export_dir: Path,
    output_dir: Path | None,
    source_to_training: dict[int, int],
    ignore_source_ids: set[int],
    id_to_name: dict[int, str],
    excluded_by_id: dict[int, str],
    inspect_index: int | None,
) -> dict[str, Any]:
    points_path = labeler_dir / "velodyne" / f"{frame_id}.bin"
    mask_path = export_dir / frame_id / "semantic_mask.npy"
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing point cloud: {points_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing exported mask: {mask_path}")

    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    mask = np.load(mask_path, allow_pickle=False).reshape(-1)
    if mask.size != points.shape[0]:
        raise ValueError(f"{frame_id}: mask size {mask.size} != point count {points.shape[0]}")

    valid = valid_litept_points(points)
    valid_indices = np.flatnonzero(valid)
    invalid_nonfinite = ~np.isfinite(points[:, :3]).all(axis=1)
    invalid_near_zero = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (np.abs(points[:, 0]) < 1e-4)
        & (np.abs(points[:, 1]) < 1e-4)
        & (np.abs(points[:, 2]) < 1e-4)
    )

    segment_full = remap_source_labels(mask, source_to_training)
    segment = segment_full[valid]
    valid_source = mask[valid].astype(np.int64, copy=False)
    ignored_reason_counts, ignored_source_counts = ignored_reasons(
        valid_source=valid_source,
        segment=segment,
        source_to_training=source_to_training,
        ignore_source_ids=ignore_source_ids,
        id_to_name=id_to_name,
        excluded_by_id=excluded_by_id,
    )

    summary: dict[str, Any] = {
        "frame_id": frame_id,
        "points": int(points.shape[0]),
        "mask_shape": [int(value) for value in np.load(mask_path, mmap_mode="r", allow_pickle=False).shape],
        "valid_points": int(valid_indices.size),
        "dropped_points": int(points.shape[0] - valid_indices.size),
        "dropped_nonfinite_xyz": int(np.count_nonzero(invalid_nonfinite)),
        "dropped_near_zero_xyz": int(np.count_nonzero(invalid_near_zero)),
        "source_label_counts": counts_dict(mask),
        "valid_source_label_counts": counts_dict(valid_source),
        "training_segment_counts": counts_dict(segment),
        "ignored_valid_points": int(np.count_nonzero(segment < 0)),
        "ignored_reason_counts": ignored_reason_counts,
        "ignored_source_counts": ignored_source_counts,
    }

    dataset_check = check_prepared_dataset(
        frame_id=frame_id,
        output_dir=output_dir,
        points=points,
        valid=valid,
        expected_segment=segment,
    )
    if dataset_check:
        summary["prepared_dataset"] = dataset_check

    if inspect_index is not None:
        summary["inspect_index"] = inspect_prepared_index(
            inspect_index=inspect_index,
            points=points,
            mask=mask,
            valid_indices=valid_indices,
            segment=segment,
            frame_strength=normalize_litept_strength(points[valid, 3]).reshape(-1),
            source_to_training=source_to_training,
            id_to_name=id_to_name,
            output_dir=output_dir,
            frame_id=frame_id,
        )
    return summary


def ignored_reasons(
    *,
    valid_source: np.ndarray,
    segment: np.ndarray,
    source_to_training: dict[int, int],
    ignore_source_ids: set[int],
    id_to_name: dict[int, str],
    excluded_by_id: dict[int, str],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    ignored_sources = valid_source[segment < 0]
    for source_id in ignored_sources:
        source_int = int(source_id)
        source_counts[source_int] += 1
        if source_int in ignore_source_ids:
            reason_counts["ignore_source_id"] += 1
        elif source_int in excluded_by_id:
            reason_counts[f"excluded_{excluded_by_id[source_int]}"] += 1
        elif source_int not in source_to_training:
            reason_counts["unknown_or_not_trainable_source_id"] += 1
        else:
            reason_counts["other"] += 1
    source_payload = {
        str(source_id): {
            "name": id_to_name.get(source_id, f"id_{source_id}"),
            "count": int(count),
            "reason": (
                "ignore_source_id"
                if source_id in ignore_source_ids
                else excluded_by_id.get(source_id, "unknown_or_not_trainable_source_id")
            ),
        }
        for source_id, count in sorted(source_counts.items())
    }
    return {key: int(value) for key, value in sorted(reason_counts.items())}, source_payload


def check_prepared_dataset(
    *,
    frame_id: str,
    output_dir: Path | None,
    points: np.ndarray,
    valid: np.ndarray,
    expected_segment: np.ndarray,
) -> dict[str, Any] | None:
    if output_dir is None:
        return None
    candidates = [output_dir / "dataset" / split / frame_id for split in ("train", "val")]
    frame_dir = next((path for path in candidates if path.is_dir()), None)
    if frame_dir is None:
        return {"found": False}
    coord = np.load(frame_dir / "coord.npy", allow_pickle=False)
    strength = np.load(frame_dir / "strength.npy", allow_pickle=False).reshape(-1)
    segment = np.load(frame_dir / "segment.npy", allow_pickle=False)
    expected_coord = points[valid, :3].astype(np.float32, copy=False)
    expected_strength = normalize_litept_strength(points[valid, 3]).reshape(-1)
    return {
        "found": True,
        "split": frame_dir.parent.name,
        "path": str(frame_dir),
        "coord_shape": [int(value) for value in coord.shape],
        "segment_shape": [int(value) for value in segment.shape],
        "strength_shape": [int(value) for value in strength.shape],
        "coord_matches_valid_points": bool(coord.shape == expected_coord.shape and np.allclose(coord, expected_coord, equal_nan=True)),
        "segment_matches_remap": bool(segment.shape == expected_segment.shape and np.array_equal(segment, expected_segment)),
        "strength_matches_normalized_intensity": bool(
            strength.shape == expected_strength.shape and np.allclose(strength, expected_strength, equal_nan=True)
        ),
    }


def inspect_prepared_index(
    *,
    inspect_index: int,
    points: np.ndarray,
    mask: np.ndarray,
    valid_indices: np.ndarray,
    segment: np.ndarray,
    frame_strength: np.ndarray,
    source_to_training: dict[int, int],
    id_to_name: dict[int, str],
    output_dir: Path | None,
    frame_id: str,
) -> dict[str, Any]:
    if inspect_index < 0 or inspect_index >= valid_indices.size:
        raise IndexError(f"inspect index {inspect_index} outside prepared valid point count {valid_indices.size}")
    raw_index = int(valid_indices[inspect_index])
    source_id = int(mask[raw_index])
    payload: dict[str, Any] = {
        "prepared_index": int(inspect_index),
        "raw_point_index": raw_index,
        "point_xyzi": [float(value) for value in points[raw_index]],
        "source_id": source_id,
        "source_name": id_to_name.get(source_id, f"id_{source_id}"),
        "expected_training_id": int(source_to_training.get(source_id, -1)),
        "prepared_segment": int(segment[inspect_index]),
        "raw_intensity": float(points[raw_index, 3]),
        "normalized_strength": float(frame_strength[inspect_index]),
    }
    if output_dir is not None:
        frame_dir = next(
            (output_dir / "dataset" / split / frame_id for split in ("train", "val") if (output_dir / "dataset" / split / frame_id).is_dir()),
            None,
        )
        if frame_dir is not None:
            payload["dataset_coord"] = [float(value) for value in np.load(frame_dir / "coord.npy")[inspect_index]]
            payload["dataset_segment"] = int(np.load(frame_dir / "segment.npy")[inspect_index])
            payload["dataset_strength"] = float(np.load(frame_dir / "strength.npy")[inspect_index].reshape(-1)[0])
    return payload


def counts_dict(values: np.ndarray) -> dict[str, int]:
    ids, counts = np.unique(np.asarray(values).reshape(-1), return_counts=True)
    return {str(int(label_id)): int(count) for label_id, count in zip(ids, counts)}


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    ignored_reasons = Counter()
    for summary in summaries:
        for key in ("points", "valid_points", "dropped_points", "dropped_nonfinite_xyz", "dropped_near_zero_xyz", "ignored_valid_points"):
            totals[key] += int(summary[key])
        ignored_reasons.update({key: int(value) for key, value in summary["ignored_reason_counts"].items()})
    payload = {key: int(value) for key, value in totals.items()}
    payload["ignored_reason_counts"] = {key: int(value) for key, value in sorted(ignored_reasons.items())}
    return payload


if __name__ == "__main__":
    main()
