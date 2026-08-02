from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autolabeler.data.class_config import load_semantic_classes

from .csv_points import HL320_FEATURE_NAMES, build_hl320_features, load_hl320_csv, primary_return_mask, primary_returns_frame


IGNORE_ID = 255


@dataclass(frozen=True)
class HL320DatasetBuildResult:
    output_dir: Path
    manifest_path: Path
    frame_count: int
    train_frames: list[str]
    val_frames: list[str]


def build_hl320_dataset(
    *,
    csv_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    classes_yaml: str | Path = "configs/classes.yaml",
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
    ignore_id: int = IGNORE_ID,
    prepare_output: bool = True,
    return_mode: str = "primary",
) -> HL320DatasetBuildResult:
    csv_root = Path(csv_dir).expanduser().resolve()
    labels_root = Path(labels_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    classes_path = Path(classes_yaml).expanduser().resolve()
    if not csv_root.is_dir():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_root}")
    class_to_source_id = load_semantic_classes(str(classes_path))
    known_source_ids = {int(value) for value in class_to_source_id.values()}
    ignore_source_ids = {
        int(value)
        for name, value in class_to_source_id.items()
        if int(value) == int(ignore_id) or str(name).casefold() in {"ignore", "ignored"}
    }
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if return_mode not in {"primary", "all"}:
        raise ValueError(f"return_mode must be 'primary' or 'all', got {return_mode!r}")
    if prepare_output:
        _prepare_output_dir(out_root, overwrite=overwrite)
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(csv_root.glob("*.csv"))
    if len(csv_paths) < 2 and val_ratio > 0.0:
        raise ValueError("At least two frames are required when validation split is enabled")
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found in {csv_root}")

    frame_infos = []
    for csv_path in csv_paths:
        echo_frame = load_hl320_csv(csv_path)
        primary_mask = primary_return_mask(echo_frame)
        frame = echo_frame if return_mode == "all" else primary_returns_frame(echo_frame)
        labels_path = resolve_label_path(labels_root, frame.frame_id)
        labels, label_layout = load_flat_labels(
            labels_path,
            expected_points=frame.point_count,
            expected_all_points=echo_frame.point_count,
            primary_mask=primary_mask if return_mode == "primary" else None,
        )
        if return_mode == "all" and label_layout == "primary_returns":
            label_layout = "all_returns"
        valid = valid_xyz_mask(frame.points)
        features = build_hl320_features(frame, echo_frame=echo_frame if return_mode == "primary" else None)
        source_counts = _class_counts(labels[valid], ignore_ids=ignore_source_ids)
        unknown_source_ids = sorted(
            int(value)
            for value in np.unique(labels[valid])
            if int(value) not in known_source_ids and int(value) not in ignore_source_ids
        )
        frame_infos.append(
            {
                "frame_id": frame.frame_id,
                "csv_path": csv_path,
                "labels_path": labels_path,
                "label_layout": label_layout,
                "points": frame.points,
                "features": features,
                "labels": labels,
                "valid": valid,
                "source_counts": source_counts,
                "unknown_source_ids": unknown_source_ids,
            }
        )

    train_ids, val_ids = class_aware_split(
        [(str(info["frame_id"]), dict(info["source_counts"])) for info in frame_infos],
        val_ratio=val_ratio,
        seed=seed,
        ignore_id=ignore_id,
    )
    split_by_frame = {frame_id: "train" for frame_id in train_ids}
    split_by_frame.update({frame_id: "val" for frame_id in val_ids})
    active_source_ids = {
        int(source_id)
        for info in frame_infos
        if split_by_frame[str(info["frame_id"])] == "train"
        for source_id, count in dict(info["source_counts"]).items()
        if int(count) > 0 and int(source_id) in known_source_ids
    }
    taxonomy = build_dense_taxonomy(
        class_to_source_id,
        active_source_ids=active_source_ids,
        ignore_source_ids=ignore_source_ids,
    )
    source_to_training = {
        int(source_id): int(training_id)
        for source_id, training_id in taxonomy["source_id_to_training_id"].items()
    }

    manifest: dict[str, Any] = {
        "version": 1,
        "format": "hl320_pointwise_v1",
        "csv_dir": str(csv_root),
        "csv_layout": "all_returns" if return_mode == "all" else "all_returns_with_primary_block_id_0",
        "return_mode": return_mode,
        "labels_dir": str(labels_root),
        "output_dir": str(out_root),
        "classes_yaml": str(classes_path),
        "feature_names": HL320_FEATURE_NAMES,
        "litept_feature_keys": ["coord", "strength"],
        "litept_in_channels": 3 + len(HL320_FEATURE_NAMES),
        "ignore_id": int(ignore_id),
        "training_ignore_index": -1,
        "taxonomy": taxonomy,
        "split": {
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "train_frames": train_ids,
            "val_frames": val_ids,
        },
        "frames": [],
    }

    aggregate_source_counts: dict[int, int] = {}
    aggregate_training_counts: dict[int, int] = {}
    for info in frame_infos:
        frame_id = str(info["frame_id"])
        split = split_by_frame[frame_id]
        valid = np.asarray(info["valid"], dtype=bool)
        frame_out = out_root / "dataset" / split / frame_id
        frame_out.mkdir(parents=True, exist_ok=True)
        coord = np.asarray(info["points"], dtype=np.float32)[valid, :3]
        features = np.asarray(info["features"], dtype=np.float32)[valid]
        labels = np.asarray(info["labels"], dtype=np.int32)[valid]
        segment = remap_source_to_training(labels, source_to_training)
        training_counts = _class_counts(segment, ignore_ids={-1})
        np.save(frame_out / "coord.npy", coord)
        np.save(frame_out / "features.npy", features)
        np.save(frame_out / "strength.npy", features)
        np.save(frame_out / "segment.npy", segment)
        metadata = {
            "frame_id": frame_id,
            "split": split,
            "source_csv": str(info["csv_path"]),
            "primary_return": "all blockID rows" if return_mode == "all" else "blockID == 0",
            "return_mode": return_mode,
            "source_labels": str(info["labels_path"]),
            "source_label_layout": str(info["label_layout"]),
            "raw_points": int(np.asarray(info["points"]).shape[0]),
            "valid_points": int(coord.shape[0]),
            "dropped_points": int(np.asarray(info["points"]).shape[0] - coord.shape[0]),
            "source_class_counts": {str(key): int(value) for key, value in dict(info["source_counts"]).items()},
            "training_class_counts": {str(key): int(value) for key, value in training_counts.items()},
            "unknown_source_ids": [int(value) for value in info["unknown_source_ids"]],
            "feature_names": HL320_FEATURE_NAMES,
        }
        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["frames"].append(metadata)
        for class_id, count in dict(info["source_counts"]).items():
            aggregate_source_counts[int(class_id)] = aggregate_source_counts.get(int(class_id), 0) + int(count)
        for class_id, count in training_counts.items():
            aggregate_training_counts[int(class_id)] = aggregate_training_counts.get(int(class_id), 0) + int(count)

    manifest["source_class_counts"] = {str(key): int(value) for key, value in sorted(aggregate_source_counts.items())}
    manifest["training_class_counts"] = {str(key): int(value) for key, value in sorted(aggregate_training_counts.items())}
    manifest_path = out_root / "hl320_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return HL320DatasetBuildResult(
        output_dir=out_root,
        manifest_path=manifest_path,
        frame_count=len(frame_infos),
        train_frames=train_ids,
        val_frames=val_ids,
    )


def resolve_label_path(labels_root: Path, frame_id: str) -> Path:
    candidates = [
        labels_root / frame_id / "semantic_mask.npy",
        labels_root / f"{frame_id}.npy",
        labels_root / f"{frame_id}_pred.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No flat label mask found for frame {frame_id!r} under {labels_root}")


def load_flat_labels(
    path: Path,
    *,
    expected_points: int,
    expected_all_points: int | None = None,
    primary_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    labels = np.load(path, allow_pickle=False)
    arr = np.asarray(labels)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Labels {path} dtype must be integer, got {arr.dtype}")
    if arr.shape == (expected_points,):
        return arr.astype(np.int32, copy=False), "primary_returns"
    if expected_all_points is not None and arr.shape == (expected_all_points,):
        if primary_mask is None:
            raise ValueError(f"Labels {path} are all-return labels, but no primary_return_mask was provided")
        primary = np.asarray(primary_mask, dtype=bool)
        if primary.shape != (expected_all_points,):
            raise ValueError(
                f"primary_return_mask has shape {primary.shape}, expected {(expected_all_points,)} for labels {path}"
            )
        selected = arr[primary]
        if selected.shape != (expected_points,):
            raise ValueError(
                f"Selecting primary labels from {path} produced shape {selected.shape}, expected {(expected_points,)}"
            )
        return selected.astype(np.int32, copy=False), "all_returns_selected_primary"
    expected = [(expected_points,)]
    if expected_all_points is not None:
        expected.append((expected_all_points,))
    raise ValueError(f"Labels {path} have shape {arr.shape}, expected one of {expected}")


def valid_xyz_mask(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32)[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    return finite & nonzero


def class_aware_split(
    frame_counts: list[tuple[str, dict[int, int]]],
    *,
    val_ratio: float,
    seed: int,
    ignore_id: int = IGNORE_ID,
) -> tuple[list[str], list[str]]:
    frame_ids = [frame_id for frame_id, _ in frame_counts]
    if val_ratio <= 0.0:
        return frame_ids, []
    rng = np.random.default_rng(seed)
    shuffled = list(frame_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val = set(shuffled[:val_count])
    train = set(shuffled[val_count:])

    class_to_frames: dict[int, list[str]] = {}
    for frame_id, counts in frame_counts:
        for class_id, count in counts.items():
            if int(class_id) == int(ignore_id) or int(count) <= 0:
                continue
            class_to_frames.setdefault(int(class_id), []).append(frame_id)

    for class_id, frames in sorted(class_to_frames.items()):
        if len(frames) < 2:
            if frames[0] not in train:
                val.remove(frames[0])
                train.add(frames[0])
            continue
        if not any(frame_id in train for frame_id in frames):
            chosen = max(frames, key=lambda frame_id: _count_for(frame_counts, frame_id, class_id))
            val.remove(chosen)
            train.add(chosen)
        if not any(frame_id in val for frame_id in frames) and len(val) < len(frame_ids) - 1:
            chosen = min((frame_id for frame_id in frames if frame_id in train), key=lambda frame_id: _count_for(frame_counts, frame_id, class_id))
            train.remove(chosen)
            val.add(chosen)

    return sorted(train), sorted(val)


def _count_for(frame_counts: list[tuple[str, dict[int, int]]], frame_id: str, class_id: int) -> int:
    for current_frame_id, counts in frame_counts:
        if current_frame_id == frame_id:
            return int(counts.get(class_id, 0))
    return 0


def build_dense_taxonomy(
    class_to_source_id: dict[str, int],
    *,
    active_source_ids: set[int],
    ignore_source_ids: set[int],
) -> dict[str, Any]:
    source_to_name = {int(source_id): str(name) for name, source_id in class_to_source_id.items()}
    training_to_source = [
        source_id
        for source_id in sorted(active_source_ids)
        if source_id in source_to_name and source_id not in ignore_source_ids
    ]
    if not training_to_source:
        raise ValueError("No trainable source ids found in the training split")
    source_to_training = {source_id: training_id for training_id, source_id in enumerate(training_to_source)}
    return {
        "num_classes": len(training_to_source),
        "class_names": [source_to_name[source_id] for source_id in training_to_source],
        "source_id_to_training_id": source_to_training,
        "training_id_to_source_id": training_to_source,
        "ignore_source_ids": sorted(ignore_source_ids),
        "excluded_source_ids": [
            {
                "source_id": source_id,
                "name": source_to_name[source_id],
                "reason": "no_train_points",
            }
            for source_id in sorted(source_to_name)
            if source_id not in source_to_training and source_id not in ignore_source_ids
        ],
    }


def remap_source_to_training(labels: np.ndarray, source_to_training: dict[int, int]) -> np.ndarray:
    result = np.full(np.asarray(labels).shape, -1, dtype=np.int32)
    for source_id, training_id in source_to_training.items():
        result[np.asarray(labels) == int(source_id)] = int(training_id)
    return result


def _class_counts(labels: np.ndarray, *, ignore_ids: set[int]) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {
        int(value): int(count)
        for value, count in zip(values, counts)
        if int(value) not in ignore_ids
    }


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
