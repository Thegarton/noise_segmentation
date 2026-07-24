#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.hl320.csv_points import build_hl320_features, load_hl320_csv, primary_returns_frame  # noqa: E402
from autolabeler.teachers.litept_adapter import (  # noqa: E402
    LitePTUnavailableError,
    _add_litept_to_path,
    _load_litept_model,
    normalize_litept_strength,
    remap_training_predictions,
)
from autolabeler.teachers.litept_finetune import valid_litept_points  # noqa: E402


IGNORE_ID = 255


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    fine_tune_dir = Path(args.fine_tune_dir).expanduser().resolve() if args.fine_tune_dir else None
    litept_root = Path(args.litept_root).expanduser().resolve()

    checkpoint = resolve_checkpoint(args.checkpoint, fine_tune_dir)
    litept_config = resolve_litept_config(args.litept_config, fine_tune_dir)
    csv_paths = discover_csv_files(csv_dir, max_frames=args.max_frames)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "csv_dir": str(csv_dir),
                    "output_dir": str(out_dir),
                    "frames": [path.stem for path in csv_paths],
                    "frame_count": len(csv_paths),
                    "litept_root": str(litept_root),
                    "checkpoint": str(checkpoint),
                    "litept_config": str(litept_config),
                    "feature_mode": args.feature_mode,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    _add_litept_to_path(str(litept_root))
    model = _load_litept_model(
        litept_root=str(litept_root),
        checkpoint=str(checkpoint),
        config_path=str(litept_config),
        litept_dataset="custom",
        litept_config=str(litept_config),
        device=args.device,
        force_torch_pointrope=args.force_torch_pointrope,
    )
    feature_mode = resolve_feature_mode(args.feature_mode, model)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, csv_path in enumerate(csv_paths, start=1):
        if args.skip_existing and outputs_exist(out_dir / csv_path.stem):
            frames.append({"frame_id": csv_path.stem, "status": "exists"})
            continue
        print(f"[{index:04d}/{len(csv_paths):04d}] {csv_path}", file=sys.stderr, flush=True)
        frame = load_inference_frame(csv_path, feature_mode=feature_mode)
        result = predict_flat_points(model, frame["points"], strength=frame["strength"])
        frame_out = out_dir / frame["frame_id"]
        frame_out.mkdir(parents=True, exist_ok=True)
        np.save(frame_out / "semantic_mask.npy", result["semantic_mask"])
        np.save(frame_out / "confidence.npy", result["confidence"])
        np.save(frame_out / "valid_indices.npy", result["valid_indices"])
        metadata = build_metadata(
            model=model,
            csv_path=csv_path,
            frame=frame,
            result=result,
            checkpoint=checkpoint,
            litept_config=litept_config,
            feature_mode=feature_mode,
        )
        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        frames.append(
            {
                "frame_id": frame["frame_id"],
                "points": int(frame["points"].shape[0]),
                "valid_points": int(result["valid_indices"].size),
                "semantic_mask": str(frame_out / "semantic_mask.npy"),
                "confidence": str(frame_out / "confidence.npy"),
                "valid_indices": str(frame_out / "valid_indices.npy"),
                "metadata": str(frame_out / "metadata.json"),
                "status": "created",
            }
        )

    manifest = {
        "version": 1,
        "mode": "litept_flat_csv_inference",
        "csv_dir": str(csv_dir),
        "output_dir": str(out_dir),
        "litept_root": str(litept_root),
        "checkpoint": str(checkpoint),
        "litept_config": str(litept_config),
        "class_names": model.class_names,
        "num_classes": model.num_classes,
        "training_id_to_source_id": model.training_id_to_source_id or list(range(model.num_classes)),
        "output_ignore_index": model.output_ignore_index,
        "feature_mode": feature_mode,
        "frames": frames,
    }
    manifest_path = out_dir / "litept_flat_csv_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exported_frames": len(frames), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fine-tuned LitePT model on flat HL320 CSV point clouds.")
    parser.add_argument("--litept-root", required=True)
    parser.add_argument("--csv-dir", required=True, help="Directory with 000000.csv, 000001.csv, ... flat point tables.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fine-tune-dir", default=None, help="Fine-tune output dir. Used to resolve config/checkpoint by default.")
    parser.add_argument("--checkpoint", default=None, help="Defaults to <fine-tune-dir>/experiment/model/model_best.pth")
    parser.add_argument("--litept-config", default=None, help="Defaults to <fine-tune-dir>/litept_custom_config.py")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=("auto", "legacy_intensity", "hl320"),
        default="auto",
        help="legacy_intensity uses only XYZI intensity; hl320 uses the same feature matrix as build_hl320_dataset.py.",
    )
    parser.add_argument("--force-torch-pointrope", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str | None, fine_tune_dir: Path | None) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
    elif fine_tune_dir is not None:
        path = (fine_tune_dir / "experiment" / "model" / "model_best.pth").resolve()
    else:
        raise ValueError("Either --checkpoint or --fine-tune-dir is required")
    if not path.is_file():
        raise FileNotFoundError(f"LitePT checkpoint does not exist: {path}")
    return path


def resolve_litept_config(litept_config: str | None, fine_tune_dir: Path | None) -> Path:
    if litept_config:
        path = Path(litept_config).expanduser().resolve()
    elif fine_tune_dir is not None:
        path = (fine_tune_dir / "litept_custom_config.py").resolve()
    else:
        raise ValueError("Either --litept-config or --fine-tune-dir is required")
    if not path.is_file():
        raise FileNotFoundError(f"LitePT config does not exist: {path}")
    return path


def discover_csv_files(csv_dir: Path, *, max_frames: int | None) -> list[Path]:
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV dir does not exist: {csv_dir}")
    paths = sorted(csv_dir.glob("*.csv"))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")
    return paths


def outputs_exist(frame_out: Path) -> bool:
    return all((frame_out / name).is_file() for name in ("semantic_mask.npy", "confidence.npy", "metadata.json"))


def load_flat_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        header = None
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                header = split_table_row(stripped)
                break
        if header is None:
            raise ValueError(f"CSV point cloud is empty: {path}")
        column_map = {name.strip().casefold(): index for index, name in enumerate(header)}
        required = {name: require_column(column_map, name, path) for name in ("x", "y", "z", "intensity")}
        optional_names = ("cxd", "cyd", "azimuth", "vertical", "slot", "pixel", "hcell", "vcell")
        optional = {name: column_map[name] for name in optional_names if name in column_map}
        max_column = max([*required.values(), *optional.values()] or [0])

        points = []
        optional_values = {name: [] for name in optional}
        for line_number, raw_line in enumerate(handle, start=2):
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = split_table_row(stripped)
            if len(tokens) <= max_column:
                raise ValueError(f"{path}:{line_number}: expected at least {max_column + 1} columns, got {len(tokens)}")
            points.append(
                [
                    parse_float(tokens[required["x"]], path, line_number, "x"),
                    parse_float(tokens[required["y"]], path, line_number, "y"),
                    parse_float(tokens[required["z"]], path, line_number, "z"),
                    parse_float(tokens[required["intensity"]], path, line_number, "intensity"),
                ]
            )
            for name, column in optional.items():
                optional_values[name].append(parse_float(tokens[column], path, line_number, name))

    if not points:
        raise ValueError(f"CSV point cloud contains no points: {path}")
    return {
        "frame_id": path.stem,
        "points": np.asarray(points, dtype=np.float32),
        "columns": header,
        "optional": {name: np.asarray(values, dtype=np.float32) for name, values in optional_values.items()},
    }


def load_inference_frame(path: Path, *, feature_mode: str) -> dict[str, Any]:
    if feature_mode == "hl320":
        echo_frame = load_hl320_csv(path)
        frame = primary_returns_frame(echo_frame)
        return {
            "frame_id": frame.frame_id,
            "points": frame.points,
            "strength": build_hl320_features(frame, echo_frame=echo_frame),
            "columns": frame.columns,
            "optional": frame.fields,
        }
    frame = load_flat_csv(path)
    frame["strength"] = normalize_litept_strength(frame["points"][:, 3]).reshape(-1, 1)
    return frame


def split_table_row(line: str) -> list[str]:
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return line.replace("\t", " ").split()


def require_column(column_map: dict[str, int], name: str, source: Path) -> int:
    key = name.casefold()
    if key not in column_map:
        raise ValueError(f"CSV {source} is missing required column {name!r}")
    return int(column_map[key])


def parse_float(value: str, source: Path, line_number: int, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: cannot parse {column}={value!r} as float") from exc


def resolve_feature_mode(requested: str, model: Any) -> str:
    if requested != "auto":
        return requested
    feature_names = getattr(model.cfg, "feature_names", None)
    in_channels = _model_in_channels(model)
    if feature_names is not None or in_channels > 4:
        return "hl320"
    return "legacy_intensity"


def _model_in_channels(model: Any) -> int:
    try:
        return int(model.cfg.model.backbone.in_channels)
    except Exception:
        return 4


def predict_flat_points(model: Any, points_xyzi: np.ndarray, *, strength: np.ndarray | None = None) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from datasets.transform import Compose, TRANSFORMS
        from datasets.utils import collate_fn
    except Exception as exc:  # pragma: no cover - depends on external LitePT env
        raise LitePTUnavailableError(f"Failed to import LitePT inference transforms: {exc}") from exc

    points = np.asarray(points_xyzi, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points_xyzi must have shape [N,4], got {points.shape}")
    if strength is None:
        strength = normalize_litept_strength(points[:, 3]).reshape(-1, 1)
    strength = np.asarray(strength, dtype=np.float32)
    if strength.ndim != 2 or strength.shape[0] != points.shape[0]:
        raise ValueError(f"strength must have shape [N,C], got {strength.shape} for {points.shape[0]} points")
    valid = valid_litept_points(points)
    valid_indices = np.flatnonzero(valid)

    semantic = np.full((points.shape[0],), model.output_ignore_index, dtype=np.uint16)
    confidence = np.zeros((points.shape[0],), dtype=np.float32)
    if valid_indices.size == 0:
        return {"semantic_mask": semantic, "confidence": confidence, "valid_indices": valid_indices}

    data_dict = {
        "coord": points[valid_indices, :3].astype(np.float32, copy=False),
        "strength": strength[valid_indices].astype(np.float32, copy=False),
        "segment": np.full((valid_indices.size,), -1, dtype=np.int64),
        "index_valid_keys": ["coord", "strength", "segment"],
    }
    test_cfg = model.cfg.data.test.test_cfg
    voxelize = TRANSFORMS.build(test_cfg.voxelize)
    post_transform = Compose(test_cfg.post_transform)

    fragments = [post_transform(fragment) for fragment in voxelize(data_dict)]
    pred = torch.zeros((valid_indices.size, model.num_classes), device=model.device)
    counts = torch.zeros((valid_indices.size, 1), device=model.device)
    with torch.no_grad():
        for fragment in fragments:
            input_dict = collate_fn([fragment])
            for key, value in list(input_dict.items()):
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(model.device, non_blocking=True)
            idx_part = input_dict["index"].long()
            pred_part = model.model(input_dict)["seg_logits"]
            pred_part = F.softmax(pred_part, dim=-1)
            pred.index_add_(0, idx_part, pred_part)
            counts.index_add_(0, idx_part, torch.ones((idx_part.numel(), 1), device=model.device))

    pred = pred / counts.clamp_min(1.0)
    confidence_valid, labels = pred.max(dim=1)
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    source_ids = remap_training_predictions(labels_np, model.training_id_to_source_id or list(range(model.num_classes)))
    semantic[valid_indices] = source_ids
    confidence[valid_indices] = confidence_valid.detach().cpu().numpy().astype(np.float32)
    return {"semantic_mask": semantic, "confidence": confidence, "valid_indices": valid_indices}


def build_metadata(
    *,
    model: Any,
    csv_path: Path,
    frame: dict[str, Any],
    result: dict[str, Any],
    checkpoint: Path,
    litept_config: Path,
    feature_mode: str,
) -> dict[str, Any]:
    source_ids = model.training_id_to_source_id or list(range(model.num_classes))
    semantic = np.asarray(result["semantic_mask"])
    unique_ids, counts = np.unique(semantic, return_counts=True)
    return {
        "frame_id": frame["frame_id"],
        "source_csv": str(csv_path),
        "source_format": "flat_csv",
        "point_count": int(frame["points"].shape[0]),
        "valid_point_count": int(result["valid_indices"].size),
        "invalid_point_count": int(frame["points"].shape[0] - result["valid_indices"].size),
        "semantic_mask": "semantic_mask.npy",
        "confidence_mask": "confidence.npy",
        "valid_indices": "valid_indices.npy",
        "label_space": "litept_custom_flat_csv_semseg",
        "litept_dataset": "custom",
        "litept_config": str(litept_config),
        "checkpoint": str(checkpoint),
        "feature_mode": feature_mode,
        "feature_names": list(getattr(model.cfg, "feature_names", [])),
        "class_names": model.class_names,
        "num_classes": model.num_classes,
        "semantic_classes": {name: int(source_id) for name, source_id in zip(model.class_names, source_ids)},
        "training_id_to_source_id": [int(value) for value in source_ids],
        "ignore_index": int(model.output_ignore_index),
        "class_point_counts": {str(int(label_id)): int(count) for label_id, count in zip(unique_ids, counts)},
        "csv_columns": frame["columns"],
        "csv_optional_columns": sorted(frame["optional"]),
        "device": str(model.device),
        "device_name": model.device_name,
        "device_capability": model.device_capability,
        "pointrope_backend": model.pointrope_backend,
    }


if __name__ == "__main__":
    main()
