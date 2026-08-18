#!/usr/bin/env python3
"""Create an automatically sorted front/rear/side vehicle review dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_orientation.dataset import (  # noqa: E402
    assign_stratified_splits,
    collect_mixed_source_images,
    source_id_from_relative_path,
    summarize_manifest,
    write_jsonl,
)
from vehicle_orientation.preprocessing import (  # noqa: E402
    FisheyeConfig,
    FisheyePreprocessor,
    deduplicate_mask_indices,
    extract_mask_crop,
)
from vehicle_orientation.sam3_adapter import Sam3VehicleDetector, VehicleDetection  # noqa: E402


DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "vehicle_dataset_prompts.yaml"
ORIENTATION_LABELS = ("front", "rear", "side")


@dataclass(frozen=True)
class AutoLabeledVehicle:
    label: str
    mask: np.ndarray
    vehicle_score: float
    vehicle_prompt: str
    vehicle_box: np.ndarray | None
    orientation_score: float | None
    orientation_prompt: str | None
    orientation_overlap: float
    label_source: str


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, overwrite=args.overwrite, source_root=source_root)
    for label in ORIENTATION_LABELS:
        (out_dir / "review" / label).mkdir(parents=True, exist_ok=True)
    if args.min_mask_size <= 0:
        raise ValueError(f"--min-mask-size must be positive, got {args.min_mask_size}")

    source_records = collect_mixed_source_images(source_root, recursive=not args.non_recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        source_records = source_records[: args.max_images]
    prompt_groups = load_prompt_groups(args.prompt_config)
    if args.prompt:
        prompt_groups["generic"] = [value.strip() for value in args.prompt if value.strip()]
    labeled_prompts = [
        (label, prompt)
        for label in ("generic", *ORIENTATION_LABELS)
        for prompt in prompt_groups[label]
    ]

    fisheye_config = FisheyeConfig(
        output_width=args.output_width,
        output_height=args.output_height,
        fisheye_format="circular",
        dtype=args.fisheye_dtype,
        projection=args.projection,
        fov=args.fov,
        pfov=args.pfov,
        pfov_axis="horizontal",
        xcenter=args.xcenter,
        ycenter=args.ycenter,
        radius=args.radius,
        interpolation=args.interpolation,
        mask_radius=args.mask_radius,
        mask_lower_radius=args.mask_lower_radius,
        mask_center_y_offset=args.mask_center_y_offset,
        color_correction=not args.no_color_correction,
    )
    preprocessor = FisheyePreprocessor(fisheye_config)
    detector = Sam3VehicleDetector(
        sam3_root=args.sam3_root,
        model_dir=args.sam3_model_path,
        min_score=args.min_score,
        use_fa3=args.use_fa3,
        cache_visual_features=args.cache_visual_features,
    )

    samples: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    for index, source in enumerate(source_records, start=1):
        started_at = time.perf_counter()
        source_path = Path(source["source_path"])
        source_id = source_id_from_relative_path(source["source_relative_path"])
        prepared_path = out_dir / "prepared" / f"{source_id}.jpg"
        print(f"[{index:05d}/{len(source_records):05d}] {source_path}", file=sys.stderr, flush=True)

        image_bgr = _read_bgr(source_path)
        prepared_bgr = preprocessor.prepare_bgr(image_bgr)
        _write_bgr(prepared_path, prepared_bgr)
        prepared_rgb = np.ascontiguousarray(prepared_bgr[..., ::-1])
        raw_detections = detector.detect_labeled(prepared_path, labeled_prompts=labeled_prompts)
        vehicles = auto_label_vehicle_instances(
            raw_detections,
            min_mask_size=args.min_mask_size,
            nms_iou=args.nms_iou,
            orientation_match_overlap=args.orientation_match_overlap,
        )

        for instance_index, vehicle in enumerate(vehicles):
            crop = extract_mask_crop(prepared_rgb, vehicle.mask, padding=args.crop_padding)
            sample_id = f"{source_id}_{instance_index:03d}"
            sample_dir = out_dir / "samples" / sample_id
            review_path = out_dir / "review" / vehicle.label / f"{sample_id}.png"
            paths = _save_crop_outputs(sample_dir, crop, review_path=review_path)
            record = {
                "sample_id": sample_id,
                "source_id": source_id,
                "label": vehicle.label,
                "initial_label": vehicle.label,
                "source_path": str(source_path),
                "source_relative_path": source["source_relative_path"],
                "prepared_image": str(prepared_path.relative_to(out_dir)),
                "rgb_crop": str(paths["rgb"].relative_to(out_dir)),
                "mask": str(paths["mask"].relative_to(out_dir)),
                "masked_rgb": str(paths["masked"].relative_to(out_dir)),
                "classifier_image": str(paths["review"].relative_to(out_dir)),
                "preview": str(paths["preview"].relative_to(out_dir)),
                "vehicle_prompt": vehicle.vehicle_prompt,
                "vehicle_score": float(vehicle.vehicle_score),
                "vehicle_box_xywh": None
                if vehicle.vehicle_box is None
                else np.asarray(vehicle.vehicle_box, dtype=np.float32).tolist(),
                "orientation_prompt": vehicle.orientation_prompt,
                "orientation_score": vehicle.orientation_score,
                "orientation_overlap": float(vehicle.orientation_overlap),
                "auto_label_source": vehicle.label_source,
                "label_source": vehicle.label_source,
                "crop_bbox_xyxy": list(crop.bbox_xyxy),
                "mask_pixels": int(np.count_nonzero(crop.mask)),
                "metadata": str((sample_dir / "metadata.json").relative_to(out_dir)),
            }
            samples.append(record)

        label_counts = Counter(item.label for item in vehicles)
        source_results.append(
            {
                "source_id": source_id,
                "source_path": str(source_path),
                "source_relative_path": source["source_relative_path"],
                "prepared_image": str(prepared_path.relative_to(out_dir)),
                "raw_detections": len(raw_detections),
                "kept_vehicles": len(vehicles),
                "auto_label_counts": dict(sorted(label_counts.items())),
                "processing_time_seconds": round(time.perf_counter() - started_at, 6),
            }
        )

    if not samples:
        raise RuntimeError("SAM3 produced no vehicle crops; lower thresholds or inspect the prompt config")
    samples = assign_stratified_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_sample_metadata(out_dir, samples)

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, samples)
    manifest = {
        "version": 2,
        "mode": "mixed_images_auto_orientation",
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "prompt_config": str(Path(args.prompt_config).expanduser().resolve()),
        "prompt_groups": prompt_groups,
        "sam3_root": str(Path(args.sam3_root).expanduser().resolve()),
        "sam3_model_path": str(Path(args.sam3_model_path).expanduser().resolve()),
        "cache_visual_features": bool(args.cache_visual_features),
        "min_score": float(args.min_score),
        "min_mask_size": int(args.min_mask_size),
        "nms_iou": float(args.nms_iou),
        "orientation_match_overlap": float(args.orientation_match_overlap),
        "crop_padding": float(args.crop_padding),
        "fisheye": fisheye_config.to_dict(),
        "remap_cache_entries": preprocessor.cache_size,
        "split": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "seed": int(args.seed),
            "grouped_by": "source_id",
        },
        "manual_review": {
            "folders": ["review/front", "review/rear", "review/side"],
            "instructions": "Move PNG files between review folders, then run reindex_dataset.py.",
        },
        "summary": summarize_manifest(samples),
        "sources": source_results,
        "manifest_jsonl": str(manifest_path),
    }
    manifest_json = out_dir / "dataset_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_json), **manifest["summary"]}, ensure_ascii=False, indent=2))


def auto_label_vehicle_instances(
    detections: Sequence[VehicleDetection],
    *,
    min_mask_size: int,
    nms_iou: float,
    orientation_match_overlap: float,
) -> list[AutoLabeledVehicle]:
    if not 0.0 <= orientation_match_overlap <= 1.0:
        raise ValueError(f"orientation_match_overlap must be in [0,1], got {orientation_match_overlap}")
    valid = [
        item
        for item in detections
        if item.label in {"generic", *ORIENTATION_LABELS}
        and int(np.count_nonzero(np.asarray(item.mask, dtype=bool))) >= min_mask_size
    ]
    generic = [item for item in valid if item.label == "generic"]
    orientation = [item for item in valid if item.label in ORIENTATION_LABELS]
    generic_keep = deduplicate_mask_indices(
        [item.mask for item in generic],
        [item.score for item in generic],
        iou_threshold=nms_iou,
    )
    generic = [generic[index] for index in generic_keep]

    output: list[AutoLabeledVehicle] = []
    matched_orientation: set[int] = set()
    for vehicle in generic:
        candidates = []
        for orientation_index, candidate in enumerate(orientation):
            if orientation_index in matched_orientation:
                continue
            overlap = mask_overlap_coefficient(vehicle.mask, candidate.mask)
            if overlap < orientation_match_overlap:
                continue
            weighted_score = float(candidate.score) * (0.5 + 0.5 * overlap)
            candidates.append((weighted_score, float(candidate.score), overlap, orientation_index, candidate))
        if candidates:
            _, _, overlap, orientation_index, best = max(
                candidates,
                key=lambda item: (item[0], item[1], item[2], -item[3]),
            )
            matched_orientation.add(orientation_index)
            output.append(
                AutoLabeledVehicle(
                    label=best.label,
                    mask=np.asarray(vehicle.mask, dtype=bool),
                    vehicle_score=float(vehicle.score),
                    vehicle_prompt=vehicle.prompt,
                    vehicle_box=vehicle.box,
                    orientation_score=float(best.score),
                    orientation_prompt=best.prompt,
                    orientation_overlap=float(overlap),
                    label_source="matched_orientation_prompt",
                )
            )
        else:
            output.append(
                AutoLabeledVehicle(
                    label="side",
                    mask=np.asarray(vehicle.mask, dtype=bool),
                    vehicle_score=float(vehicle.score),
                    vehicle_prompt=vehicle.prompt,
                    vehicle_box=vehicle.box,
                    orientation_score=None,
                    orientation_prompt=None,
                    orientation_overlap=0.0,
                    label_source="unmatched_generic_fallback_side",
                )
            )

    return output


def mask_overlap_coefficient(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape:
        raise ValueError(f"Mask shape mismatch: {left_mask.shape} vs {right_mask.shape}")
    intersection = int(np.count_nonzero(left_mask & right_mask))
    denominator = min(int(np.count_nonzero(left_mask)), int(np.count_nonzero(right_mask)))
    return float(intersection) / float(denominator) if denominator > 0 else 0.0


def load_prompt_groups(path: str | Path) -> dict[str, list[str]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Prompt config does not exist: {config_path}")
    groups: dict[str, list[str]] = {}
    current: str | None = None
    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            current = stripped[:-1].strip()
            groups.setdefault(current, [])
        elif indent >= 2 and stripped.startswith("- ") and current is not None:
            value = stripped[2:].strip()
            if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            if value:
                groups[current].append(value)
        else:
            raise ValueError(f"Unsupported prompt YAML syntax at {config_path}:{line_number}")
    missing = [label for label in ("generic", *ORIENTATION_LABELS) if not groups.get(label)]
    if missing:
        raise ValueError(f"Prompt config must contain non-empty generic/front/rear/side groups; missing {missing}")
    return {label: groups[label] for label in ("generic", *ORIENTATION_LABELS)}


def write_sample_metadata(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    for record in samples:
        metadata_path = out_dir / record["metadata"]
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_output_dir(out_dir: Path, *, overwrite: bool, source_root: Path) -> None:
    if out_dir == source_root or source_root in out_dir.parents or out_dir in source_root.parents:
        raise ValueError("--out-dir and --source-root must be disjoint directories")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}; use --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _read_bgr(path: Path) -> np.ndarray:
    cv2 = _import_cv2()
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return np.asarray(image, dtype=np.uint8)


def _write_bgr(path: Path, image: np.ndarray) -> None:
    cv2 = _import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise OSError(f"OpenCV could not encode image for {path}")
    encoded.tofile(path)


def _save_crop_outputs(sample_dir: Path, crop: Any, *, review_path: Path) -> dict[str, Path]:
    from PIL import Image  # noqa: WPS433

    sample_dir.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / "rgb.png"
    mask_path = sample_dir / "mask.png"
    masked_path = sample_dir / "masked_rgb.png"
    preview_path = sample_dir / "preview.jpg"
    Image.fromarray(crop.rgb, mode="RGB").save(rgb_path)
    Image.fromarray(crop.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(crop.masked_rgb, mode="RGB").save(masked_path)
    Image.fromarray(crop.masked_rgb, mode="RGB").save(review_path)
    overlay = crop.rgb.copy()
    color = np.asarray([50, 220, 90], dtype=np.float32)
    overlay[crop.mask] = (overlay[crop.mask].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
    separator = np.full((crop.rgb.shape[0], 8, 3), 255, dtype=np.uint8)
    preview = np.concatenate([crop.rgb, separator, crop.masked_rgb, separator, overlay], axis=1)
    Image.fromarray(preview, mode="RGB").save(preview_path, quality=95)
    return {"rgb": rgb_path, "mask": mask_path, "masked": masked_path, "review": review_path, "preview": preview_path}


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Dataset building requires opencv-python") from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect cars in one mixed image folder and auto-sort masked crops into front/rear/side review folders."
    )
    parser.add_argument("--source-root", required=True, help="One directory containing mixed camera images.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-model-path", required=True)
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--prompt", action="append", default=None, help="Override generic vehicle prompts; repeat as needed.")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-mask-size", type=int, default=900)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--orientation-match-overlap", type=float, default=0.30)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument(
        "--cache-visual-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse SAM3 image-backbone features across prompts (enabled by default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=3060)
    parser.add_argument("--fisheye-dtype", choices=("linear", "equalarea", "orthographic", "stereographic"), default="equalarea")
    parser.add_argument("--projection", choices=("perspective", "cylindrical"), default="cylindrical")
    parser.add_argument("--fov", type=float, default=190.0)
    parser.add_argument("--pfov", type=float, default=140.0)
    parser.add_argument("--xcenter", type=float, default=960.0)
    parser.add_argument("--ycenter", type=float, default=750.0)
    parser.add_argument("--radius", type=float, default=1068.0)
    parser.add_argument(
        "--mask-radius",
        "--mask-upper-radius",
        dest="mask_radius",
        type=float,
        default=860.0,
        help="Radius used for the upper half of the source-image mask.",
    )
    parser.add_argument(
        "--mask-lower-radius",
        type=float,
        default=None,
        help="Optional smaller radius for the lower half. Default uses --mask-radius for both halves.",
    )
    parser.add_argument("--mask-center-y-offset", type=float, default=-125.0)
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="lanczos")
    parser.add_argument("--no-color-correction", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
