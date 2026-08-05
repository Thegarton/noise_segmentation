#!/usr/bin/env python3
"""Build a front/rear/other vehicle crop dataset with one reused SAM3 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_orientation.dataset import (  # noqa: E402
    assign_stratified_splits,
    collect_source_images,
    limit_sources_balanced,
    parse_folder_mappings,
    summarize_manifest,
    write_jsonl,
)
from vehicle_orientation.preprocessing import (  # noqa: E402
    FisheyeConfig,
    FisheyePreprocessor,
    deduplicate_mask_indices,
    extract_mask_crop,
)
from vehicle_orientation.sam3_adapter import Sam3VehicleDetector  # noqa: E402


DEFAULT_PROMPTS = ("vehicle", "car", "passenger vehicle")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, overwrite=args.overwrite, source_root=source_root)

    folder_mapping = parse_folder_mappings(args.folder_map)
    source_records = collect_source_images(source_root, folder_mapping)
    source_records = limit_sources_balanced(source_records, args.max_images)
    prompts = tuple(prompt.strip() for prompt in (args.prompt or DEFAULT_PROMPTS) if prompt.strip())
    if not prompts:
        raise ValueError("At least one vehicle prompt is required")
    if args.min_mask_size <= 0:
        raise ValueError(f"--min-mask-size must be positive, got {args.min_mask_size}")

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
        mask_center_y_offset=args.mask_center_y_offset,
        color_correction=not args.no_color_correction,
    )
    preprocessor = FisheyePreprocessor(fisheye_config)
    detector = Sam3VehicleDetector(
        sam3_root=args.sam3_root,
        model_dir=args.sam3_model_path,
        min_score=args.min_score,
        use_fa3=args.use_fa3,
    )

    samples: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    for index, source in enumerate(source_records, start=1):
        started_at = time.perf_counter()
        source_path = Path(source["source_path"])
        source_id = _source_id(source["label"], source["source_relative_path"])
        prepared_path = out_dir / "prepared" / source["label"] / f"{source_id}.jpg"
        print(f"[{index:05d}/{len(source_records):05d}] {source_path}", file=sys.stderr, flush=True)

        image_bgr = _read_bgr(source_path)
        prepared_bgr = preprocessor.prepare_bgr(image_bgr)
        _write_bgr(prepared_path, prepared_bgr)
        prepared_rgb = np.ascontiguousarray(prepared_bgr[..., ::-1])

        raw_detections = detector.detect(prepared_path, prompts=prompts)
        valid_detections = [
            item for item in raw_detections if int(np.count_nonzero(item.mask)) >= args.min_mask_size
        ]
        keep = deduplicate_mask_indices(
            [item.mask for item in valid_detections],
            [item.score for item in valid_detections],
            iou_threshold=args.nms_iou,
        )
        detections = [valid_detections[item_index] for item_index in keep]
        for instance_index, detection in enumerate(detections):
            crop = extract_mask_crop(
                prepared_rgb,
                detection.mask,
                padding=args.crop_padding,
            )
            sample_id = f"{source_id}_{instance_index:03d}"
            sample_dir = out_dir / "samples" / source["label"] / sample_id
            paths = _save_crop_outputs(sample_dir, crop)
            record = {
                "sample_id": sample_id,
                "source_id": source_id,
                "label": source["label"],
                "source_path": str(source_path),
                "source_relative_path": source["source_relative_path"],
                "prepared_image": str(prepared_path.relative_to(out_dir)),
                "rgb_crop": str(paths["rgb"].relative_to(out_dir)),
                "mask": str(paths["mask"].relative_to(out_dir)),
                "masked_rgb": str(paths["masked"].relative_to(out_dir)),
                "preview": str(paths["preview"].relative_to(out_dir)),
                "sam3_prompt": detection.prompt,
                "sam3_score": float(detection.score),
                "sam3_box_xywh": None
                if detection.box is None
                else np.asarray(detection.box, dtype=np.float32).tolist(),
                "crop_bbox_xyxy": list(crop.bbox_xyxy),
                "mask_pixels": int(np.count_nonzero(crop.mask)),
                "metadata": str((sample_dir / "metadata.json").relative_to(out_dir)),
            }
            samples.append(record)

        source_results.append(
            {
                "source_id": source_id,
                "label": source["label"],
                "source_path": str(source_path),
                "source_relative_path": source["source_relative_path"],
                "prepared_image": str(prepared_path.relative_to(out_dir)),
                "raw_detections": len(raw_detections),
                "valid_detections": len(valid_detections),
                "kept_detections": len(detections),
                "processing_time_seconds": round(time.perf_counter() - started_at, 6),
            }
        )

    if not samples:
        raise RuntimeError("SAM3 produced no vehicle crops; lower --min-score/--min-mask-size or inspect prompts")
    samples = assign_stratified_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    for record in samples:
        metadata_path = out_dir / record["metadata"]
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = out_dir / "manifest.jsonl"
    write_jsonl(manifest_path, samples)
    manifest = {
        "version": 1,
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "folder_mapping": folder_mapping,
        "prompts": list(prompts),
        "sam3_root": str(Path(args.sam3_root).expanduser().resolve()),
        "sam3_model_path": str(Path(args.sam3_model_path).expanduser().resolve()),
        "min_score": float(args.min_score),
        "min_mask_size": int(args.min_mask_size),
        "nms_iou": float(args.nms_iou),
        "crop_padding": float(args.crop_padding),
        "fisheye": fisheye_config.to_dict(),
        "remap_cache_entries": preprocessor.cache_size,
        "split": {
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "seed": int(args.seed),
        },
        "summary": summarize_manifest(samples),
        "sources": source_results,
        "manifest_jsonl": str(manifest_path),
    }
    manifest_json = out_dir / "dataset_manifest.json"
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_json), **manifest["summary"]}, ensure_ascii=False, indent=2))


def _prepare_output_dir(out_dir: Path, *, overwrite: bool, source_root: Path) -> None:
    if out_dir == source_root or source_root in out_dir.parents or out_dir in source_root.parents:
        raise ValueError("--out-dir and --source-root must be disjoint directories")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}; use --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _source_id(label: str, relative_path: str) -> str:
    digest = hashlib.sha1(f"{label}:{relative_path}".encode("utf-8")).hexdigest()[:12]
    stem = Path(relative_path).stem
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    return f"{label}_{safe_stem}_{digest}"


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
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, np.asarray(image, dtype=np.uint8))
    if not ok:
        raise OSError(f"OpenCV could not encode image for {path}")
    encoded.tofile(path)


def _save_crop_outputs(sample_dir: Path, crop: Any) -> dict[str, Path]:
    from PIL import Image  # noqa: WPS433

    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / "rgb.png"
    mask_path = sample_dir / "mask.png"
    masked_path = sample_dir / "masked_rgb.png"
    preview_path = sample_dir / "preview.jpg"
    Image.fromarray(crop.rgb, mode="RGB").save(rgb_path)
    Image.fromarray(crop.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(crop.masked_rgb, mode="RGB").save(masked_path)
    overlay = crop.rgb.copy()
    color = np.asarray([50, 220, 90], dtype=np.float32)
    overlay[crop.mask] = (
        overlay[crop.mask].astype(np.float32) * 0.55 + color * 0.45
    ).astype(np.uint8)
    separator = np.full((crop.rgb.shape[0], 8, 3), 255, dtype=np.uint8)
    preview = np.concatenate([crop.rgb, separator, crop.masked_rgb, separator, overlay], axis=1)
    Image.fromarray(preview, mode="RGB").save(preview_path, quality=95)
    return {"rgb": rgb_path, "mask": mask_path, "masked": masked_path, "preview": preview_path}


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Dataset building requires opencv-python") from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SAM3 mask-based front/rear/other vehicle dataset.")
    parser.add_argument("--source-root", required=True, help="Root containing front, rear, and other image folders.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--folder-map",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Override a class folder, for example rear=backlights_and_licence_plate.",
    )
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-model-path", required=True)
    parser.add_argument("--prompt", action="append", default=None, help="Vehicle prompt; repeat for multiple prompts.")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-mask-size", type=int, default=900)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--use-fa3", action="store_true")
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
    parser.add_argument("--mask-radius", type=float, default=860.0)
    parser.add_argument("--mask-center-y-offset", type=float, default=-125.0)
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="lanczos")
    parser.add_argument("--no-color-correction", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
