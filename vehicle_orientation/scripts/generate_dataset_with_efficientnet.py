#!/usr/bin/env python3
"""Detect vehicles with SAM3 and auto-sort crops with a trained EfficientNet."""

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
    CLASS_NAMES,
    assign_stratified_splits,
    collect_mixed_source_images,
    load_jsonl,
    source_id_from_relative_path,
    summarize_manifest,
    write_jsonl,
)
from vehicle_orientation.model import VehicleOrientationClassifier  # noqa: E402
from vehicle_orientation.preprocessing import (  # noqa: E402
    FisheyeConfig,
    FisheyePreprocessor,
    deduplicate_mask_indices,
    extract_mask_crop,
)
from vehicle_orientation.recovery import (  # noqa: E402
    merge_recovered_with_existing,
    recover_review_records,
)
from vehicle_orientation.sam3_adapter import Sam3VehicleDetector, VehicleDetection  # noqa: E402


DEFAULT_VEHICLE_PROMPTS = (
    "passenger car",
    "sport utility vehicle",
    "van",
    "pickup truck",
    "truck",
    "bus",
)
CLASS_COLORS = {
    "front": np.asarray([40, 200, 80], dtype=np.uint8),
    "rear": np.asarray([230, 70, 70], dtype=np.uint8),
    "side": np.asarray([55, 125, 235], dtype=np.uint8),
}


@dataclass(frozen=True)
class ClassifiedVehicle:
    detection: VehicleDetection
    label: str
    confidence: float
    margin: float
    probabilities: tuple[float, float, float]
    needs_review: bool


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    _prepare_output_dir(out_dir, source_root=image_dir, overwrite=args.overwrite, resume=args.resume)
    for label in CLASS_NAMES:
        (out_dir / "review" / label).mkdir(parents=True, exist_ok=True)

    prompts = tuple(value.strip() for value in (args.prompt or DEFAULT_VEHICLE_PROMPTS) if value.strip())
    if not prompts:
        raise ValueError("At least one generic vehicle prompt is required")
    if args.min_mask_size <= 0:
        raise ValueError(f"--min-mask-size must be positive, got {args.min_mask_size}")
    source_records = collect_mixed_source_images(image_dir, recursive=not args.non_recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        source_records = source_records[: args.max_images]
    samples: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    completed_source_ids: set[str] = set()
    if args.resume:
        samples, source_results, completed_source_ids = _load_resume_progress(
            image_dir=image_dir,
            out_dir=out_dir,
            source_records=source_records,
            recursive=not args.non_recursive,
        )
        print(
            f"Resume: {len(completed_source_ids)} completed frames, {len(samples)} reviewed samples recovered",
            file=sys.stderr,
            flush=True,
        )
        _write_sample_metadata(out_dir, samples)

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
    classifier = VehicleOrientationClassifier(checkpoint, device=args.device)
    crop_padding = float(classifier.crop_padding)

    for image_index, source in enumerate(source_records, start=1):
        source_path = Path(source["source_path"])
        source_id = source_id_from_relative_path(source["source_relative_path"])
        if source_id in completed_source_ids:
            print(
                f"[{image_index:05d}/{len(source_records):05d}] SKIP completed {source_path}",
                file=sys.stderr,
                flush=True,
            )
            continue
        _remove_stale_source_outputs(out_dir, source_id=source_id)
        frame_started = time.perf_counter()
        prepared_path = out_dir / "prepared" / f"{source_id}.jpg"
        annotated_path = out_dir / "annotated" / f"{source_id}.jpg"
        print(f"[{image_index:05d}/{len(source_records):05d}] {source_path}", file=sys.stderr, flush=True)

        preprocess_started = time.perf_counter()
        source_bgr = _read_bgr(source_path)
        prepared_bgr = preprocessor.prepare_bgr(source_bgr)
        _write_bgr(prepared_path, prepared_bgr)
        prepared_rgb = np.ascontiguousarray(prepared_bgr[..., ::-1])
        preprocess_seconds = time.perf_counter() - preprocess_started

        sam3_started = time.perf_counter()
        raw_detections = detector.detect(prepared_path, prompts=prompts)
        detections = filter_vehicle_detections(
            raw_detections,
            min_mask_size=args.min_mask_size,
            nms_iou=args.nms_iou,
        )
        sam3_seconds = time.perf_counter() - sam3_started

        classifier_started = time.perf_counter()
        classified = classify_vehicle_detections(
            classifier=classifier,
            image_rgb=prepared_rgb,
            detections=detections,
            review_min_confidence=args.review_min_confidence,
            review_min_margin=args.review_min_margin,
        )
        classifier_seconds = time.perf_counter() - classifier_started

        frame_records: list[dict[str, Any]] = []
        for instance_index, vehicle in enumerate(classified):
            crop = extract_mask_crop(prepared_rgb, vehicle.detection.mask, padding=crop_padding)
            sample_id = f"{source_id}_{instance_index:03d}"
            sample_dir = out_dir / "samples" / sample_id
            review_path = out_dir / "review" / vehicle.label / f"{sample_id}.png"
            paths = _save_crop_outputs(sample_dir, crop, review_path=review_path)
            probabilities = {
                class_name: float(vehicle.probabilities[index])
                for index, class_name in enumerate(CLASS_NAMES)
            }
            record = {
                "sample_id": sample_id,
                "source_id": source_id,
                "label": vehicle.label,
                "initial_label": vehicle.label,
                "source_path": str(source_path),
                "source_relative_path": source["source_relative_path"],
                "prepared_image": str(prepared_path.relative_to(out_dir)),
                "annotated_image": str(annotated_path.relative_to(out_dir)),
                "rgb_crop": str(paths["rgb"].relative_to(out_dir)),
                "mask": str(paths["mask"].relative_to(out_dir)),
                "masked_rgb": str(paths["masked"].relative_to(out_dir)),
                "classifier_image": str(paths["review"].relative_to(out_dir)),
                "preview": str(paths["preview"].relative_to(out_dir)),
                "vehicle_prompt": vehicle.detection.prompt,
                "vehicle_score": float(vehicle.detection.score),
                "vehicle_box_xywh": None
                if vehicle.detection.box is None
                else np.asarray(vehicle.detection.box, dtype=np.float32).tolist(),
                "classifier_checkpoint": str(checkpoint),
                "classifier_label": vehicle.label,
                "classifier_confidence": float(vehicle.confidence),
                "classifier_margin": float(vehicle.margin),
                "classifier_probabilities": probabilities,
                "classifier_needs_review": bool(vehicle.needs_review),
                "auto_label_source": "efficientnet_b0",
                "label_source": "efficientnet_b0",
                "crop_bbox_xyxy": list(crop.bbox_xyxy),
                "crop_padding": crop_padding,
                "mask_pixels": int(np.count_nonzero(crop.mask)),
                "metadata": str((sample_dir / "metadata.json").relative_to(out_dir)),
            }
            frame_records.append(record)

        _save_annotated_frame(annotated_path, prepared_rgb, classified)
        label_counts = Counter(item.label for item in classified)
        source_result = {
            "source_id": source_id,
            "source_path": str(source_path),
            "source_relative_path": source["source_relative_path"],
            "prepared_image": str(prepared_path.relative_to(out_dir)),
            "annotated_image": str(annotated_path.relative_to(out_dir)),
            "raw_vehicle_detections": len(raw_detections),
            "kept_vehicle_detections": len(detections),
            "class_counts": dict(sorted(label_counts.items())),
            "needs_review": sum(int(item.needs_review) for item in classified),
            "timing_seconds": {
                "preprocess": round(preprocess_seconds, 6),
                "sam3": round(sam3_seconds, 6),
                "efficientnet": round(classifier_seconds, 6),
                "total": round(time.perf_counter() - frame_started, 6),
            },
        }
        samples.extend(frame_records)
        source_results.append(source_result)
        completed_source_ids.add(source_id)
        _write_sample_metadata(out_dir, frame_records)
        _checkpoint_progress(
            out_dir=out_dir,
            samples=samples,
            source_results=source_results,
            completed_source_ids=completed_source_ids,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            status="running",
        )

    if not samples:
        raise RuntimeError("SAM3 produced no vehicle crops; lower --min-score/--min-mask-size or inspect images")
    samples = assign_stratified_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    _write_sample_metadata(out_dir, samples)
    manifest_path = out_dir / "manifest.jsonl"
    _write_jsonl_atomic(manifest_path, samples)
    summary = summarize_manifest(samples)
    dataset_manifest = {
        "version": 3,
        "mode": "sam3_vehicle_efficientnet_bootstrap",
        "source_root": str(image_dir),
        "out_dir": str(out_dir),
        "sam3_root": str(Path(args.sam3_root).expanduser().resolve()),
        "sam3_model_path": str(Path(args.sam3_model_path).expanduser().resolve()),
        "vehicle_prompts": list(prompts),
        "sam3_min_score": float(args.min_score),
        "min_mask_size": int(args.min_mask_size),
        "nms_iou": float(args.nms_iou),
        "classifier_checkpoint": str(checkpoint),
        "classifier_device": str(classifier.device),
        "classifier_input_size": int(classifier.input_size),
        "classifier_crop_padding": crop_padding,
        "review_min_confidence": float(args.review_min_confidence),
        "review_min_margin": float(args.review_min_margin),
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
            "folders": [f"review/{label}" for label in CLASS_NAMES],
            "instructions": "Move or delete PNG files, then run reindex_dataset.py.",
        },
        "needs_review_samples": sum(int(item["classifier_needs_review"]) for item in samples),
        "summary": summary,
        "sources": source_results,
        "manifest_jsonl": str(manifest_path),
        "generation_state": str(out_dir / "generation_state.json"),
        "completed_frames": len(completed_source_ids),
    }
    manifest_json = out_dir / "dataset_manifest.json"
    _write_json_atomic(manifest_json, dataset_manifest)
    _write_generation_state(
        out_dir=out_dir,
        source_results=source_results,
        completed_source_ids=completed_source_ids,
        samples=len(samples),
        status="complete",
    )
    print(json.dumps({"manifest": str(manifest_json), **summary}, ensure_ascii=False, indent=2))


def filter_vehicle_detections(
    detections: Sequence[VehicleDetection],
    *,
    min_mask_size: int,
    nms_iou: float,
) -> list[VehicleDetection]:
    valid = [
        item
        for item in detections
        if int(np.count_nonzero(np.asarray(item.mask, dtype=bool))) >= int(min_mask_size)
    ]
    keep = deduplicate_mask_indices(
        [item.mask for item in valid],
        [item.score for item in valid],
        iou_threshold=nms_iou,
    )
    return [valid[index] for index in keep]


def classify_vehicle_detections(
    *,
    classifier: Any,
    image_rgb: np.ndarray,
    detections: Sequence[VehicleDetection],
    review_min_confidence: float,
    review_min_margin: float,
) -> list[ClassifiedVehicle]:
    if not detections:
        return []
    decisions = classifier.classify(
        image_rgb,
        [item.mask for item in detections],
        min_confidence=review_min_confidence,
        min_margin=review_min_margin,
    )
    if len(decisions) != len(detections):
        raise ValueError(f"Classifier returned {len(decisions)} decisions for {len(detections)} vehicle masks")
    output = []
    for detection, decision in zip(detections, decisions):
        label = str(decision.predicted_class)
        if label not in CLASS_NAMES:
            raise ValueError(f"Classifier returned unknown orientation class {label!r}; expected {CLASS_NAMES}")
        confidence = float(decision.confidence)
        margin = float(decision.margin)
        probabilities = tuple(float(value) for value in decision.probabilities)
        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError(f"Expected {len(CLASS_NAMES)} classifier probabilities, got {probabilities}")
        output.append(
            ClassifiedVehicle(
                detection=detection,
                label=label,
                confidence=confidence,
                margin=margin,
                probabilities=probabilities,
                needs_review=confidence < review_min_confidence or margin < review_min_margin,
            )
        )
    return output


def _save_annotated_frame(path: Path, image_rgb: np.ndarray, vehicles: Sequence[ClassifiedVehicle]) -> None:
    from PIL import Image, ImageDraw  # noqa: WPS433

    overlay = np.asarray(image_rgb, dtype=np.uint8).copy()
    for vehicle in vehicles:
        mask = np.asarray(vehicle.detection.mask, dtype=bool)
        color = CLASS_COLORS[vehicle.label]
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45).astype(np.uint8)
    canvas = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for vehicle in vehicles:
        mask = np.asarray(vehicle.detection.mask, dtype=bool)
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            continue
        x0, y0 = int(columns.min()), int(rows.min())
        x1, y1 = int(columns.max()) + 1, int(rows.max()) + 1
        color = tuple(int(value) for value in CLASS_COLORS[vehicle.label])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        marker = " review" if vehicle.needs_review else ""
        text = f"{vehicle.label} {vehicle.confidence:.2f}{marker}"
        text_box = draw.textbbox((x0, y0), text)
        text_height = int(text_box[3] - text_box[1])
        text_y = max(0, y0 - text_height - 4)
        text_box = draw.textbbox((x0 + 2, text_y + 2), text)
        draw.rectangle((x0, text_y, text_box[2] + 2, text_box[3] + 2), fill=(0, 0, 0))
        draw.text((x0 + 2, text_y + 2), text, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


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
    overlay[crop.mask] = (
        overlay[crop.mask].astype(np.float32) * 0.55
        + np.asarray([50, 220, 90], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    separator = np.full((crop.rgb.shape[0], 8, 3), 255, dtype=np.uint8)
    preview = np.concatenate([crop.rgb, separator, crop.masked_rgb, separator, overlay], axis=1)
    Image.fromarray(preview, mode="RGB").save(preview_path, quality=95)
    return {"rgb": rgb_path, "mask": mask_path, "masked": masked_path, "review": review_path, "preview": preview_path}


def _write_sample_metadata(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    for record in samples:
        metadata_path = out_dir / str(record["metadata"])
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_resume_progress(
    *,
    image_dir: Path,
    out_dir: Path,
    source_records: list[dict[str, str]],
    recursive: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    state = _read_json_object(out_dir / "generation_state.json")
    completed_source_ids = {str(value) for value in state.get("completed_source_ids", [])}
    if not completed_source_ids:
        annotated_dir = out_dir / "annotated"
        if annotated_dir.is_dir():
            completed_source_ids = {
                path.stem
                for path in annotated_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            }

    available_source_ids = {
        source_id_from_relative_path(record["source_relative_path"])
        for record in source_records
    }
    unknown_completed = sorted(completed_source_ids - available_source_ids)
    if unknown_completed:
        print(
            f"Warning: {len(unknown_completed)} completed frame ids are outside the current --max-images selection",
            file=sys.stderr,
        )

    manifest_path = out_dir / "manifest.jsonl"
    existing = load_jsonl(manifest_path) if manifest_path.is_file() else []
    recovery = recover_review_records(
        image_dir=image_dir,
        dataset_dir=out_dir,
        recursive=recursive,
        completed_source_ids=completed_source_ids if completed_source_ids else None,
    )
    if not completed_source_ids:
        completed_source_ids = {str(record["source_id"]) for record in recovery.records}
    samples = merge_recovered_with_existing(recovery.records, existing)
    source_results_value = state.get("sources", [])
    source_results = [dict(value) for value in source_results_value if isinstance(value, dict)]
    return samples, source_results, completed_source_ids


def _checkpoint_progress(
    *,
    out_dir: Path,
    samples: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    completed_source_ids: set[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    status: str,
) -> None:
    checkpoint_records = assign_stratified_splits(
        samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    _write_jsonl_atomic(out_dir / "manifest.jsonl", checkpoint_records)
    _write_generation_state(
        out_dir=out_dir,
        source_results=source_results,
        completed_source_ids=completed_source_ids,
        samples=len(samples),
        status=status,
    )


def _write_generation_state(
    *,
    out_dir: Path,
    source_results: list[dict[str, Any]],
    completed_source_ids: set[str],
    samples: int,
    status: str,
) -> None:
    _write_json_atomic(
        out_dir / "generation_state.json",
        {
            "version": 1,
            "status": status,
            "completed_frames": len(completed_source_ids),
            "completed_source_ids": sorted(completed_source_ids),
            "samples": int(samples),
            "sources": source_results,
        },
    )


def _remove_stale_source_outputs(out_dir: Path, *, source_id: str) -> None:
    for path in (out_dir / "samples").glob(f"{source_id}_*") if (out_dir / "samples").is_dir() else []:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    for label in CLASS_NAMES:
        class_dir = out_dir / "review" / label
        if not class_dir.is_dir():
            continue
        for path in class_dir.glob(f"{source_id}_*"):
            if path.is_file():
                path.unlink()
    for path in (out_dir / "prepared" / f"{source_id}.jpg", out_dir / "annotated" / f"{source_id}.jpg"):
        if path.is_file():
            path.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, records)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prepare_output_dir(out_dir: Path, *, source_root: Path, overwrite: bool, resume: bool) -> None:
    if out_dir == source_root or source_root in out_dir.parents or out_dir in source_root.parents:
        raise ValueError("--out-dir and --image-dir must be disjoint directories")
    if resume:
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume because output directory does not exist: {out_dir}")
        return
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


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Dataset generation requires opencv-python") from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect generic vehicles with SAM3 and sort their masked crops with a trained EfficientNet."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing mixed camera images.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Trained vehicle orientation model_best.pth.")
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-model-path", required=True)
    parser.add_argument("--prompt", action="append", default=None, help="Generic vehicle prompt; repeat as needed.")
    parser.add_argument("--min-score", type=float, default=0.45, help="SAM3 vehicle detection threshold.")
    parser.add_argument("--min-mask-size", type=int, default=900)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--review-min-confidence", type=float, default=0.70)
    parser.add_argument("--review-min-margin", type=float, default=0.10)
    parser.add_argument("--device", default="auto", help="EfficientNet device: auto, cuda, or cpu.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted output directory and skip frames checkpointed in generation_state.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=3060)
    parser.add_argument(
        "--fisheye-dtype",
        choices=("linear", "equalarea", "orthographic", "stereographic"),
        default="equalarea",
    )
    parser.add_argument("--projection", choices=("perspective", "cylindrical"), default="cylindrical")
    parser.add_argument("--fov", type=float, default=190.0)
    parser.add_argument("--pfov", type=float, default=140.0)
    parser.add_argument("--xcenter", type=float, default=960.0)
    parser.add_argument("--ycenter", type=float, default=750.0)
    parser.add_argument("--radius", type=float, default=1068.0)
    parser.add_argument("--mask-radius", type=float, default=1515.0)
    parser.add_argument("--mask-center-y-offset", type=float, default=300.0)
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="lanczos")
    parser.add_argument("--no-color-correction", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
