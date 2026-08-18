#!/usr/bin/env python3
"""Build a manually reviewable SAM3 sign-type dataset from camera images."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sign_type_classifier.clustering import SignCandidate, cluster_sign_detections  # noqa: E402
from sign_type_classifier.colour_correction import simple_colour_correction_rgb  # noqa: E402
from sign_type_classifier.dataset import (  # noqa: E402
    CLASS_NAMES,
    assign_grouped_splits,
    collect_source_images,
    load_jsonl,
    source_id_from_relative_path,
    summarize_manifest,
    write_jsonl,
)
from sign_type_classifier.detection_filters import (  # noqa: E402
    GEOMETRY_FILTER_RULES,
    SIZE_FILTER_RULES,
    filter_detections,
    rejection_to_json,
)
from sign_type_classifier.features import (  # noqa: E402
    SignCrops,
    build_numeric_features,
    extract_sign_crops,
    numeric_feature_names,
)
from sign_type_classifier.sam3_adapter import Sam3SignDetector  # noqa: E402


DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "sign_type_prompts.yaml"
CLASS_COLORS = {
    label: tuple(int(value) for value in color)
    for label, color in zip(
        CLASS_NAMES,
        (
            (235, 80, 70),
            (235, 165, 45),
            (75, 190, 225),
            (65, 115, 220),
            (175, 90, 220),
            (60, 190, 105),
        ),
    )
}


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prompt_config = Path(args.prompt_config).expanduser().resolve()
    _validate_args(args)
    _prepare_output_dir(
        out_dir,
        source_root=source_root,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    for label in CLASS_NAMES:
        (out_dir / "review" / label).mkdir(parents=True, exist_ok=True)

    prompts_by_label = load_prompt_config(prompt_config)
    labeled_prompts = [
        (label, prompt)
        for label in CLASS_NAMES
        for prompt in prompts_by_label[label]
    ]
    source_records = collect_source_images(source_root, recursive=not args.non_recursive)
    if args.max_images is not None:
        source_records = source_records[: args.max_images]

    run_signature = build_run_signature(
        source_root=source_root,
        prompt_config=prompt_config,
        args=args,
    )
    samples: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    completed_source_ids: set[str] = set()
    if args.resume:
        samples, source_results, completed_source_ids = load_resume_progress(
            out_dir,
            expected_signature=run_signature,
        )
        print(
            f"Resume: {len(completed_source_ids)} completed frames, {len(samples)} samples",
            file=sys.stderr,
            flush=True,
        )

    pending = [
        source
        for source in source_records
        if source_id_from_relative_path(source["source_relative_path"]) not in completed_source_ids
    ]
    detector = None
    if pending:
        detector = Sam3SignDetector(
            sam3_root=args.sam3_root,
            model_dir=args.sam3_model_path,
            min_score=args.min_score,
            use_fa3=args.use_fa3,
            cache_visual_features=args.cache_visual_features,
        )

    feature_names = numeric_feature_names(CLASS_NAMES)
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
        assert detector is not None
        _remove_stale_source_outputs(out_dir, source_id=source_id)
        samples = [item for item in samples if str(item.get("source_id")) != source_id]
        frame_started = time.perf_counter()
        print(f"[{image_index:05d}/{len(source_records):05d}] {source_path}", file=sys.stderr, flush=True)

        image_rgb = read_rgb(source_path)
        if args.colour_correction:
            image_rgb = simple_colour_correction_rgb(image_rgb)
        source_copy = copy_source_image(source_path, out_dir=out_dir, source_id=source_id)
        sam3_started = time.perf_counter()
        detection_kwargs: dict[str, Any] = {"labeled_prompts": labeled_prompts}
        if args.colour_correction:
            detection_kwargs["image_rgb"] = image_rgb
        raw_detections = detector.detect_labeled(source_path, **detection_kwargs)
        accepted_detections, rejected_detections = (
            filter_detections(raw_detections)
            if args.geometry_filters
            else (list(raw_detections), [])
        )
        candidates = cluster_sign_detections(
            accepted_detections,
            class_names=CLASS_NAMES,
            min_mask_size=args.min_mask_size,
            iou_threshold=args.cluster_iou,
            containment_threshold=args.cluster_containment,
        )
        sam3_seconds = time.perf_counter() - sam3_started

        frame_records = []
        for instance_index, candidate in enumerate(candidates):
            sample_id = f"{source_id}_{instance_index:03d}"
            crops = extract_sign_crops(
                image_rgb,
                candidate.mask,
                crop_padding=args.crop_padding,
                context_scale=args.context_scale,
            )
            numeric_features = build_numeric_features(
                image_rgb,
                candidate.mask,
                class_names=CLASS_NAMES,
                class_scores=candidate.class_scores,
                context_scale=args.context_scale,
            )
            sample_dir = out_dir / "samples" / sample_id
            review_path = out_dir / "review" / candidate.label / f"{sample_id}.jpg"
            paths = save_sample_assets(
                sample_dir=sample_dir,
                review_path=review_path,
                image_rgb=image_rgb,
                candidate=candidate,
                crops=crops,
            )
            record = build_sample_record(
                sample_id=sample_id,
                source_id=source_id,
                source_path=source_path,
                source_relative_path=str(source["source_relative_path"]),
                source_copy=source_copy,
                out_dir=out_dir,
                candidate=candidate,
                crops=crops,
                paths=paths,
                numeric_features=numeric_features,
                feature_names=feature_names,
            )
            write_json_atomic(sample_dir / "metadata.json", record)
            frame_records.append(record)

        label_counts = Counter(item.label for item in candidates)
        source_result = {
            "source_id": source_id,
            "source_path": str(source_path),
            "source_relative_path": source["source_relative_path"],
            "source_image": str(source_copy.relative_to(out_dir)),
            "image_size": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
            "colour_correction": bool(args.colour_correction),
            "raw_detections": len(raw_detections),
            "accepted_detections": len(accepted_detections),
            "rejected_detections": len(rejected_detections),
            "detection_rejections": [rejection_to_json(item) for item in rejected_detections],
            "candidate_instances": len(candidates),
            "class_counts": dict(sorted(label_counts.items())),
            "timing_seconds": {
                "sam3": round(sam3_seconds, 6),
                "total": round(time.perf_counter() - frame_started, 6),
            },
        }
        samples.extend(frame_records)
        source_results = [item for item in source_results if str(item.get("source_id")) != source_id]
        source_results.append(source_result)
        completed_source_ids.add(source_id)
        checkpoint_progress(
            out_dir=out_dir,
            samples=samples,
            source_results=source_results,
            completed_source_ids=completed_source_ids,
            run_signature=run_signature,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            status="running",
        )

    samples = assign_grouped_splits(
        samples,
        class_names=CLASS_NAMES,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_sample_metadata(out_dir, samples)
    write_jsonl_atomic(out_dir / "manifest.jsonl", samples)
    summary = summarize_manifest(samples)
    dataset_manifest = {
        "version": 1,
        "mode": "sam3_sign_type_autosort",
        "source_root": str(source_root),
        "out_dir": str(out_dir),
        "class_names": list(CLASS_NAMES),
        "prompt_config": str(prompt_config),
        "prompts": prompts_by_label,
        "run_signature": run_signature,
        "sam3_root": str(Path(args.sam3_root).expanduser().resolve()),
        "sam3_model_path": str(Path(args.sam3_model_path).expanduser().resolve()),
        "cache_visual_features": bool(args.cache_visual_features),
        "colour_correction": bool(args.colour_correction),
        "geometry_filters": bool(args.geometry_filters),
        "geometry_filter_rules": GEOMETRY_FILTER_RULES,
        "size_filter_rules": SIZE_FILTER_RULES,
        "min_score": float(args.min_score),
        "min_mask_size": int(args.min_mask_size),
        "cluster_iou": float(args.cluster_iou),
        "cluster_containment": float(args.cluster_containment),
        "crop_padding": float(args.crop_padding),
        "context_scale": float(args.context_scale),
        "numeric_feature_names": list(feature_names),
        "split": split_metadata(seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio),
        "manual_review": {
            "folders": [f"review/{label}" for label in CLASS_NAMES],
            "instructions": "Move review JPGs between class folders or delete false positives, then run reindex_dataset.py.",
        },
        "summary": summary,
        "sources": source_results,
        "completed_frames": len(completed_source_ids),
        "manifest_jsonl": str(out_dir / "manifest.jsonl"),
        "generation_state": str(out_dir / "generation_state.json"),
    }
    write_json_atomic(out_dir / "dataset_manifest.json", dataset_manifest)
    write_generation_state(
        out_dir=out_dir,
        source_results=source_results,
        completed_source_ids=completed_source_ids,
        samples=len(samples),
        run_signature=run_signature,
        status="complete",
    )
    print(json.dumps({"manifest": str(out_dir / "dataset_manifest.json"), **summary}, ensure_ascii=False, indent=2))


def load_prompt_config(path: str | Path) -> dict[str, list[str]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Prompt config does not exist: {config_path}")
    groups: dict[str, list[str]] = {}
    current = None
    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            current = stripped[:-1].strip()
            if current in groups:
                raise ValueError(f"Duplicate prompt group {current!r} at {config_path}:{line_number}")
            groups[current] = []
        elif indent >= 2 and stripped.startswith("- ") and current is not None:
            value = stripped[2:].strip()
            if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            if value:
                groups[current].append(value)
        else:
            raise ValueError(f"Unsupported prompt YAML syntax at {config_path}:{line_number}")
    missing = [label for label in CLASS_NAMES if not groups.get(label)]
    extra = sorted(set(groups) - set(CLASS_NAMES))
    if missing or extra:
        raise ValueError(f"Prompt config must contain exactly {CLASS_NAMES}; missing={missing}, extra={extra}")
    return {label: groups[label] for label in CLASS_NAMES}


def build_run_signature(*, source_root: Path, prompt_config: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_root": str(source_root),
        "prompt_sha256": hashlib.sha256(prompt_config.read_bytes()).hexdigest(),
        "class_names": list(CLASS_NAMES),
        "recursive": not bool(args.non_recursive),
        "min_score": float(args.min_score),
        "min_mask_size": int(args.min_mask_size),
        "cluster_iou": float(args.cluster_iou),
        "cluster_containment": float(args.cluster_containment),
        "crop_padding": float(args.crop_padding),
        "context_scale": float(args.context_scale),
        "sam3_model_path": str(Path(args.sam3_model_path).expanduser().resolve()),
        "colour_correction": bool(getattr(args, "colour_correction", False)),
        "geometry_filters": bool(getattr(args, "geometry_filters", True)),
    }


def build_sample_record(
    *,
    sample_id: str,
    source_id: str,
    source_path: Path,
    source_relative_path: str,
    source_copy: Path,
    out_dir: Path,
    candidate: SignCandidate,
    crops: SignCrops,
    paths: dict[str, Path],
    numeric_features: dict[str, float],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    canonical = candidate.canonical_detection
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "label": candidate.label,
        "initial_label": candidate.label,
        "label_source": "sam3_prompt_top1",
        "source_path": str(source_path),
        "source_relative_path": source_relative_path,
        "source_image": str(source_copy.relative_to(out_dir)),
        "rgb_crop": str(paths["rgb"].relative_to(out_dir)),
        "mask": str(paths["mask"].relative_to(out_dir)),
        "masked_rgb": str(paths["masked"].relative_to(out_dir)),
        "context_rgb": str(paths["context"].relative_to(out_dir)),
        "context_mask": str(paths["context_mask"].relative_to(out_dir)),
        "review_image": str(paths["review"].relative_to(out_dir)),
        "preview": str(paths["preview"].relative_to(out_dir)),
        "metadata": str((paths["rgb"].parent / "metadata.json").relative_to(out_dir)),
        "sam3_score": float(candidate.score),
        "sam3_margin": float(candidate.margin),
        "sam3_class_scores": {label: float(candidate.class_scores[label]) for label in CLASS_NAMES},
        "sam3_prompt": canonical.prompt,
        "sam3_raw_box": None if canonical.box is None else np.asarray(canonical.box, dtype=np.float32).tolist(),
        "sam3_cluster": [detection_to_json(item) for item in candidate.detections],
        "mask_bbox_xyxy": list(crops.mask_bbox_xyxy),
        "crop_bbox_xyxy": list(crops.tight_bbox_xyxy),
        "context_bbox_xyxy": list(crops.context_bbox_xyxy),
        "mask_pixels": int(np.count_nonzero(candidate.mask)),
        "numeric_features": {name: float(numeric_features[name]) for name in feature_names},
        "numeric_feature_vector": [float(numeric_features[name]) for name in feature_names],
    }


def detection_to_json(detection: Any) -> dict[str, Any]:
    return {
        "label": str(detection.label),
        "prompt": str(detection.prompt),
        "score": float(detection.score),
        "box": None if detection.box is None else np.asarray(detection.box, dtype=np.float32).tolist(),
        "mask_pixels": int(np.count_nonzero(detection.mask)),
    }


def save_sample_assets(
    *,
    sample_dir: Path,
    review_path: Path,
    image_rgb: np.ndarray,
    candidate: SignCandidate,
    crops: SignCrops,
) -> dict[str, Path]:
    from PIL import Image  # noqa: WPS433

    sample_dir.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb": sample_dir / "rgb.png",
        "mask": sample_dir / "mask.png",
        "masked": sample_dir / "masked_rgb.png",
        "context": sample_dir / "context_rgb.png",
        "context_mask": sample_dir / "context_mask.png",
        "preview": sample_dir / "preview.jpg",
        "review": review_path,
    }
    Image.fromarray(crops.rgb, mode="RGB").save(paths["rgb"])
    Image.fromarray(crops.mask.astype(np.uint8) * 255, mode="L").save(paths["mask"])
    Image.fromarray(crops.masked_rgb, mode="RGB").save(paths["masked"])
    Image.fromarray(crops.context_rgb, mode="RGB").save(paths["context"])
    Image.fromarray(crops.context_mask.astype(np.uint8) * 255, mode="L").save(paths["context_mask"])
    preview = make_review_preview(image_rgb=image_rgb, candidate=candidate, crops=crops)
    preview.save(paths["preview"], quality=95)
    preview.save(paths["review"], quality=95)
    return paths


def make_review_preview(*, image_rgb: np.ndarray, candidate: SignCandidate, crops: SignCrops) -> Any:
    from PIL import Image, ImageDraw, ImageOps  # noqa: WPS433

    color = CLASS_COLORS[candidate.label]
    full = Image.fromarray(image_rgb, mode="RGB")
    full_draw = ImageDraw.Draw(full)
    full_draw.rectangle(crops.mask_bbox_xyxy, outline=color, width=max(2, full.width // 800))
    context_overlay = crops.context_rgb.copy()
    context_overlay[crops.context_mask] = (
        context_overlay[crops.context_mask].astype(np.float32) * 0.55
        + np.asarray(color, dtype=np.float32) * 0.45
    ).astype(np.uint8)

    panels = [
        _fit_panel(full, size=(720, 430), label="full image"),
        _fit_panel(Image.fromarray(context_overlay), size=(520, 430), label="context + mask"),
        _fit_panel(Image.fromarray(crops.masked_rgb), size=(420, 430), label="masked object"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), 490), color=(28, 28, 28))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 60))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    title = f"{candidate.label}  score={candidate.score:.3f}  margin={candidate.margin:.3f}"
    draw.rectangle((0, 0, canvas.width, 60), fill=(18, 18, 18))
    draw.text((16, 18), title, fill=color)
    return canvas


def _fit_panel(image: Any, *, size: tuple[int, int], label: str) -> Any:
    from PIL import Image, ImageDraw, ImageOps  # noqa: WPS433

    contained = ImageOps.contain(image, (size[0], size[1] - 28), method=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", size, color=(40, 40, 40))
    panel.paste(contained, ((size[0] - contained.width) // 2, 28 + (size[1] - 28 - contained.height) // 2))
    ImageDraw.Draw(panel).text((8, 7), label, fill=(235, 235, 235))
    return panel


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: WPS433

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def copy_source_image(source_path: Path, *, out_dir: Path, source_id: str) -> Path:
    suffix = source_path.suffix.lower() if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".png"
    target = out_dir / "images" / f"{source_id}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return target


def checkpoint_progress(
    *,
    out_dir: Path,
    samples: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    completed_source_ids: set[str],
    run_signature: dict[str, Any],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    status: str,
) -> None:
    records = assign_grouped_splits(
        samples,
        class_names=CLASS_NAMES,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    write_sample_metadata(out_dir, records)
    write_jsonl_atomic(out_dir / "manifest.jsonl", records)
    write_generation_state(
        out_dir=out_dir,
        source_results=source_results,
        completed_source_ids=completed_source_ids,
        samples=len(records),
        run_signature=run_signature,
        status=status,
    )


def load_resume_progress(
    out_dir: Path,
    *,
    expected_signature: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    state = read_json_object(out_dir / "generation_state.json")
    if not state:
        raise FileNotFoundError(f"Cannot resume without {out_dir / 'generation_state.json'}")
    if state.get("run_signature") != expected_signature:
        raise ValueError("Cannot resume: source, prompts, model, or dataset parameters differ from generation_state.json")
    completed = {str(value) for value in state.get("completed_source_ids", [])}
    manifest_path = out_dir / "manifest.jsonl"
    records = load_jsonl(manifest_path) if manifest_path.is_file() else []
    records = [item for item in records if str(item.get("source_id")) in completed]
    sources = [
        dict(item)
        for item in state.get("sources", [])
        if isinstance(item, dict) and str(item.get("source_id")) in completed
    ]
    return records, sources, completed


def write_generation_state(
    *,
    out_dir: Path,
    source_results: list[dict[str, Any]],
    completed_source_ids: set[str],
    samples: int,
    run_signature: dict[str, Any],
    status: str,
) -> None:
    write_json_atomic(
        out_dir / "generation_state.json",
        {
            "version": 1,
            "status": status,
            "run_signature": run_signature,
            "completed_frames": len(completed_source_ids),
            "completed_source_ids": sorted(completed_source_ids),
            "samples": int(samples),
            "sources": source_results,
        },
    )


def write_sample_metadata(out_dir: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        write_json_atomic(out_dir / str(record["metadata"]), record)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, records)
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def split_metadata(*, seed: int, train_ratio: float, val_ratio: float) -> dict[str, Any]:
    return {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "seed": int(seed),
        "grouped_by": "source_id",
    }


def _remove_stale_source_outputs(out_dir: Path, *, source_id: str) -> None:
    samples_dir = out_dir / "samples"
    if samples_dir.is_dir():
        for path in samples_dir.glob(f"{source_id}_*"):
            if path.is_dir():
                shutil.rmtree(path)
    for label in CLASS_NAMES:
        review_dir = out_dir / "review" / label
        if review_dir.is_dir():
            for path in review_dir.glob(f"{source_id}_*.jpg"):
                path.unlink()


def _prepare_output_dir(out_dir: Path, *, source_root: Path, overwrite: bool, resume: bool) -> None:
    if out_dir == source_root or out_dir in source_root.parents or source_root in out_dir.parents:
        raise ValueError("--out-dir and --source-root must be disjoint directories")
    if resume:
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume because output directory does not exist: {out_dir}")
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}; use --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError(f"--max-images must be positive, got {args.max_images}")
    if args.min_mask_size <= 0:
        raise ValueError(f"--min-mask-size must be positive, got {args.min_mask_size}")
    for name in ("min_score", "cluster_iou", "cluster_containment"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1], got {value}")
    if not 0.0 <= args.crop_padding <= 1.0:
        raise ValueError(f"--crop-padding must be in [0,1], got {args.crop_padding}")
    if args.context_scale < 1.0:
        raise ValueError(f"--context-scale must be >=1, got {args.context_scale}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 sign prompts on images and build six manually reviewable class folders."
    )
    parser.add_argument("--source-root", required=True, help="Directory with source images.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--sam3-model-path", required=True)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--min-mask-size", type=int, default=900)
    parser.add_argument("--cluster-iou", type=float, default=0.55)
    parser.add_argument("--cluster-containment", type=float, default=0.80)
    parser.add_argument("--crop-padding", type=float, default=0.12)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument(
        "--cache-visual-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse SAM3 image-backbone features across prompts (enabled by default).",
    )
    parser.add_argument(
        "--colour-correction",
        "--color-correction",
        dest="colour_correction",
        action="store_true",
        help=(
            "Apply OpenCV SimpleWB (P=0.5) and gray-world correction in memory "
            "before SAM3; no corrected full-frame image is saved."
        ),
    )
    parser.add_argument(
        "--geometry-filters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter detections with label-specific normalized XYWH geometry rules.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    main()
