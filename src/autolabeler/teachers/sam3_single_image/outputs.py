"""Persistence, validation, and visualization for SAM3 frame results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .types import IMAGE_SUFFIXES, SIGN_TYPE_LABELS, ImageResult, Sam3Instance


def save_image_outputs(
    *,
    image: Any,
    image_path: Path,
    output_dir: Path,
    instances: list[Sam3Instance],
    overlay_instances: list[Sam3Instance] | None = None,
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
) -> None:
    from PIL import Image  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic_mask.npy", semantic_mask)
    np.save(output_dir / "confidence.npy", confidence)
    save_instances_npz(
        output_dir / "instances.npz", instances, shape=semantic_mask.shape
    )

    image_np = np.asarray(image, dtype=np.uint8)
    visible_instances = instances if overlay_instances is None else overlay_instances
    overlay = make_overlay(
        image_np,
        semantic_mask,
        instances=visible_instances,
        confidence=confidence,
    )
    semantic_color = make_semantic_color(semantic_mask)
    Image.fromarray(overlay).save(output_dir / "overlay.jpg", quality=95)
    Image.fromarray(semantic_color).save(output_dir / "semantic_color.png")
    Image.fromarray(make_preview(image_np, semantic_color, overlay)).save(
        output_dir / "preview.jpg", quality=95
    )

    image.save(output_dir / "image.jpg", quality=95)

    instances_json = [instance_to_log_json(item) for item in instances]
    (output_dir / "instances.json").write_text(
        json.dumps(instances_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "source_image.txt").write_text(str(image_path), encoding="utf-8")


def instance_to_log_json(item: Sam3Instance) -> dict[str, Any]:
    return {
        "label": item.label,
        "class_id": int(item.class_id),
        "prompt": item.prompt,
        "score": float(item.score),
        "box": None
        if item.box is None
        else np.asarray(item.box, dtype=np.float32).tolist(),
        "visible_pixel_count": int(np.count_nonzero(item.mask)),
        "source_label": item.source_label,
        "orientation_label": item.orientation_label,
        "orientation_score": item.orientation_score,
        "orientation_margin": item.orientation_margin,
        "orientation_probabilities": None
        if item.orientation_probabilities is None
        else list(item.orientation_probabilities),
        "orientation_fallback": bool(item.orientation_fallback),
        "orientation_fallback_reason": item.orientation_fallback_reason,
        "sign_classifier_label": item.sign_classifier_label,
        "sign_classifier_assigned_label": item.sign_classifier_assigned_label,
        "sign_classifier_confidence": item.sign_classifier_confidence,
        "sign_classifier_margin": item.sign_classifier_margin,
        "sign_classifier_probabilities": None
        if item.sign_classifier_probabilities is None
        else dict(item.sign_classifier_probabilities),
        "sign_classifier_image_probabilities": None
        if item.sign_classifier_image_probabilities is None
        else dict(item.sign_classifier_image_probabilities),
        "sign_classifier_numeric_probabilities": None
        if item.sign_classifier_numeric_probabilities is None
        else dict(item.sign_classifier_numeric_probabilities),
        "sign_sam3_class_scores": None
        if item.sign_sam3_class_scores is None
        else dict(item.sign_sam3_class_scores),
        "sign_classifier_accepted": bool(item.sign_classifier_accepted),
        "sign_classifier_fallback_reason": item.sign_classifier_fallback_reason,
    }


def build_classes_log(
    *,
    result: ImageResult,
    prompt_config: Path,
    min_score: float,
    processing_time_seconds: float,
) -> dict[str, Any]:
    visible_instances = [
        instance_to_log_json(item) for item in result.overlay_instances
    ]
    return {
        "version": 2,
        "image_path": str(result.image_path),
        "prompt_config": str(prompt_config),
        "min_score": float(min_score),
        "object_count": len(visible_instances),
        "class_list": make_class_list(result.class_pixel_counts),
        "class_pixel_counts": result.class_pixel_counts,
        "instances": visible_instances,
        "processing_time_seconds": round(processing_time_seconds, 6),
    }


def save_instances_npz(
    path: Path, instances: list[Sam3Instance], *, shape: tuple[int, int]
) -> None:
    height, width = shape
    if instances:
        payload: dict[str, np.ndarray] = {
            "masks": np.stack([item.mask for item in instances], axis=0).astype(
                bool, copy=False
            ),
            "scores": np.asarray([item.score for item in instances], dtype=np.float32),
            "labels": np.asarray([item.label for item in instances]),
            "class_ids": np.asarray(
                [item.class_id for item in instances], dtype=np.uint16
            ),
            "prompts": np.asarray([item.prompt for item in instances]),
            "source_labels": np.asarray(
                [item.source_label or "" for item in instances]
            ),
            "orientation_labels": np.asarray(
                [item.orientation_label or "" for item in instances]
            ),
            "orientation_scores": np.asarray(
                [
                    np.nan if item.orientation_score is None else item.orientation_score
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "orientation_margins": np.asarray(
                [
                    np.nan
                    if item.orientation_margin is None
                    else item.orientation_margin
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "orientation_probabilities": np.asarray(
                [
                    (np.nan, np.nan, np.nan)
                    if item.orientation_probabilities is None
                    else item.orientation_probabilities
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "orientation_fallback": np.asarray(
                [item.orientation_fallback for item in instances],
                dtype=bool,
            ),
            "orientation_fallback_reasons": np.asarray(
                [item.orientation_fallback_reason or "" for item in instances]
            ),
            "sign_classifier_labels": np.asarray(
                [item.sign_classifier_label or "" for item in instances]
            ),
            "sign_classifier_assigned_labels": np.asarray(
                [item.sign_classifier_assigned_label or "" for item in instances]
            ),
            "sign_classifier_confidences": np.asarray(
                [
                    np.nan
                    if item.sign_classifier_confidence is None
                    else item.sign_classifier_confidence
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "sign_classifier_margins": np.asarray(
                [
                    np.nan
                    if item.sign_classifier_margin is None
                    else item.sign_classifier_margin
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "sign_classifier_probabilities": np.asarray(
                [
                    (np.nan,) * (len(SIGN_TYPE_LABELS) + 1)
                    if item.sign_classifier_probabilities is None
                    else tuple(value for _, value in item.sign_classifier_probabilities)
                    for item in instances
                ],
                dtype=np.float32,
            ),
            "sign_classifier_accepted": np.asarray(
                [item.sign_classifier_accepted for item in instances],
                dtype=bool,
            ),
            "sign_classifier_fallback_reasons": np.asarray(
                [item.sign_classifier_fallback_reason or "" for item in instances]
            ),
        }
        if all(item.box is not None for item in instances):
            payload["boxes"] = np.stack(
                [np.asarray(item.box, dtype=np.float32) for item in instances], axis=0
            )
    else:
        payload = {
            "masks": np.zeros((0, height, width), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.asarray([], dtype=str),
            "class_ids": np.zeros((0,), dtype=np.uint16),
            "prompts": np.asarray([], dtype=str),
            "source_labels": np.asarray([], dtype=str),
            "orientation_labels": np.asarray([], dtype=str),
            "orientation_scores": np.zeros((0,), dtype=np.float32),
            "orientation_margins": np.zeros((0,), dtype=np.float32),
            "orientation_probabilities": np.zeros((0, 3), dtype=np.float32),
            "orientation_fallback": np.zeros((0,), dtype=bool),
            "orientation_fallback_reasons": np.asarray([], dtype=str),
            "sign_classifier_labels": np.asarray([], dtype=str),
            "sign_classifier_assigned_labels": np.asarray([], dtype=str),
            "sign_classifier_confidences": np.zeros((0,), dtype=np.float32),
            "sign_classifier_margins": np.zeros((0,), dtype=np.float32),
            "sign_classifier_probabilities": np.zeros(
                (0, len(SIGN_TYPE_LABELS) + 1),
                dtype=np.float32,
            ),
            "sign_classifier_accepted": np.zeros((0,), dtype=bool),
            "sign_classifier_fallback_reasons": np.asarray([], dtype=str),
        }
    np.savez_compressed(path, **payload)


def make_overlay(
    image_np: np.ndarray,
    semantic_mask: np.ndarray,
    *,
    alpha: float = 0.45,
    instances: list[Sam3Instance] | None = None,
    confidence: np.ndarray | None = None,
    min_label_area: int = 64,
) -> np.ndarray:
    overlay = image_np.copy()
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color = color_for_class(class_id)
        mask = semantic_mask == class_id
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
    if instances:
        overlay = drew_instance_labels(
            overlay=overlay,
            semantic_mask=semantic_mask,
            instances=instances,
            confidence=confidence,
            min_label_area=min_label_area,
        )
    return overlay


def drew_instance_labels(
    overlay: np.ndarray,
    *,
    semantic_mask: np.ndarray,
    instances: list[Sam3Instance],
    confidence: np.ndarray | None,
    min_label_area: int,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont  # noqa: WPS433

    image = Image.fromarray(np.asarray(overlay, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("DejaVuSans.ttf", size=24)
    height, width = semantic_mask.shape

    for item in sorted(instances, key=lambda value: value.score, reverse=True):
        visible = item.mask & (semantic_mask == item.class_id)
        if confidence is not None:
            visible &= np.isclose(confidence, np.float32(item.score), atol=1e-6)
        if int(np.count_nonzero(visible)) < min_label_area:
            continue

        text = f"{item.label} {item.score:.2f}"
        label_x, label_y = label_anchor(
            visible, image_size=(width, height), text=text, draw=draw, font=font
        )
        color = color_for_class(item.class_id)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = 5
        box = [
            label_x,
            label_y,
            label_x + text_width + padding * 2,
            label_y + text_height + padding * 2,
        ]
        draw.rectangle(
            box,
            fill=(int(color[0]), int(color[1]), int(color[2]), 220),
            outline=(0, 0, 0, 220),
        )
        draw.text(
            (label_x + padding, label_y + padding),
            text,
            font=font,
            fill=contrast_text_color(color),
        )
    return np.asarray(image, dtype=np.uint8)


def label_anchor(
    visible: np.ndarray, *, image_size: tuple[int, int], text: str, draw: Any, font: Any
) -> tuple[int, int]:
    width, height = image_size
    ys, xs = np.nonzero(visible)
    center_x = int(np.median(xs))
    center_y = int(np.median(ys))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0] + 8
    text_height = text_bbox[3] - text_bbox[1] + 8
    label_x = min(max(center_x - text_width // 2, 0), max(width - text_width - 1, 0))
    label_y = min(max(center_y - text_height // 2, 0), max(height - text_height - 1, 0))
    return int(label_x), int(label_y)


def contrast_text_color(color: np.ndarray) -> tuple[int, int, int, int]:
    luminance = (
        0.2126 * float(color[0]) + 0.7152 * float(color[1]) + 0.0722 * float(color[2])
    )
    return (0, 0, 0, 255) if luminance > 145.0 else (255, 255, 255, 255)


def make_semantic_color(semantic_mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    for class_id in sorted(int(x) for x in np.unique(semantic_mask) if int(x) != 0):
        color[semantic_mask == class_id] = color_for_class(class_id)
    return color


def make_preview(
    image_np: np.ndarray, semantic_color: np.ndarray, overlay: np.ndarray
) -> np.ndarray:
    separator = np.full((image_np.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate(
        [image_np, separator, semantic_color, separator, overlay], axis=1
    )


def color_for_class(class_id: int) -> np.ndarray:
    rng = np.random.default_rng(class_id * 1009 + 17)
    return rng.integers(40, 240, size=3, dtype=np.uint8)


def collect_images(image_dir: Path, *, recursive: bool = False) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def output_dir_for_image(
    out_dir: Path, image_dir: Path, image_path: Path, *, recursive: bool
) -> Path:
    if not recursive:
        return out_dir / image_path.stem
    rel = image_path.relative_to(image_dir)
    return out_dir / rel.with_suffix("")


def outputs_exist(
    frame_out: Path,
    *,
    classes_log_enabled: bool = False,
) -> bool:
    required = [
        frame_out / "semantic_mask.npy",
        frame_out / "confidence.npy",
        frame_out / "instances.npz",
        frame_out / "overlay.jpg",
        frame_out / "metadata.json",
    ]
    if classes_log_enabled:
        required.append(frame_out / "classes_log.json")
    return all(path.is_file() for path in required)


def validate_outputs(frame_out: Path) -> None:
    semantic = np.load(frame_out / "semantic_mask.npy")
    confidence = np.load(frame_out / "confidence.npy")
    if semantic.shape != confidence.shape:
        raise ValueError(
            f"semantic/confidence shape mismatch in {frame_out}: {semantic.shape} vs {confidence.shape}"
        )
    with np.load(frame_out / "instances.npz", allow_pickle=False) as data:
        masks = np.asarray(data["masks"])
        scores = np.asarray(data["scores"])
        labels = np.asarray(data["labels"])
        class_ids = np.asarray(data["class_ids"])
        prompts = np.asarray(data["prompts"])
        orientation_probabilities = np.asarray(data["orientation_probabilities"])
        orientation_scores = np.asarray(data["orientation_scores"])
        orientation_fallback = np.asarray(data["orientation_fallback"])
        sign_probabilities = np.asarray(data["sign_classifier_probabilities"])
        sign_confidences = np.asarray(data["sign_classifier_confidences"])
        sign_accepted = np.asarray(data["sign_classifier_accepted"])
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(
            f"instances masks must be bool[N,H,W], got {masks.shape} {masks.dtype}"
        )
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"instances scores mismatch in {frame_out}")
    if (
        labels.shape != (masks.shape[0],)
        or class_ids.shape != (masks.shape[0],)
        or prompts.shape != (masks.shape[0],)
    ):
        raise ValueError(f"instances metadata mismatch in {frame_out}")
    if orientation_probabilities.shape != (masks.shape[0], 3):
        raise ValueError(f"instances orientation probabilities mismatch in {frame_out}")
    if orientation_scores.shape != (masks.shape[0],) or orientation_fallback.shape != (
        masks.shape[0],
    ):
        raise ValueError(f"instances orientation metadata mismatch in {frame_out}")
    if sign_probabilities.shape != (masks.shape[0], len(SIGN_TYPE_LABELS) + 1):
        raise ValueError(f"instances sign probabilities mismatch in {frame_out}")
    if sign_confidences.shape != (masks.shape[0],) or sign_accepted.shape != (
        masks.shape[0],
    ):
        raise ValueError(f"instances sign metadata mismatch in {frame_out}")


def make_class_list(class_pixel_counts):
    return list(class_pixel_counts.keys())


def classes_from_semantic_mask(
    path: Path,
    *,
    label_to_id: Mapping[str, int],
    min_mask_size: int,
) -> set[str]:
    semantic_mask = np.load(path)
    classes: set[str] = set()
    for label, class_id in label_to_id.items():
        if int(np.count_nonzero(semantic_mask == int(class_id))) >= min_mask_size:
            classes.add(label)
    return classes
