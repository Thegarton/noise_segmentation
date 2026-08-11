"""Context-preserving crops and fixed numeric features for sign instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


HUE_BINS = 8
SATURATION_BINS = 4
VALUE_BINS = 4
GEOMETRY_FEATURE_NAMES = (
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_bottom",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "mask_centroid_x",
    "mask_centroid_y",
    "mask_area",
    "mask_bbox_fill_ratio",
    "mask_compactness",
    "touches_left",
    "touches_top",
    "touches_right",
    "touches_bottom",
    "distance_to_center",
    "distance_to_left",
    "distance_to_top",
    "distance_to_right",
    "distance_to_bottom",
)


@dataclass(frozen=True)
class SignCrops:
    mask_bbox_xyxy: tuple[int, int, int, int]
    tight_bbox_xyxy: tuple[int, int, int, int]
    context_bbox_xyxy: tuple[int, int, int, int]
    rgb: np.ndarray
    mask: np.ndarray
    masked_rgb: np.ndarray
    context_rgb: np.ndarray
    context_mask: np.ndarray


def extract_sign_crops(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    crop_padding: float = 0.12,
    context_scale: float = 3.0,
    background: int = 127,
) -> SignCrops:
    image = _validate_image(image_rgb)
    binary = _validate_mask(mask, shape=image.shape[:2])
    if not 0.0 <= float(crop_padding) <= 1.0:
        raise ValueError(f"crop_padding must be in [0,1], got {crop_padding}")
    if not np.isfinite(context_scale) or context_scale < 1.0:
        raise ValueError(f"context_scale must be finite and >=1, got {context_scale}")

    mask_bbox = mask_bbox_xyxy(binary)
    tight_bbox = _pad_bbox(mask_bbox, image.shape[:2], padding=float(crop_padding))
    context_bbox = _scale_bbox(mask_bbox, image.shape[:2], scale=float(context_scale))
    x0, y0, x1, y1 = tight_bbox
    rgb = image[y0:y1, x0:x1].copy()
    crop_mask = binary[y0:y1, x0:x1].copy()
    masked = np.full_like(rgb, np.uint8(background))
    masked[crop_mask] = rgb[crop_mask]

    cx0, cy0, cx1, cy1 = context_bbox
    context_rgb = image[cy0:cy1, cx0:cx1].copy()
    context_mask = binary[cy0:cy1, cx0:cx1].copy()
    return SignCrops(
        mask_bbox_xyxy=mask_bbox,
        tight_bbox_xyxy=tight_bbox,
        context_bbox_xyxy=context_bbox,
        rgb=rgb,
        mask=crop_mask,
        masked_rgb=masked,
        context_rgb=context_rgb,
        context_mask=context_mask,
    )


def build_numeric_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    class_names: Sequence[str],
    class_scores: dict[str, float],
    context_scale: float = 3.0,
) -> dict[str, float]:
    image = _validate_image(image_rgb)
    binary = _validate_mask(mask, shape=image.shape[:2])
    names = tuple(str(value) for value in class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")

    height, width = image.shape[:2]
    x0, y0, x1, y1 = mask_bbox_xyxy(binary)
    box_width = x1 - x0
    box_height = y1 - y0
    box_area = box_width * box_height
    rows, columns = np.nonzero(binary)
    mask_area = int(rows.size)
    center_x = (x0 + x1) * 0.5 / width
    center_y = (y0 + y1) * 0.5 / height
    centroid_x = float(columns.mean()) / width
    centroid_y = float(rows.mean()) / height
    perimeter = _binary_perimeter(binary)
    compactness = 4.0 * np.pi * mask_area / (perimeter * perimeter) if perimeter > 0.0 else 0.0

    features: dict[str, float] = {
        "bbox_x0": x0 / width,
        "bbox_y0": y0 / height,
        "bbox_x1": x1 / width,
        "bbox_y1": y1 / height,
        "bbox_center_x": center_x,
        "bbox_center_y": center_y,
        "bbox_bottom": y1 / height,
        "bbox_width": box_width / width,
        "bbox_height": box_height / height,
        "bbox_area": box_area / float(width * height),
        "bbox_aspect_ratio": box_width / float(max(box_height, 1)),
        "mask_centroid_x": centroid_x,
        "mask_centroid_y": centroid_y,
        "mask_area": mask_area / float(width * height),
        "mask_bbox_fill_ratio": mask_area / float(max(box_area, 1)),
        "mask_compactness": float(compactness),
        "touches_left": float(x0 == 0),
        "touches_top": float(y0 == 0),
        "touches_right": float(x1 == width),
        "touches_bottom": float(y1 == height),
        "distance_to_center": float(np.hypot(center_x - 0.5, center_y - 0.5)),
        "distance_to_left": x0 / width,
        "distance_to_top": y0 / height,
        "distance_to_right": (width - x1) / width,
        "distance_to_bottom": (height - y1) / height,
    }

    score_values = []
    for label in names:
        score = float(class_scores.get(label, 0.0))
        if not np.isfinite(score):
            raise ValueError(f"Non-finite class score for {label!r}: {score}")
        features[f"sam_score__{label}"] = score
        score_values.append(score)
    score_values.sort(reverse=True)
    features["sam_top1_score"] = score_values[0]
    features["sam_top1_margin"] = score_values[0] - score_values[1] if len(score_values) > 1 else score_values[0]

    if not np.isfinite(context_scale) or context_scale < 1.0:
        raise ValueError(f"context_scale must be finite and >=1, got {context_scale}")
    context_bbox = _scale_bbox((x0, y0, x1, y1), (height, width), scale=float(context_scale))
    cx0, cy0, cx1, cy1 = context_bbox
    context_region = np.zeros((height, width), dtype=bool)
    context_region[cy0:cy1, cx0:cx1] = True
    context_ring = context_region & ~binary
    cv2 = _import_cv2()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200) > 0
    features.update(_appearance_features(image, hsv, gray, edges, binary, prefix="object"))
    features.update(_appearance_features(image, hsv, gray, edges, context_ring, prefix="context"))
    return features


def numeric_feature_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")
    return (
        *GEOMETRY_FEATURE_NAMES,
        *(f"sam_score__{label}" for label in names),
        "sam_top1_score",
        "sam_top1_margin",
        *_appearance_feature_names("object"),
        *_appearance_feature_names("context"),
    )


def mask_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must have shape [H,W], got {binary.shape}")
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("Cannot compute bbox for an empty mask")
    return int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1


def _appearance_features(
    image_rgb: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    region: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    region_mask = np.asarray(region, dtype=bool)
    pixels = image_rgb[region_mask]
    keys: dict[str, float] = {}
    if pixels.size == 0:
        for channel in ("r", "g", "b"):
            keys[f"{prefix}_rgb_mean_{channel}"] = 0.0
            keys[f"{prefix}_rgb_std_{channel}"] = 0.0
        for index in range(HUE_BINS):
            keys[f"{prefix}_hue_hist_{index}"] = 0.0
        for index in range(SATURATION_BINS):
            keys[f"{prefix}_saturation_hist_{index}"] = 0.0
        for index in range(VALUE_BINS):
            keys[f"{prefix}_value_hist_{index}"] = 0.0
        keys[f"{prefix}_gray_mean"] = 0.0
        keys[f"{prefix}_gray_std"] = 0.0
        keys[f"{prefix}_edge_density"] = 0.0
        return keys

    normalized = pixels.astype(np.float32) / 255.0
    for index, channel in enumerate(("r", "g", "b")):
        keys[f"{prefix}_rgb_mean_{channel}"] = float(normalized[:, index].mean())
        keys[f"{prefix}_rgb_std_{channel}"] = float(normalized[:, index].std())

    hsv_pixels = hsv[region_mask]
    for channel, bins, upper, name in (
        (0, HUE_BINS, 180.0, "hue"),
        (1, SATURATION_BINS, 256.0, "saturation"),
        (2, VALUE_BINS, 256.0, "value"),
    ):
        histogram, _ = np.histogram(hsv_pixels[:, channel], bins=bins, range=(0.0, upper))
        histogram = histogram.astype(np.float64) / max(int(histogram.sum()), 1)
        for index, value in enumerate(histogram):
            keys[f"{prefix}_{name}_hist_{index}"] = float(value)

    gray_pixels = gray[region_mask].astype(np.float32) / 255.0
    keys[f"{prefix}_gray_mean"] = float(gray_pixels.mean())
    keys[f"{prefix}_gray_std"] = float(gray_pixels.std())
    keys[f"{prefix}_edge_density"] = float(np.count_nonzero(edges & region_mask)) / float(np.count_nonzero(region_mask))
    return keys


def _appearance_feature_names(prefix: str) -> tuple[str, ...]:
    return (
        *(name for channel in ("r", "g", "b") for name in (
            f"{prefix}_rgb_mean_{channel}",
            f"{prefix}_rgb_std_{channel}",
        )),
        *(f"{prefix}_hue_hist_{index}" for index in range(HUE_BINS)),
        *(f"{prefix}_saturation_hist_{index}" for index in range(SATURATION_BINS)),
        *(f"{prefix}_value_hist_{index}" for index in range(VALUE_BINS)),
        f"{prefix}_gray_mean",
        f"{prefix}_gray_std",
        f"{prefix}_edge_density",
    )


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    padding: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    pad_x = int(np.ceil((x1 - x0) * padding))
    pad_y = int(np.ceil((y1 - y0) * padding))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    height, width = image_shape
    target_width = min(width, max(x1 - x0, int(np.ceil((x1 - x0) * scale))))
    target_height = min(height, max(y1 - y0, int(np.ceil((y1 - y0) * scale))))
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    new_x0 = int(np.floor(center_x - target_width * 0.5))
    new_y0 = int(np.floor(center_y - target_height * 0.5))
    new_x0 = min(max(new_x0, 0), width - target_width)
    new_y0 = min(max(new_y0, 0), height - target_height)
    return new_x0, new_y0, new_x0 + target_width, new_y0 + target_height


def _binary_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return float(horizontal + vertical)


def _validate_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {value.shape}")
    return value


def _validate_mask(mask: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match image shape {shape}")
    if not np.any(value):
        raise ValueError("mask must contain at least one pixel")
    return value


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Numeric appearance features require opencv-python") from exc
    return cv2
