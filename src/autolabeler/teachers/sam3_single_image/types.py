"""Shared data structures and constants for single-image SAM3 inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_KEYS = (
    "masks",
    "pred_masks",
    "out_binary_masks",
    "video_res_masks",
    "mask_logits",
    "out_mask_logits",
)
SCORE_KEYS = (
    "scores",
    "pred_scores",
    "object_scores",
    "ious",
    "obj_scores",
    "out_probs",
)
BOX_KEYS = ("boxes_xyxy", "boxes", "pred_boxes", "out_boxes_xywh")
SIGN_TYPE_LABELS = (
    "height_restriction_sign_at_underground",
    "height_restriction_barrel",
    "underground_parking_sign",
    "overhead_traffic_sign",
    "side_plate",
    "induction_sign",
)
SIGN_TYPE_REJECT_LABEL = "not_a_sign"


@dataclass(frozen=True)
class Sam3Instance:
    label: str
    class_id: int
    prompt: str
    score: float
    mask: np.ndarray
    # [x_min, y_min, width, height] in 0-1.0 format
    box: np.ndarray | None = None
    source_label: str | None = None
    orientation_label: str | None = None
    orientation_score: float | None = None
    orientation_margin: float | None = None
    orientation_probabilities: tuple[float, float, float] | None = None
    orientation_fallback: bool = False
    orientation_fallback_reason: str | None = None
    sign_classifier_label: str | None = None
    sign_classifier_assigned_label: str | None = None
    sign_classifier_confidence: float | None = None
    sign_classifier_margin: float | None = None
    sign_classifier_probabilities: tuple[tuple[str, float], ...] | None = None
    sign_classifier_image_probabilities: tuple[tuple[str, float], ...] | None = None
    sign_classifier_numeric_probabilities: tuple[tuple[str, float], ...] | None = None
    sign_sam3_class_scores: tuple[tuple[str, float], ...] | None = None
    sign_classifier_accepted: bool = False
    sign_classifier_fallback_reason: str | None = None


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    output_dir: Path
    image_size: tuple[int, int]
    instances: list[Sam3Instance]
    class_pixel_counts: dict[str, int]
    overlay_instances: tuple[Sam3Instance, ...] = ()
    orientation_predictions: tuple[dict[str, Any], ...] = ()
    sign_predictions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FolderDetectionSummary:
    detected_classes: frozenset[str]
    frame_ids: tuple[str, ...]
    frame_class_presence: tuple[frozenset[str], ...]
