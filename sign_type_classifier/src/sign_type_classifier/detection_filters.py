"""Geometry and normalized box-size filters for SAM3 detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .clustering import SignDetection


GEOMETRY_FILTER_RULES: dict[str, dict[str, float]] = {
    "arrestor": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 2.0,
    },
    "wheel_chock": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.5,
    },
    "underground_parking_sign": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 1.0,
    },
    "ground_markings": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 0.0,
    },
    "induction_sign": {
        "min_y_bottom": 0.50,
        "min_aspect_ratio": 1.0,
        "max_aspect_ratio": 4.0,
    },
    "parking_barrier_lock": {
        "min_y_bottom": 0.5,
        "min_aspect_ratio": 0.0,
    },
    "pillar_corner_guard": {
        "min_y_bottom": 0.2,
        "min_aspect_ratio": 0.9,
    },
    "height_restriction_barrel": {
        "min_y_bottom": 0.4,
        "min_aspect_ratio": 0.0,
    },
}


SIZE_FILTER_RULES: dict[str, dict[str, float]] = {
    "arrestor": {
        "max_box_width": 0.4,
        "max_box_height": 0.15,
        "max_box_area": 0.07,
    },
    "wheel_chock": {
        "max_box_width": 0.4,
        "max_box_height": 0.15,
        "max_box_area": 0.07,
    },
}


@dataclass(frozen=True)
class DetectionRejection:
    detection: SignDetection
    filter_name: str
    reason: str
    box_metrics: dict[str, float] | None


def reject_instance_by_geometry(
    item: SignDetection,
    *,
    rules: Mapping[str, Mapping[str, float]] = GEOMETRY_FILTER_RULES,
) -> bool:
    """Return whether a normalized XYWH box violates its label geometry rule."""
    rule = rules.get(item.label)
    if rule is None:
        return False

    min_y_bottom = _rule_threshold(rule, "min_y_bottom", label=item.label)
    min_aspect_ratio = _rule_threshold(rule, "min_aspect_ratio", label=item.label)
    max_aspect_ratio = (
        _rule_threshold(rule, "max_aspect_ratio", label=item.label)
        if "max_aspect_ratio" in rule
        else None
    )
    if not 0.0 <= min_y_bottom <= 1.0 or min_aspect_ratio < 0.0:
        raise ValueError(
            f"Invalid geometry filter thresholds for label {item.label!r}: {rule!r}"
        )
    if max_aspect_ratio is not None and max_aspect_ratio < min_aspect_ratio:
        raise ValueError(
            f"Invalid geometry filter aspect-ratio range for label {item.label!r}: {rule!r}"
        )

    metrics = normalized_xywh_metrics(item.box)
    if metrics is None:
        return True
    y_bottom = metrics["y_bottom"]
    aspect_ratio = metrics["aspect_ratio"]

    if item.label == "underground_parking_sign":
        return y_bottom > min_y_bottom or aspect_ratio < min_aspect_ratio
    if item.label == "induction_sign":
        assert max_aspect_ratio is not None
        return (
            y_bottom > min_y_bottom
            or aspect_ratio < min_aspect_ratio
            or aspect_ratio > max_aspect_ratio
        )
    if item.label in {"ground_markings", "parking_barrier_lock"}:
        return y_bottom < min_y_bottom
    if item.label == "pillar_corner_guard":
        return y_bottom < min_y_bottom or aspect_ratio > min_aspect_ratio
    return y_bottom < min_y_bottom or aspect_ratio < min_aspect_ratio


def reject_instance_by_size(
    item: SignDetection,
    *,
    rules: Mapping[str, Mapping[str, float]] = SIZE_FILTER_RULES,
) -> bool:
    """Return whether a normalized XYWH box exceeds its label size limits."""
    rule = rules.get(item.label)
    if rule is None:
        return False

    max_width = _rule_threshold(rule, "max_box_width", label=item.label)
    max_height = _rule_threshold(rule, "max_box_height", label=item.label)
    max_area = _rule_threshold(rule, "max_box_area", label=item.label)
    if max_width < 0.0 or max_height < 0.0 or max_area < 0.0:
        raise ValueError(f"Invalid size filter thresholds for label {item.label!r}: {rule!r}")

    metrics = normalized_xywh_metrics(item.box)
    if metrics is None:
        return True
    return (
        metrics["width"] > max_width
        or metrics["height"] > max_height
        or metrics["area"] > max_area
    )


def filter_detections(
    detections: Sequence[SignDetection],
    *,
    geometry_rules: Mapping[str, Mapping[str, float]] = GEOMETRY_FILTER_RULES,
    size_rules: Mapping[str, Mapping[str, float]] = SIZE_FILTER_RULES,
) -> tuple[list[SignDetection], list[DetectionRejection]]:
    """Apply geometry first and size second while preserving detection order."""
    accepted: list[SignDetection] = []
    rejected: list[DetectionRejection] = []
    for item in detections:
        if reject_instance_by_geometry(item, rules=geometry_rules):
            rejected.append(
                DetectionRejection(
                    detection=item,
                    filter_name="geometry",
                    reason="box violates label geometry rule",
                    box_metrics=normalized_xywh_metrics(item.box),
                )
            )
            continue
        if reject_instance_by_size(item, rules=size_rules):
            rejected.append(
                DetectionRejection(
                    detection=item,
                    filter_name="size",
                    reason="box exceeds label size rule",
                    box_metrics=normalized_xywh_metrics(item.box),
                )
            )
            continue
        accepted.append(item)
    return accepted, rejected


def normalized_xywh_metrics(box: Any) -> dict[str, float] | None:
    if box is None:
        return None
    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x_min, y_min, width, height = (float(value) for value in values)
    if x_min < 0.0 or y_min < 0.0 or width < 0.0 or height <= 1e-6:
        return None
    return {
        "x_min": x_min,
        "y_min": y_min,
        "width": width,
        "height": height,
        "area": width * height,
        "y_bottom": y_min + height,
        "aspect_ratio": width / height,
    }


def rejection_to_json(item: DetectionRejection) -> dict[str, Any]:
    detection = item.detection
    return {
        "label": str(detection.label),
        "prompt": str(detection.prompt),
        "score": float(detection.score),
        "box": (
            None
            if detection.box is None
            else np.asarray(detection.box, dtype=np.float32).reshape(-1).tolist()
        ),
        "filter": item.filter_name,
        "reason": item.reason,
        "box_metrics": item.box_metrics,
    }


def _rule_threshold(rule: Mapping[str, float], name: str, *, label: str) -> float:
    try:
        value = float(rule[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid filter rule for label {label!r}: {rule!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {name} in filter rule for label {label!r}: {rule!r}")
    return value
