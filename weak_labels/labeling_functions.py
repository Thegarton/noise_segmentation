"""Weak labeling functions for residual noise family bootstrapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FAMILIES = {
    "crosstalk_family",
    "reflection_family",
    "particulate_family",
    "distance_family",
    "adhesive_family",
    "unknown_artifact",
    "background",
}


@dataclass
class WeakLabel:
    label: str
    confidence: float
    rule: str


def _safe_get(desc: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = desc.get(key, default)
    try:
        return float(val)
    except Exception:
        return default


def lf_crosstalk(desc: dict[str, Any]) -> WeakLabel | None:
    linearity = _safe_get(desc, "linearity")
    scattering = _safe_get(desc, "scattering")
    bbox = desc.get("bbox_size", [0.0, 0.0, 0.0])
    span_xy = max(float(bbox[0]), float(bbox[1]))
    if linearity > 0.85 and scattering < 0.08 and span_xy > 2.0:
        return WeakLabel("crosstalk_family", 0.82, "lf_crosstalk")
    return None


def lf_reflection(desc: dict[str, Any]) -> WeakLabel | None:
    mean_z = _safe_get(desc, "mean_z", 0.0)
    std_range = _safe_get(desc, "std_range")
    if mean_z < -0.2 and std_range > 2.0:
        return WeakLabel("reflection_family", 0.78, "lf_reflection")
    return None


def lf_particulate(desc: dict[str, Any]) -> WeakLabel | None:
    n = _safe_get(desc, "num_points")
    scattering = _safe_get(desc, "scattering")
    bbox = desc.get("bbox_size", [0.0, 0.0, 0.0])
    volume = float(bbox[0]) * float(bbox[1]) * float(bbox[2])
    density = n / max(volume, 1e-3)
    if n < 80 and scattering > 0.25 and density < 50:
        return WeakLabel("particulate_family", 0.75, "lf_particulate")
    return None


def lf_distance(desc: dict[str, Any]) -> WeakLabel | None:
    mean_range = _safe_get(desc, "mean_range")
    n = _safe_get(desc, "num_points")
    if mean_range > 50.0 and n < 30:
        return WeakLabel("distance_family", 0.72, "lf_distance")
    return None


def lf_adhesive(desc: dict[str, Any]) -> WeakLabel | None:
    dist_to_obj = _safe_get(desc, "distance_to_nearest_object", 999.0)
    planarity = _safe_get(desc, "planarity")
    bbox = desc.get("bbox_size", [0.0, 0.0, 0.0])
    if dist_to_obj < 0.25 and planarity > 0.3 and max(bbox) < 1.2:
        return WeakLabel("adhesive_family", 0.80, "lf_adhesive")
    return None


def assign_weak_family_label(descriptor: dict[str, Any]) -> WeakLabel:
    """Apply heuristic rules and return strongest weak label."""
    voters = [
        lf_crosstalk(descriptor),
        lf_reflection(descriptor),
        lf_particulate(descriptor),
        lf_distance(descriptor),
        lf_adhesive(descriptor),
    ]
    votes = [v for v in voters if v is not None]
    if not votes:
        return WeakLabel("unknown_artifact", 0.35, "fallback_unknown")
    return max(votes, key=lambda x: x.confidence)
