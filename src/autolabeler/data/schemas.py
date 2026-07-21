from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Any

Provenance = Literal[
    "manual", "camera_teacher", "lidar_teacher", "temporal_propagated", "rule_labeled",
    "student_predicted", "ensemble_agreed", "human_corrected"
]


@dataclass
class OrganizedLiDARFrame:
    frame_id: str
    points_range: Any
    points_flat: Any
    timestamp_s: Optional[int] = None
    timestamp_u: Optional[int] = None
    timestamp_us: Optional[int] = None
    background_light_intensity: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticSegmentationResult:
    frame_id: str
    semantic_mask: Any
    confidence_mask: Optional[Any] = None
    pseudo_label_version: str = "semantic_v0"
    provenance: Provenance = "student_predicted"
    metadata: dict[str, Any] = field(default_factory=dict)
