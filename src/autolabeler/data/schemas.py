from __future__ import annotations

from dataclasses import dataclass, field, asdict
<<<<<<< codex/build-offline-auto-labeling-system-wl2yjl
from typing import Optional, Literal, Any
=======
from typing import Optional, Literal
>>>>>>> main

Provenance = Literal[
    "manual", "camera_teacher", "lidar_teacher", "temporal_propagated", "rule_labeled",
    "student_predicted", "ensemble_agreed", "human_corrected"
]
ReviewStatus = Literal["auto_accepted", "needs_review", "reviewed_accepted", "reviewed_rejected"]


@dataclass
class OrganizedLiDARFrame:
    frame_id: str
<<<<<<< codex/build-offline-auto-labeling-system-wl2yjl
    points_range: Any
    points_flat: Any
    timestamp_s: Optional[int] = None
    timestamp_u: Optional[int] = None
    timestamp_us: Optional[int] = None
    background_light_intensity: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)
=======
    # pcd in range image format, shape (H, W, C) where C >= 3 (x, y, z, [intensity, ...])
    points_range: "object"
    # pcd in flat format, shape (N, C) where C >= 3 (x, y, z, [intensity, ...])
    points_flat: "object"
>>>>>>> main


@dataclass
class SequenceSample:
    current: OrganizedLiDARFrame
    past: list[OrganizedLiDARFrame] = field(default_factory=list)
    future: list[OrganizedLiDARFrame] = field(default_factory=list)


@dataclass
class Box3D:
<<<<<<< codex/build-offline-auto-labeling-system-wl2yjl
    center: list[float]
    size: list[float]
=======
    # tx, ty, tz in LiDAR coordinate
    center: list[float]
    # h, w, l in LiDAR coordinate
    size: list[float]
    # rotation around z-axis in radians, in LiDAR coordinate
>>>>>>> main
    yaw: float
    box_type: str = "adaptive_obb"


@dataclass
class LabelInstance:
    frame_id: str
    semantic_class: str
    instance_id: int
    track_id: Optional[int]
    point_indices: list[int]
    range_image_indices: list[list[int]]
    box_3d: Optional[Box3D]
    mask_confidence: float
    class_confidence: float
    box_confidence: float
    final_confidence: float
    provenance: Provenance
    branch_name: str
    teacher_sources: list[str]
    review_status: ReviewStatus
    pseudo_label_version: str


@dataclass
class AutoLabelingResult:
    frame_id: str
    labels: list[LabelInstance]

    def to_jsonable(self) -> dict:
        d = {"frame_id": self.frame_id, "labels": []}
        for x in self.labels:
            y = asdict(x)
            if x.box_3d is None:
                y["box_3d"] = None
            d["labels"].append(y)
        return d
