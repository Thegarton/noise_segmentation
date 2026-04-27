"""Build residual LiDAR clouds by removing explained points.

This module removes points explained by:
1) classical object bounding boxes (axis-aligned or yaw-rotated),
2) optional irregular object masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class ResidualBuildResult:
    """Output of residual cloud construction."""

    residual_points: np.ndarray
    residual_mask: np.ndarray
    explained_mask: np.ndarray
    stats: dict


def _points_in_oriented_box(
    xyz: np.ndarray,
    center: np.ndarray,
    size_xyz: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    """Return mask for points inside a yaw-rotated 3D box.

    Box convention: size_xyz = [dx, dy, dz], yaw around +Z axis.
    """
    rel = xyz - center[None, :]
    c, s = np.cos(-yaw_rad), np.sin(-yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=xyz.dtype)
    rel_xy = rel[:, :2] @ rot.T
    half = size_xyz / 2.0
    inside_xy = (np.abs(rel_xy[:, 0]) <= half[0]) & (np.abs(rel_xy[:, 1]) <= half[1])
    inside_z = np.abs(rel[:, 2]) <= half[2]
    return inside_xy & inside_z


def _normalize_box(box: Mapping) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(box["center"], dtype=np.float32)
    size = np.asarray(box["size"], dtype=np.float32)
    yaw = float(box.get("yaw", 0.0))
    return center, size, yaw


def build_residual_cloud(
    points: np.ndarray,
    object_boxes: Iterable[Mapping],
    irregular_explained_mask: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
) -> ResidualBuildResult:
    """Build residual cloud from points and known object annotations.

    Args:
        points: [N, C] array, first 3 channels are x/y/z.
        object_boxes: iterable of dicts with keys:
            - center: [x, y, z]
            - size: [dx, dy, dz]
            - yaw: optional radians
        irregular_explained_mask: optional bool [N] mask for non-box explained points.
        keep_mask: optional bool [N] mask to force-keep points (applied at the end).
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, C] with C >= 3")

    xyz = points[:, :3]
    n = points.shape[0]
    explained = np.zeros(n, dtype=bool)

    for box in object_boxes:
        center, size, yaw = _normalize_box(box)
        explained |= _points_in_oriented_box(xyz, center, size, yaw)

    if irregular_explained_mask is not None:
        if irregular_explained_mask.shape[0] != n:
            raise ValueError("irregular_explained_mask must have length N")
        explained |= irregular_explained_mask.astype(bool)

    residual_mask = ~explained

    if keep_mask is not None:
        if keep_mask.shape[0] != n:
            raise ValueError("keep_mask must have length N")
        residual_mask |= keep_mask.astype(bool)
        explained = ~residual_mask

    residual_points = points[residual_mask]
    stats = {
        "num_input_points": int(n),
        "num_explained_points": int(explained.sum()),
        "num_residual_points": int(residual_mask.sum()),
        "residual_ratio": float(residual_mask.mean()) if n > 0 else 0.0,
    }

    return ResidualBuildResult(
        residual_points=residual_points,
        residual_mask=residual_mask,
        explained_mask=explained,
        stats=stats,
    )


def residual_from_frames(
    frame_points: Sequence[np.ndarray],
    frame_boxes: Sequence[Iterable[Mapping]],
    frame_irregular_masks: Optional[Sequence[np.ndarray]] = None,
) -> list[ResidualBuildResult]:
    """Batch helper for multiple frames."""
    if len(frame_points) != len(frame_boxes):
        raise ValueError("frame_points and frame_boxes must have same length")

    out: list[ResidualBuildResult] = []
    for i, (pts, boxes) in enumerate(zip(frame_points, frame_boxes)):
        irr = None if frame_irregular_masks is None else frame_irregular_masks[i]
        out.append(build_residual_cloud(pts, boxes, irregular_explained_mask=irr))
    return out
