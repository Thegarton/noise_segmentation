"""Build residual LiDAR clouds by removing explained points.

Supported bbox input formats (per box mapping):
1) Generic absolute center/size:
   {"center": [x,y,z], "size": [dx,dy,dz], "yaw": rad}
2) centroid_abs + dims:
   {"centroid_abs": [x,y,z], "dims": [dx,dy,dz], "yaw"|"heading": rad}
3) KITTI-style 3D label:
   {"location": [x,y,z], "dimensions": [h,w,l], "rotation_y": rad}
   Internally converted to size=[l,w,h].
4) Axis-aligned bounds:
   {"min": [xmin,ymin,zmin], "max": [xmax,ymax,zmax]}
5) centroid_rel + dims:
   {"centroid_rel": [x,y,z], "dims": [dx,dy,dz], ...}
   Requires frame_meta with "ego_translation_abs" and optional "ego_yaw_abs".

Additional utilities:
- Read KITTI .bin point clouds (float32 x,y,z,intensity).
- Parse KITTI Raw/Tracking label text and group boxes by frame.
- Build residuals for 10-frame snippets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class ResidualBuildResult:
    """Output of residual cloud construction."""

    residual_points: np.ndarray
    residual_mask: np.ndarray
    explained_mask: np.ndarray
    stats: dict


def _as_vec3(value: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be length-3, got shape {arr.shape}")
    return arr


def _points_in_oriented_box(
    xyz: np.ndarray,
    center: np.ndarray,
    size_xyz: np.ndarray,
    yaw_rad: float,
) -> np.ndarray:
    rel = xyz - center[None, :]
    c, s = np.cos(-yaw_rad), np.sin(-yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=xyz.dtype)
    rel_xy = rel[:, :2] @ rot.T
    half = size_xyz / 2.0
    inside_xy = (np.abs(rel_xy[:, 0]) <= half[0]) & (np.abs(rel_xy[:, 1]) <= half[1])
    inside_z = np.abs(rel[:, 2]) <= half[2]
    return inside_xy & inside_z


def _rel_to_abs(rel_xyz: np.ndarray, frame_meta: Mapping) -> np.ndarray:
    t = _as_vec3(frame_meta["ego_translation_abs"], "ego_translation_abs")
    yaw = float(frame_meta.get("ego_yaw_abs", 0.0))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    abs_xy = rel_xyz[:2] @ rot.T + t[:2]
    abs_z = rel_xyz[2] + t[2]
    return np.array([abs_xy[0], abs_xy[1], abs_z], dtype=np.float32)


def _normalize_box(box: Mapping, frame_meta: Optional[Mapping] = None) -> tuple[np.ndarray, np.ndarray, float]:
    if "center" in box and "size" in box:
        center = _as_vec3(box["center"], "center")
        size = _as_vec3(box["size"], "size")
        yaw = float(box.get("yaw", box.get("heading", 0.0)))
        return center, size, yaw

    if "centroid_abs" in box and ("dims" in box or "size" in box):
        center = _as_vec3(box["centroid_abs"], "centroid_abs")
        size = _as_vec3(box.get("dims", box.get("size")), "dims/size")
        yaw = float(box.get("yaw", box.get("heading", 0.0)))
        return center, size, yaw

    if "location" in box and "dimensions" in box:
        center = _as_vec3(box["location"], "location")
        h, w, l = np.asarray(box["dimensions"], dtype=np.float32).tolist()
        size = np.array([l, w, h], dtype=np.float32)
        yaw = float(box.get("rotation_y", box.get("yaw", 0.0)))
        return center, size, yaw

    if "min" in box and "max" in box:
        mn = _as_vec3(box["min"], "min")
        mx = _as_vec3(box["max"], "max")
        center = (mn + mx) / 2.0
        size = np.maximum(mx - mn, 1e-4)
        return center, size, 0.0

    if "centroid_rel" in box and ("dims" in box or "size" in box):
        if frame_meta is None:
            raise ValueError("centroid_rel provided but frame_meta is missing")
        rel = _as_vec3(box["centroid_rel"], "centroid_rel")
        center = _rel_to_abs(rel, frame_meta)
        size = _as_vec3(box.get("dims", box.get("size")), "dims/size")
        yaw = float(box.get("yaw", box.get("heading", 0.0))) + float(frame_meta.get("ego_yaw_abs", 0.0))
        return center, size, yaw

    raise ValueError(
        "Unsupported bbox format. Supported keys include: "
        "(center,size), (centroid_abs,dims), (location,dimensions), (min,max), (centroid_rel,dims)."
    )


def read_kitti_bin(bin_path: str | Path) -> np.ndarray:
    """Read KITTI velodyne .bin into [N,4] float32 array: x,y,z,intensity."""
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid KITTI .bin file (size not divisible by 4): {bin_path}")
    return arr.reshape(-1, 4)


def parse_kitti_raw_labels(label_path: str | Path) -> dict[int, list[dict]]:
    """Parse KITTI Raw/Tracking style labels into frame->bbox mappings.

    Expected per line (tracking/raw style):
    frame track_id type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
    h w l x y z rotation_y [...optional extras]
    """
    by_frame: dict[int, list[dict]] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 17:
                raise ValueError(f"Line {line_num} in {label_path} has {len(parts)} fields; expected >=17")
            frame_id = int(parts[0])
            obj_type = parts[2]
            h, w, l = map(float, parts[10:13])
            x, y, z = map(float, parts[13:16])
            ry = float(parts[16])
            box = {
                "type": obj_type,
                "location": [x, y, z],
                "dimensions": [h, w, l],
                "rotation_y": ry,
            }
            by_frame.setdefault(frame_id, []).append(box)
    return by_frame


def build_residual_cloud(
    points: np.ndarray,
    object_boxes: Iterable[Mapping],
    irregular_explained_mask: Optional[np.ndarray] = None,
    keep_mask: Optional[np.ndarray] = None,
    frame_meta: Optional[Mapping] = None,
) -> ResidualBuildResult:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, C] with C >= 3")

    xyz = points[:, :3]
    n = points.shape[0]
    explained = np.zeros(n, dtype=bool)

    for box in object_boxes:
        center, size, yaw = _normalize_box(box, frame_meta=frame_meta)
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
    frame_meta: Optional[Sequence[Optional[Mapping]]] = None,
) -> list[ResidualBuildResult]:
    if len(frame_points) != len(frame_boxes):
        raise ValueError("frame_points and frame_boxes must have same length")

    out: list[ResidualBuildResult] = []
    for i, (pts, boxes) in enumerate(zip(frame_points, frame_boxes)):
        irr = None if frame_irregular_masks is None else frame_irregular_masks[i]
        meta = None if frame_meta is None else frame_meta[i]
        out.append(build_residual_cloud(pts, boxes, irregular_explained_mask=irr, frame_meta=meta))
    return out


def build_kitti_residual_sequence(
    bin_paths: Sequence[str | Path],
    labels_by_frame: Mapping[int, Sequence[Mapping]],
    num_frames: int = 10,
) -> list[ResidualBuildResult]:
    """Build residuals for a KITTI sequence snippet (default 10 frames).

    Args:
        bin_paths: ordered list of .bin frame paths.
        labels_by_frame: mapping frame_idx -> iterable of KITTI-like bbox dicts.
        num_frames: number of first frames to process from bin_paths.
    """
    if len(bin_paths) < num_frames:
        raise ValueError(f"Expected at least {num_frames} .bin frames, got {len(bin_paths)}")

    results: list[ResidualBuildResult] = []
    for frame_idx, bin_path in enumerate(bin_paths[:num_frames]):
        points = read_kitti_bin(bin_path)
        boxes = labels_by_frame.get(frame_idx, [])
        results.append(build_residual_cloud(points, boxes))
    return results
