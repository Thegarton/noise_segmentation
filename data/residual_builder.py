"""Residual cloud builder for KITTI Raw XML annotations only.

Scope intentionally narrowed to one data contract:
- Point cloud directory with frame files named: `{i:06d}.bin`.
- XML annotation file containing 3D boxes for all frames in the snippet.
- Sequence is processed as 10 frames by default (configurable).

Point cloud format:
- KITTI velodyne `.bin` with float32 tuples `(x, y, z, intensity)`.

XML expected fields per box:
- frame index
- center/location x, y, z
- size/dimensions h, w, l (converted to dx=l, dy=w, dz=h)
- yaw/rotation_y (radians)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class ResidualBuildResult:
    residual_points: np.ndarray
    residual_mask: np.ndarray
    explained_mask: np.ndarray
    stats: dict


def read_kitti_bin(bin_path: str | Path) -> np.ndarray:
    """Read KITTI velodyne .bin into [N,4] float32 array: x,y,z,intensity."""
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid KITTI .bin file (size not divisible by 4): {bin_path}")
    return arr.reshape(-1, 4)


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


def _find_first_float(elem: ET.Element, names: Iterable[str]) -> float | None:
    for name in names:
        child = elem.find(name)
        if child is not None and child.text is not None:
            return float(child.text)
    return None


def parse_kitti_raw_xml_boxes(xml_path: str | Path) -> dict[int, list[dict]]:
    """Parse KITTI Raw XML annotations into frame-indexed 3D boxes.

    The parser is permissive for tag naming variants and expects each object element
    to expose frame id, center/location xyz, dimensions (h,w,l), and yaw/rotation_y.
    """
    root = ET.parse(xml_path).getroot()
    out: dict[int, list[dict]] = {}

    object_nodes = root.findall('.//object') + root.findall('.//tracklet') + root.findall('.//item')
    for node in object_nodes:
        frame = _find_first_float(node, ["frame", "frame_id", "frameIndex", "first_frame"])
        x = _find_first_float(node, ["x", "tx", "pos_x", "location_x", "center_x"])
        y = _find_first_float(node, ["y", "ty", "pos_y", "location_y", "center_y"])
        z = _find_first_float(node, ["z", "tz", "pos_z", "location_z", "center_z"])
        h = _find_first_float(node, ["h", "height"])
        w = _find_first_float(node, ["w", "width"])
        l = _find_first_float(node, ["l", "length"])
        yaw = _find_first_float(node, ["rotation_y", "yaw", "rz", "rotation_z"])

        if None in (frame, x, y, z, h, w, l):
            continue
        if yaw is None:
            yaw = 0.0

        box = {
            "location": [float(x), float(y), float(z)],
            "dimensions": [float(h), float(w), float(l)],
            "rotation_y": float(yaw),
        }
        frame_idx = int(frame)
        out.setdefault(frame_idx, []).append(box)

    return out


def build_residual_cloud(points: np.ndarray, kitti_boxes: Iterable[dict]) -> ResidualBuildResult:
    """Build residual cloud by removing points inside KITTI boxes for one frame."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, C] with C >= 3")

    xyz = points[:, :3]
    explained = np.zeros(points.shape[0], dtype=bool)

    for box in kitti_boxes:
        center = np.asarray(box["location"], dtype=np.float32)
        h, w, l = np.asarray(box["dimensions"], dtype=np.float32)
        size = np.array([l, w, h], dtype=np.float32)
        yaw = float(box.get("rotation_y", 0.0))
        explained |= _points_in_oriented_box(xyz, center, size, yaw)

    residual_mask = ~explained
    residual_points = points[residual_mask]
    stats = {
        "num_input_points": int(points.shape[0]),
        "num_explained_points": int(explained.sum()),
        "num_residual_points": int(residual_mask.sum()),
        "residual_ratio": float(residual_mask.mean()) if points.shape[0] > 0 else 0.0,
    }
    return ResidualBuildResult(residual_points, residual_mask, explained, stats)


def build_kitti_raw_xml_residual_sequence(
    point_cloud_dir: str | Path,
    xml_mask_path: str | Path,
    num_frames: int = 10,
    start_index: int = 0,
) -> list[ResidualBuildResult]:
    """Build residuals for KITTI Raw XML annotations across sequential .bin frames.

    Frame file names must follow `{i:06d}.bin`.
    """
    point_cloud_dir = Path(point_cloud_dir)
    labels_by_frame = parse_kitti_raw_xml_boxes(xml_mask_path)

    results: list[ResidualBuildResult] = []
    for frame_idx in range(start_index, start_index + num_frames):
        bin_path = point_cloud_dir / f"{frame_idx:06d}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing frame file: {bin_path}")
        points = read_kitti_bin(bin_path)
        boxes = labels_by_frame.get(frame_idx, [])
        results.append(build_residual_cloud(points, boxes))
    return results
