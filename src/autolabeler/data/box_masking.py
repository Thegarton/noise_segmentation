from __future__ import annotations

import math

import numpy as np

from .bin_loader import W


def points_inside_box(
    points_flat: np.ndarray,
    center: list[float],
    size: list[float],
    yaw: float,
) -> tuple[list[int], list[list[int]], float]:
    width, height, length = [float(x) for x in size]
    pts = np.asarray(points_flat, dtype=np.float32)
    xyz = pts[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    nonzero = ~((np.abs(xyz[:, 0]) < 1e-4) & (np.abs(xyz[:, 1]) < 1e-4) & (np.abs(xyz[:, 2]) < 1e-4))
    valid = finite & nonzero

    c = np.asarray(center, dtype=np.float32)
    shifted = xyz - c[None, :]
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)
    local_x = shifted[:, 0] * cos_y - shifted[:, 1] * sin_y
    local_y = shifted[:, 0] * sin_y + shifted[:, 1] * cos_y
    half_l = length * 0.5
    half_w = width * 0.5
    half_h = height * 0.5

    inside = (
        valid
        & (np.abs(local_x) <= half_l)
        & (np.abs(local_y) <= half_w)
        & (np.abs(shifted[:, 2]) <= half_h)
    )
    indices = np.flatnonzero(inside).astype(int).tolist()
    range_indices = [[int(i // W), int(i % W)] for i in indices]

    expected_points = max(12.0, (width * height * length) * 2.5)
    mask_conf = min(1.0, len(indices) / expected_points)
    return indices, range_indices, float(mask_conf)
