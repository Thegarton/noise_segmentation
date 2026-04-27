"""Range-image projection utilities for LiDAR residual processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RangeImageResult:
    range_image: np.ndarray
    intensity_image: np.ndarray
    height_image: np.ndarray
    residual_mask_image: np.ndarray
    point_count_image: np.ndarray
    index_image: np.ndarray


def spherical_features(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute range, azimuth, elevation for each point."""
    r = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    el = np.arctan2(xyz[:, 2], np.maximum(np.linalg.norm(xyz[:, :2], axis=1), 1e-6))
    return r, az, el


def project_to_range_image(
    points: np.ndarray,
    residual_mask: Optional[np.ndarray] = None,
    num_azimuth_bins: int = 2048,
    num_elevation_bins: int = 64,
    elevation_min_rad: float = np.deg2rad(-25.0),
    elevation_max_rad: float = np.deg2rad(15.0),
) -> RangeImageResult:
    """Project points to range image keeping nearest point per pixel."""
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("points must be [N, C] where C >= 4 (x,y,z,intensity)")

    xyz = points[:, :3]
    intensity = points[:, 3]
    n = points.shape[0]
    if residual_mask is None:
        residual_mask = np.ones(n, dtype=bool)

    r, az, el = spherical_features(xyz)
    az_norm = (az + np.pi) / (2.0 * np.pi)
    el_norm = (el - elevation_min_rad) / max(elevation_max_rad - elevation_min_rad, 1e-6)

    col = np.clip((az_norm * num_azimuth_bins).astype(int), 0, num_azimuth_bins - 1)
    row = np.clip((el_norm * num_elevation_bins).astype(int), 0, num_elevation_bins - 1)

    shape = (num_elevation_bins, num_azimuth_bins)
    range_img = np.full(shape, np.inf, dtype=np.float32)
    intensity_img = np.zeros(shape, dtype=np.float32)
    height_img = np.zeros(shape, dtype=np.float32)
    residual_img = np.zeros(shape, dtype=np.uint8)
    count_img = np.zeros(shape, dtype=np.int32)
    index_img = np.full(shape, -1, dtype=np.int32)

    for i in range(n):
        rr, cc = row[i], col[i]
        count_img[rr, cc] += 1
        if r[i] < range_img[rr, cc]:
            range_img[rr, cc] = r[i]
            intensity_img[rr, cc] = intensity[i]
            height_img[rr, cc] = xyz[i, 2]
            residual_img[rr, cc] = int(bool(residual_mask[i]))
            index_img[rr, cc] = i

    range_img[~np.isfinite(range_img)] = 0.0
    return RangeImageResult(
        range_image=range_img,
        intensity_image=intensity_img,
        height_image=height_img,
        residual_mask_image=residual_img,
        point_count_image=count_img,
        index_image=index_img,
    )


def local_density_from_range_mask(mask: np.ndarray, window: int = 5) -> np.ndarray:
    """Count valid neighbors for each pixel in square window."""
    if mask.ndim != 2:
        raise ValueError("mask must be [H, W]")
    if window % 2 == 0:
        raise ValueError("window must be odd")

    pad = window // 2
    m = (mask > 0).astype(np.int32)
    padded = np.pad(m, ((pad, pad), (pad, pad)), mode="constant")
    out = np.zeros_like(m, dtype=np.int32)

    # Integral image for O(1) region sums.
    ii = padded.cumsum(0).cumsum(1)
    h, w = m.shape
    for r in range(h):
        r0, r1 = r, r + window - 1
        for c in range(w):
            c0, c1 = c, c + window - 1
            s = ii[r1, c1]
            if r0 > 0:
                s -= ii[r0 - 1, c1]
            if c0 > 0:
                s -= ii[r1, c0 - 1]
            if r0 > 0 and c0 > 0:
                s += ii[r0 - 1, c0 - 1]
            out[r, c] = s
    return out
