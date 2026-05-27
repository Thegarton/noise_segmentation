from __future__ import annotations

import numpy as np

from autolabeler.data.bin_loader import H, W


def build_noise_review_candidates(
    *,
    points_range: np.ndarray,
    semantic_mask: np.ndarray,
    confidence_mask: np.ndarray | None = None,
    ignore_id: int = 255,
    low_confidence_threshold: float = 0.25,
    streak_min_valid: int = 24,
) -> dict[str, np.ndarray]:
    points = np.asarray(points_range, dtype=np.float32)
    labels = np.asarray(semantic_mask)
    if points.shape != (H, W, 4):
        raise ValueError(f"points_range must have shape {(H, W, 4)}, got {points.shape}")
    if labels.shape != (H, W):
        raise ValueError(f"semantic_mask must have shape {(H, W)}, got {labels.shape}")

    valid_points = np.linalg.norm(points[:, :, :3], axis=2) > 0.0
    residual = valid_points & ((labels == 0) | (labels == ignore_id))
    out = {
        "residual_candidate": residual,
        "range_streak_candidate": _range_streak_candidate(points[:, :, :3], residual, min_valid=streak_min_valid),
    }
    if confidence_mask is not None:
        conf = np.asarray(confidence_mask, dtype=np.float32)
        if conf.shape != (H, W):
            raise ValueError(f"confidence_mask must have shape {(H, W)}, got {conf.shape}")
        out["low_confidence_residual_candidate"] = residual & (conf <= low_confidence_threshold)
    return out


def _range_streak_candidate(xyz: np.ndarray, residual: np.ndarray, *, min_valid: int) -> np.ndarray:
    ranges = np.linalg.norm(xyz, axis=2)
    candidate = np.zeros((H, W), dtype=bool)
    for row in range(H):
        valid_cols = np.where(residual[row])[0]
        if valid_cols.size < min_valid:
            continue
        row_ranges = ranges[row, valid_cols]
        median_range = np.median(row_ranges)
        close_to_layer = np.abs(row_ranges - median_range) < max(0.25, median_range * 0.01)
        if np.count_nonzero(close_to_layer) >= min_valid:
            candidate[row, valid_cols[close_to_layer]] = True
    return candidate

