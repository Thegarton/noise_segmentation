"""Detect horizontal and vertical streak-like artifacts in range view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class StreakCandidate:
    kind: Literal["horizontal", "vertical"]
    bbox: tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    num_pixels: int
    aspect_ratio: float
    score: float


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if not binary[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels = []
            while stack:
                rr, cc = stack.pop()
                pixels.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pixels, dtype=np.int32))
    return comps


def detect_streaks(
    residual_mask_image: np.ndarray,
    min_pixels: int = 8,
    min_aspect_ratio: float = 4.0,
) -> list[StreakCandidate]:
    """Find elongated components and classify orientation."""
    binary = residual_mask_image.astype(bool)
    comps = _connected_components(binary)
    out: list[StreakCandidate] = []

    for pix in comps:
        if pix.shape[0] < min_pixels:
            continue
        rows = pix[:, 0]
        cols = pix[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if min(height, width) <= 0:
            continue

        if width >= height:
            aspect = width / max(height, 1)
            kind = "horizontal"
        else:
            aspect = height / max(width, 1)
            kind = "vertical"

        if aspect < min_aspect_ratio:
            continue

        score = float(np.log1p(pix.shape[0]) * aspect)
        out.append(
            StreakCandidate(
                kind=kind,
                bbox=(r0, c0, r1, c1),
                num_pixels=int(pix.shape[0]),
                aspect_ratio=float(aspect),
                score=score,
            )
        )

    return sorted(out, key=lambda x: x.score, reverse=True)
