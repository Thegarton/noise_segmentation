from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from ..data.bin_loader import H, W
from ..data.schemas import OrganizedLiDARFrame


@dataclass
class ActorProposal:
    semantic_class: str
    center: list[float]
    yaw: float
    confidence: float
    point_indices: list[int]
    range_image_indices: list[list[int]]
    observed_size: list[float]


class RangePillarCenterPointLite:
    """Numpy proposal generator that keeps the organized range-view mapping.

    This is a deterministic Offline v0 stand-in for a trained CenterPoint head:
    range-view components become object-center proposals, then a fixed-size
    decoder in ActorAutoLabeler turns them into actor boxes.
    """

    def __init__(
        self,
        fixed_actor_sizes: dict[str, list[float]],
        *,
        min_range_m: float = 0.8,
        max_range_m: float = 120.0,
        min_component_points: int = 8,
        max_component_points: int = 12_000,
    ) -> None:
        self.fixed_actor_sizes = fixed_actor_sizes
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m
        self.min_component_points = min_component_points
        self.max_component_points = max_component_points

    def predict(self, frame: OrganizedLiDARFrame) -> list[ActorProposal]:
        points = np.asarray(frame.points_range, dtype=np.float32)
        xyz = points[:, :, :3]
        ranges = np.linalg.norm(xyz, axis=2)
        finite = np.isfinite(xyz).all(axis=2)
        valid = finite & (ranges >= self.min_range_m) & (ranges <= self.max_range_m)
        valid &= ~((np.abs(xyz[:, :, 0]) < 1e-4) & (np.abs(xyz[:, :, 1]) < 1e-4) & (np.abs(xyz[:, :, 2]) < 1e-4))

        z_values = xyz[:, :, 2][valid]
        if z_values.size == 0:
            return []

        ground_z = float(np.percentile(z_values, 8))
        nonground = valid & (xyz[:, :, 2] > ground_z + 0.12)
        components = self._connected_components(nonground, xyz, ranges)
        proposals = [p for comp in components for p in self._component_to_proposals(comp, xyz, ranges)]
        return self._nms(proposals)

    def _connected_components(self, mask: np.ndarray, xyz: np.ndarray, ranges: np.ndarray) -> list[np.ndarray]:
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[np.ndarray] = []

        for row in range(H):
            for col in range(W):
                if visited[row, col] or not mask[row, col]:
                    continue

                coords: list[tuple[int, int]] = []
                queue: deque[tuple[int, int]] = deque([(row, col)])
                visited[row, col] = True

                while queue:
                    r, c = queue.popleft()
                    coords.append((r, c))
                    base_range = float(ranges[r, c])
                    base_xyz = xyz[r, c]
                    range_gate = max(0.45, 0.035 * base_range)
                    distance_gate = max(0.55, 0.045 * base_range)

                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if nr < 0 or nr >= H or nc < 0 or nc >= W:
                            continue
                        if visited[nr, nc] or not mask[nr, nc]:
                            continue
                        if abs(float(ranges[nr, nc]) - base_range) > range_gate:
                            continue
                        if float(np.linalg.norm(xyz[nr, nc] - base_xyz)) > distance_gate:
                            continue
                        visited[nr, nc] = True
                        queue.append((nr, nc))

                if self.min_component_points <= len(coords) <= self.max_component_points:
                    components.append(np.asarray(coords, dtype=np.int32))

        return components

    def _component_to_proposals(self, coords: np.ndarray, xyz: np.ndarray, ranges: np.ndarray) -> list[ActorProposal]:
        pts = xyz[coords[:, 0], coords[:, 1]]
        if pts.shape[0] < self.min_component_points:
            return []

        yaw, length, width = _pca_xy_size(pts[:, :2])
        height = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))
        observed_size = [length, width, height]
        semantic_class, class_score = self._classify(observed_size, pts.shape[0])
        if semantic_class is None:
            return []

        fixed_size = self.fixed_actor_sizes[semantic_class]
        z_center = float(np.min(pts[:, 2]) + fixed_size[2] * 0.5)
        center = [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), z_center]

        density_score = min(1.0, math.log1p(pts.shape[0]) / math.log1p(180.0))
        compactness = _compactness_score(observed_size, fixed_size)
        range_score = 1.0 - min(0.45, float(np.mean(ranges[coords[:, 0], coords[:, 1]])) / 250.0)
        confidence = float(np.clip(0.45 * class_score + 0.25 * density_score + 0.2 * compactness + 0.1 * range_score, 0.0, 1.0))
        if confidence < 0.28:
            return []

        point_indices = (coords[:, 0] * W + coords[:, 1]).astype(int).tolist()
        range_image_indices = coords.astype(int).tolist()
        return [
            ActorProposal(
                semantic_class=semantic_class,
                center=center,
                yaw=float(yaw),
                confidence=confidence,
                point_indices=point_indices,
                range_image_indices=range_image_indices,
                observed_size=observed_size,
            )
        ]

    def _classify(self, observed_size: list[float], point_count: int) -> tuple[str | None, float]:
        length, width, height = observed_size
        if height < 0.35 and point_count < 40:
            return None, 0.0

        if length > 5.8 or width > 2.2 or height > 2.2:
            return "TRUCK_BUS", min(0.92, 0.55 + length / 18.0 + height / 8.0)
        if length > 2.0 and width > 0.9 and height > 0.55:
            car_score = 0.72 - min(0.25, abs(length - 3.4) / 8.0) - min(0.15, abs(width - 1.5) / 6.0)
            return "CAR", float(np.clip(car_score, 0.35, 0.86))
        if height > 1.0 and length < 1.25 and width < 1.15:
            return "PEDESTRIAN", float(np.clip(0.72 - abs(width - 0.55) * 0.25, 0.38, 0.84))
        if height > 0.8 and length <= 2.8 and width <= 1.35:
            if length >= 1.75 and width <= 0.95:
                return "MOTORCYCLE", 0.62
            return "CYCLIST", 0.58
        return None, 0.0

    def _nms(self, proposals: list[ActorProposal]) -> list[ActorProposal]:
        kept: list[ActorProposal] = []
        for prop in sorted(proposals, key=lambda x: x.confidence, reverse=True):
            center = np.asarray(prop.center[:2], dtype=np.float32)
            duplicate = False
            for prev in kept:
                prev_center = np.asarray(prev.center[:2], dtype=np.float32)
                gate = 0.35 * min(self.fixed_actor_sizes[prop.semantic_class][0], self.fixed_actor_sizes[prev.semantic_class][0])
                if float(np.linalg.norm(center - prev_center)) < max(0.75, gate):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(prop)
        return kept


def _pca_xy_size(xy: np.ndarray) -> tuple[float, float, float]:
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    if xy.shape[0] < 3:
        return 0.0, 0.0, 0.0
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = centered @ axes
    size = np.max(projected, axis=0) - np.min(projected, axis=0)
    yaw = math.atan2(float(axes[1, 0]), float(axes[0, 0]))
    length = float(max(size[0], size[1]))
    width = float(min(size[0], size[1]))
    return yaw, length, width


def _compactness_score(observed_size: list[float], fixed_size: list[float]) -> float:
    obs = np.asarray(observed_size, dtype=np.float32)
    fixed = np.asarray(fixed_size, dtype=np.float32)
    upper_ok = np.all(obs <= fixed * np.asarray([1.45, 1.6, 1.3], dtype=np.float32))
    if not upper_ok:
        return 0.2
    ratio = np.clip(obs / fixed, 0.0, 1.0)
    return float(np.clip(0.35 + 0.65 * np.mean(ratio), 0.0, 1.0))
