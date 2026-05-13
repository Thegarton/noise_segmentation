from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.bin_loader import H, W
from ..data.schemas import Box3D, LabelInstance, SequenceSample


@dataclass(frozen=True)
class NoiseComponent:
    point_indices: list[int]
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    point_count: int
    density: float
    mean_range: float
    mean_intensity: float
    bbox_size: list[float]


class NoiseAutoLabeler:
    def __init__(self, *, min_component_points: int = 3, pseudo_label_version: str = "v0_noise_rules") -> None:
        self.min_component_points = min_component_points
        self.pseudo_label_version = pseudo_label_version
        self._next_instance_id = 1

    def run(self, sample: SequenceSample, residual_mask: list[bool]) -> list[LabelInstance]:
        components = self._connected_components(sample, residual_mask)
        labels: list[LabelInstance] = []
        for component in components:
            semantic_class, class_confidence = self._classify(component)
            box = self._fit_box(sample.current.points_flat, component.point_indices, semantic_class)
            mask_confidence = float(np.clip(0.45 + 0.45 * component.density, 0.0, 0.92))
            box_confidence = self._box_confidence(component)
            final_confidence = float(np.clip(0.45 * class_confidence + 0.35 * mask_confidence + 0.20 * box_confidence, 0.0, 1.0))
            labels.append(
                LabelInstance(
                    frame_id=sample.current.frame_id,
                    semantic_class=semantic_class,
                    instance_id=self._take_instance_id(),
                    track_id=None,
                    point_indices=component.point_indices,
                    range_image_indices=[[int(i // W), int(i % W)] for i in component.point_indices],
                    box_3d=box,
                    mask_confidence=mask_confidence,
                    class_confidence=class_confidence,
                    box_confidence=box_confidence,
                    final_confidence=final_confidence,
                    provenance="rule_labeled",
                    branch_name="noise",
                    teacher_sources=["range_view_noise_rules_v0"],
                    review_status="auto_accepted" if semantic_class != "unknown_artifact" and final_confidence >= 0.62 else "needs_review",
                    pseudo_label_version=self.pseudo_label_version,
                )
            )
        return labels

    def _take_instance_id(self) -> int:
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        return instance_id

    def _connected_components(self, sample: SequenceSample, residual_mask: list[bool]) -> list[NoiseComponent]:
        points = np.asarray(sample.current.points_flat, dtype=np.float32)
        xyz = points[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        nonzero = ~((np.abs(xyz[:, 0]) < 1e-4) & (np.abs(xyz[:, 1]) < 1e-4) & (np.abs(xyz[:, 2]) < 1e-4))
        residual = np.asarray(residual_mask, dtype=bool) & finite & nonzero
        grid = residual.reshape(H, W)
        visited = np.zeros((H, W), dtype=bool)
        components: list[NoiseComponent] = []

        for row in range(H):
            for col in range(W):
                if visited[row, col] or not grid[row, col]:
                    continue
                indices = self._flood_fill(grid, visited, row, col)
                if len(indices) < self.min_component_points:
                    continue
                components.append(self._component_features(points, indices))

        return components

    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, start_row: int, start_col: int) -> list[int]:
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        indices: list[int] = []

        while stack:
            row, col = stack.pop()
            indices.append(row * W + col)
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if visited[nr, nc] or not grid[nr, nc]:
                    continue
                visited[nr, nc] = True
                stack.append((nr, nc))

        return indices

    def _component_features(self, points: np.ndarray, point_indices: list[int]) -> NoiseComponent:
        rows = np.asarray([idx // W for idx in point_indices], dtype=np.int32)
        cols = np.asarray([idx % W for idx in point_indices], dtype=np.int32)
        pts = points[np.asarray(point_indices, dtype=np.int64)]
        xyz = pts[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)
        bbox_size = (np.max(xyz, axis=0) - np.min(xyz, axis=0)).astype(float).tolist()
        row_span = int(rows.max() - rows.min() + 1)
        col_span = int(cols.max() - cols.min() + 1)
        area = max(1, row_span * col_span)

        return NoiseComponent(
            point_indices=point_indices,
            row_min=int(rows.min()),
            row_max=int(rows.max()),
            col_min=int(cols.min()),
            col_max=int(cols.max()),
            point_count=len(point_indices),
            density=float(len(point_indices) / area),
            mean_range=float(np.mean(ranges)),
            mean_intensity=float(np.mean(pts[:, 3])) if pts.shape[1] > 3 else 0.0,
            bbox_size=bbox_size,
        )

    def _classify(self, component: NoiseComponent) -> tuple[str, float]:
        row_span = component.row_max - component.row_min + 1
        col_span = component.col_max - component.col_min + 1
        aspect_col = col_span / max(1, row_span)
        aspect_row = row_span / max(1, col_span)

        if row_span <= 2 and col_span >= 8:
            return "horizontal_crosstalk_noise", float(np.clip(0.58 + min(0.22, aspect_col / 50.0), 0.0, 0.86))
        if col_span <= 2 and row_span >= 8:
            return "vertical_crosstalk_noise", float(np.clip(0.58 + min(0.22, aspect_row / 50.0), 0.0, 0.86))
        if component.mean_range >= 55.0 and component.density <= 0.55:
            return "long_distance_noise", 0.72
        if component.mean_range <= 8.0 and row_span >= 3 and col_span >= 3:
            return "near_range_layered_noise", 0.68
        if component.density <= 0.45 and component.point_count >= self.min_component_points:
            return "dust_noise", 0.62
        return "unknown_artifact", 0.45

    def _fit_box(self, points_flat: np.ndarray, point_indices: list[int], semantic_class: str) -> Box3D:
        pts = np.asarray(points_flat, dtype=np.float32)[np.asarray(point_indices, dtype=np.int64), :3]
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        center = ((mins + maxs) * 0.5).astype(float).tolist()
        size = np.maximum(maxs - mins, np.asarray([0.05, 0.05, 0.05], dtype=np.float32)).astype(float).tolist()
        return Box3D(center=center, size=size, yaw=0.0, box_type="noise_adaptive_aabb")

    def _box_confidence(self, component: NoiseComponent) -> float:
        row_span = component.row_max - component.row_min + 1
        col_span = component.col_max - component.col_min + 1
        shape_score = min(1.0, max(row_span, col_span) / 12.0)
        count_score = min(1.0, component.point_count / 24.0)
        return float(np.clip(0.35 + 0.35 * shape_score + 0.20 * count_score, 0.0, 0.9))
