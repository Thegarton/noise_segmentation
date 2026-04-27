"""Generate residual cluster proposals with DBSCAN and descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None


@dataclass
class ClusterProposal:
    cluster_id: int
    point_indices: np.ndarray
    family_hint: str
    descriptor: dict[str, Any]


def _simple_radius_cluster(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = xyz.shape[0]
    labels = -np.ones(n, dtype=np.int32)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        d2 = np.sum((xyz - xyz[i]) ** 2, axis=1)
        neigh = np.where(d2 <= eps * eps)[0]
        if neigh.size < min_samples:
            continue
        labels[neigh] = cluster_id
        cluster_id += 1
    return labels


def _pca_shape_features(xyz: np.ndarray) -> dict[str, float]:
    if xyz.shape[0] < 3:
        return {"linearity": 0.0, "planarity": 0.0, "scattering": 1.0}
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(xyz.shape[0] - 1, 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1] + 1e-9
    l1, l2, l3 = eigvals
    return {
        "linearity": float((l1 - l2) / l1),
        "planarity": float((l2 - l3) / l1),
        "scattering": float(l3 / l1),
    }


def _cluster_descriptor(points: np.ndarray) -> dict[str, Any]:
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    dims = maxs - mins
    desc = {
        "num_points": int(points.shape[0]),
        "bbox_center": ((mins + maxs) / 2.0).tolist(),
        "bbox_size": dims.tolist(),
        "mean_intensity": float(points[:, 3].mean()) if points.shape[1] > 3 else 0.0,
        "std_intensity": float(points[:, 3].std()) if points.shape[1] > 3 else 0.0,
        "mean_range": float(np.linalg.norm(xyz, axis=1).mean()),
        "std_range": float(np.linalg.norm(xyz, axis=1).std()),
    }
    desc.update(_pca_shape_features(xyz))
    return desc


def generate_dbscan_proposals(
    residual_points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 8,
    family_hint: str = "unknown_artifact",
    precomputed_labels: Optional[np.ndarray] = None,
) -> list[ClusterProposal]:
    """Run DBSCAN and package cluster proposals with descriptors."""
    if residual_points.ndim != 2 or residual_points.shape[1] < 3:
        raise ValueError("residual_points must be [N, C] with C>=3")
    if residual_points.shape[0] == 0:
        return []

    if precomputed_labels is not None:
        labels = precomputed_labels.astype(np.int32)
    elif DBSCAN is not None:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(residual_points[:, :3])
    else:
        labels = _simple_radius_cluster(residual_points[:, :3], eps=eps, min_samples=min_samples)

    proposals: list[ClusterProposal] = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        cluster_pts = residual_points[idx]
        desc = _cluster_descriptor(cluster_pts)
        proposals.append(
            ClusterProposal(
                cluster_id=int(cid),
                point_indices=idx,
                family_hint=family_hint,
                descriptor=desc,
            )
        )
    return proposals
