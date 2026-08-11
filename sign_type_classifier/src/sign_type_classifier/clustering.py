"""Cross-prompt clustering for SAM3 sign detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SignDetection:
    label: str
    prompt: str
    score: float
    mask: np.ndarray
    box: np.ndarray | None = None


@dataclass(frozen=True)
class SignCandidate:
    label: str
    score: float
    margin: float
    class_scores: dict[str, float]
    mask: np.ndarray
    canonical_detection: SignDetection
    detections: tuple[SignDetection, ...]


def cluster_sign_detections(
    detections: Sequence[SignDetection],
    *,
    class_names: Sequence[str],
    min_mask_size: int,
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.80,
) -> list[SignCandidate]:
    names = tuple(str(value) for value in class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")
    if min_mask_size <= 0:
        raise ValueError(f"min_mask_size must be positive, got {min_mask_size}")
    for name, value in (("iou_threshold", iou_threshold), ("containment_threshold", containment_threshold)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")

    normalized: list[tuple[int, SignDetection]] = []
    expected_shape = None
    for index, item in enumerate(detections):
        if item.label not in names:
            continue
        score = float(item.score)
        mask = np.asarray(item.mask, dtype=bool)
        if mask.ndim != 2 or not np.isfinite(score):
            continue
        if expected_shape is None:
            expected_shape = mask.shape
        elif mask.shape != expected_shape:
            raise ValueError(f"Detection mask shape mismatch: expected {expected_shape}, got {mask.shape}")
        if int(np.count_nonzero(mask)) < min_mask_size:
            continue
        normalized.append(
            (
                index,
                SignDetection(
                    label=str(item.label),
                    prompt=str(item.prompt),
                    score=score,
                    mask=mask,
                    box=None if item.box is None else np.asarray(item.box, dtype=np.float32),
                ),
            )
        )

    ordered = sorted(normalized, key=lambda pair: (-pair[1].score, pair[0]))
    clusters: list[list[SignDetection]] = []
    for _, detection in ordered:
        best_cluster = None
        best_overlap = -1.0
        for cluster_index, members in enumerate(clusters):
            overlap = max(
                max(mask_iou(detection.mask, member.mask), mask_containment(detection.mask, member.mask))
                for member in members
            )
            matches = any(
                mask_iou(detection.mask, member.mask) >= iou_threshold
                or mask_containment(detection.mask, member.mask) >= containment_threshold
                for member in members
            )
            if matches and overlap > best_overlap:
                best_cluster = cluster_index
                best_overlap = overlap
        if best_cluster is None:
            clusters.append([detection])
        else:
            clusters[best_cluster].append(detection)

    candidates = [_candidate_from_cluster(cluster, class_names=names) for cluster in clusters]
    return sorted(candidates, key=_candidate_position_key)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask, right_mask = _mask_pair(left, right)
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left_mask & right_mask)) / float(union)


def mask_containment(left: np.ndarray, right: np.ndarray) -> float:
    left_mask, right_mask = _mask_pair(left, right)
    denominator = min(int(np.count_nonzero(left_mask)), int(np.count_nonzero(right_mask)))
    if denominator == 0:
        return 0.0
    return float(np.count_nonzero(left_mask & right_mask)) / float(denominator)


def _candidate_from_cluster(cluster: list[SignDetection], *, class_names: tuple[str, ...]) -> SignCandidate:
    canonical = max(enumerate(cluster), key=lambda pair: (pair[1].score, -pair[0]))[1]
    class_scores = {
        label: max((item.score for item in cluster if item.label == label), default=0.0)
        for label in class_names
    }
    class_index = max(range(len(class_names)), key=lambda index: (class_scores[class_names[index]], -index))
    sorted_scores = sorted(class_scores.values(), reverse=True)
    score = float(sorted_scores[0])
    margin = score - float(sorted_scores[1]) if len(sorted_scores) > 1 else score
    return SignCandidate(
        label=class_names[class_index],
        score=score,
        margin=margin,
        class_scores=class_scores,
        mask=canonical.mask.copy(),
        canonical_detection=canonical,
        detections=tuple(cluster),
    )


def _candidate_position_key(candidate: SignCandidate) -> tuple[int, int, float]:
    rows, columns = np.nonzero(candidate.mask)
    return int(rows.min()), int(columns.min()), -float(candidate.score)


def _mask_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape:
        raise ValueError(f"Mask shape mismatch: {left_mask.shape} vs {right_mask.shape}")
    return left_mask, right_mask
