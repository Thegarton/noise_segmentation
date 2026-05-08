from __future__ import annotations

import math

import numpy as np

from ..data.bin_loader import W
from ..data.schemas import LabelInstance, SequenceSample
from ..fusion.confidence import compute_final_confidence
from .centerpoint_lite import ActorProposal, RangePillarCenterPointLite
from .fixed_box_decoder import FixedBoxDecoder, load_fixed_actor_sizes
from .openpcdet_teacher import OpenPCDetPrediction, OpenPCDetPredictionStore
from .tracker import SimpleActorTracker


class ActorAutoLabeler:
    FIXED_CLASSES = ["TRUCK_BUS", "CAR", "CYCLIST", "MOTORCYCLE", "PEDESTRIAN"]

    def __init__(
        self,
        *,
        class_config_path: str | None = None,
        openpcdet_predictions_path: str | None = None,
        score_threshold: float = 0.32,
        pseudo_label_version: str = "v0_actor_heuristic",
        use_heuristic_fallback: bool = True,
    ) -> None:
        self.fixed_sizes = load_fixed_actor_sizes(class_config_path)
        self.decoder = FixedBoxDecoder(self.fixed_sizes)
        self.proposal_model = RangePillarCenterPointLite(self.fixed_sizes)
        self.openpcdet_predictions = OpenPCDetPredictionStore.from_jsonl(openpcdet_predictions_path)
        self.tracker = SimpleActorTracker()
        self.score_threshold = score_threshold
        self.pseudo_label_version = pseudo_label_version
        self.use_heuristic_fallback = use_heuristic_fallback
        self._next_instance_id = 1

    def run(self, sample: SequenceSample) -> list[LabelInstance]:
        current_proposals = self.proposal_model.predict(sample.current) if self.use_heuristic_fallback else []
        temporal_proposals = {
            "past": [self.proposal_model.predict(frame) for frame in sample.past[-2:]] if self.use_heuristic_fallback else [],
            "future": [self.proposal_model.predict(frame) for frame in sample.future[:2]] if self.use_heuristic_fallback else [],
        }

        detections: list[dict] = []
        for prediction in self.openpcdet_predictions.get(sample.current.frame_id):
            temporal_consistency = self._teacher_temporal_consistency(prediction, sample)
            confidence = min(1.0, prediction.score * (0.85 + 0.25 * temporal_consistency))
            if confidence < self.score_threshold:
                continue
            detections.append(
                {
                    "semantic_class": prediction.semantic_class,
                    "center": prediction.center,
                    "yaw": prediction.yaw,
                    "confidence": confidence,
                    "temporal_consistency": temporal_consistency,
                    "proposal": None,
                    "provenance": "lidar_teacher",
                    "teacher_sources": [prediction.source],
                    "box_source": "openpcdet",
                }
            )

        for proposal in current_proposals:
            temporal_consistency = self._temporal_consistency(proposal, temporal_proposals)
            confidence = min(1.0, proposal.confidence * (0.82 + 0.28 * temporal_consistency))
            if confidence < self.score_threshold:
                continue

            detections.append(
                {
                    "semantic_class": proposal.semantic_class,
                    "center": proposal.center,
                    "yaw": proposal.yaw,
                    "confidence": confidence,
                    "temporal_consistency": temporal_consistency,
                    "proposal": proposal,
                    "provenance": "student_predicted",
                    "teacher_sources": ["range_pillar_centerpoint_lite"],
                    "box_source": "heuristic",
                }
            )

        detections = self._deduplicate_detections(detections)
        tracked = self.tracker.assign(detections)
        labels: list[LabelInstance] = []
        for det in tracked:
            proposal: ActorProposal | None = det["proposal"]
            center = det["smoothed_center"]
            yaw = det["smoothed_yaw"]
            box = self.decoder.decode(det["semantic_class"], center, yaw)
            point_indices, range_indices, mask_conf = _points_inside_box(sample.current.points_flat, box.center, box.size, box.yaw)

            class_conf = float(det["confidence"])
            box_conf = float(min(1.0, 0.55 + 0.25 * det["temporal_consistency"] + 0.2 * det["track_stability"]))
            if not point_indices and proposal is not None:
                point_indices = proposal.point_indices
                range_indices = proposal.range_image_indices
                mask_conf = min(mask_conf, 0.45)

            final_conf = compute_final_confidence(
                mask_conf=mask_conf,
                class_conf=class_conf,
                box_conf=box_conf,
                temporal_consistency=float(det["temporal_consistency"]),
            )
            labels.append(
                LabelInstance(
                    frame_id=sample.current.frame_id,
                    semantic_class=det["semantic_class"],
                    instance_id=self._take_instance_id(),
                    track_id=int(det["track_id"]),
                    point_indices=point_indices,
                    range_image_indices=range_indices,
                    box_3d=box,
                    mask_confidence=float(mask_conf),
                    class_confidence=class_conf,
                    box_confidence=box_conf,
                    final_confidence=float(final_conf),
                    provenance=det["provenance"],
                    branch_name="actor",
                    teacher_sources=det["teacher_sources"],
                    review_status="auto_accepted" if final_conf >= 0.72 else "needs_review",
                    pseudo_label_version=self.pseudo_label_version,
                )
            )

        return labels

    def _take_instance_id(self) -> int:
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        return instance_id

    def _temporal_consistency(self, proposal: ActorProposal, temporal_proposals: dict[str, list[list[ActorProposal]]]) -> float:
        support = 0.0
        total = 0
        center = np.asarray(proposal.center[:2], dtype=np.float32)
        size = self.fixed_sizes[proposal.semantic_class]
        gate = max(1.5, 0.45 * size[0])

        for branch in ("past", "future"):
            for proposals in temporal_proposals[branch]:
                total += 1
                best = 0.0
                for other in proposals:
                    if other.semantic_class != proposal.semantic_class:
                        continue
                    dist = float(np.linalg.norm(center - np.asarray(other.center[:2], dtype=np.float32)))
                    if dist <= gate:
                        best = max(best, 1.0 - dist / gate)
                support += best

        if total == 0:
            return 0.5
        return float(np.clip(0.35 + 0.65 * (support / total), 0.0, 1.0))

    def _teacher_temporal_consistency(self, prediction: OpenPCDetPrediction, sample: SequenceSample) -> float:
        support = 0.0
        total = 0
        center = np.asarray(prediction.center[:2], dtype=np.float32)
        size = self.fixed_sizes[prediction.semantic_class]
        gate = max(1.5, 0.45 * size[0])

        for frames in (sample.past[-2:], sample.future[:2]):
            for frame in frames:
                total += 1
                best = 0.0
                for other in self.openpcdet_predictions.get(frame.frame_id):
                    if other.semantic_class != prediction.semantic_class:
                        continue
                    dist = float(np.linalg.norm(center - np.asarray(other.center[:2], dtype=np.float32)))
                    if dist <= gate:
                        best = max(best, 1.0 - dist / gate)
                support += best

        if total == 0:
            return 0.5
        return float(np.clip(0.35 + 0.65 * (support / total), 0.0, 1.0))

    def _deduplicate_detections(self, detections: list[dict]) -> list[dict]:
        kept: list[dict] = []
        source_priority = {"openpcdet": 1, "heuristic": 0}
        ordered = sorted(
            detections,
            key=lambda x: (source_priority.get(x["box_source"], 0), x["confidence"]),
            reverse=True,
        )
        for det in ordered:
            center = np.asarray(det["center"][:2], dtype=np.float32)
            size = self.fixed_sizes[det["semantic_class"]]
            duplicate = False
            for prev in kept:
                prev_center = np.asarray(prev["center"][:2], dtype=np.float32)
                gate = max(0.75, 0.35 * min(size[0], self.fixed_sizes[prev["semantic_class"]][0]))
                if float(np.linalg.norm(center - prev_center)) < gate:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept


def _points_inside_box(points_flat: np.ndarray, center: list[float], size: list[float], yaw: float) -> tuple[list[int], list[list[int]], float]:
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
    half_l = float(size[0]) * 0.5
    half_w = float(size[1]) * 0.5
    half_h = float(size[2]) * 0.5

    inside = (
        valid
        & (np.abs(local_x) <= half_l)
        & (np.abs(local_y) <= half_w)
        & (np.abs(shifted[:, 2]) <= half_h)
    )
    indices = np.flatnonzero(inside).astype(int).tolist()
    range_indices = [[int(i // W), int(i % W)] for i in indices]

    expected_points = max(12.0, (float(size[0]) * float(size[1]) * float(size[2])) * 2.5)
    mask_conf = min(1.0, len(indices) / expected_points)
    return indices, range_indices, float(mask_conf)
