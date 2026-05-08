from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ActorTrackState:
    track_id: int
    semantic_class: str
    center: np.ndarray
    yaw: float
    age: int = 1
    missed: int = 0


def smooth_yaw(previous: float, current: float, alpha: float = 0.55) -> float:
    sin_v = (1.0 - alpha) * math.sin(previous) + alpha * math.sin(current)
    cos_v = (1.0 - alpha) * math.cos(previous) + alpha * math.cos(current)
    return math.atan2(sin_v, cos_v)


class SimpleActorTracker:
    def __init__(self, max_match_distance_m: float = 3.0, max_missed: int = 5, smooth_alpha: float = 0.55) -> None:
        self.max_match_distance_m = max_match_distance_m
        self.max_missed = max_missed
        self.smooth_alpha = smooth_alpha
        self._next_track_id = 1
        self._tracks: dict[int, ActorTrackState] = {}

    def assign(self, detections: list[dict]) -> list[dict]:
        unmatched_track_ids = set(self._tracks)
        assigned: list[dict] = []

        for det in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            center = np.asarray(det["center"], dtype=np.float32)
            best_track_id: int | None = None
            best_dist = float("inf")

            for track_id in list(unmatched_track_ids):
                track = self._tracks[track_id]
                if track.semantic_class != det["semantic_class"]:
                    continue
                dist = float(np.linalg.norm(track.center[:2] - center[:2]))
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is None or best_dist > self.max_match_distance_m:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = ActorTrackState(
                    track_id=track_id,
                    semantic_class=det["semantic_class"],
                    center=center,
                    yaw=float(det["yaw"]),
                )
            else:
                track_id = best_track_id
                track = self._tracks[track_id]
                alpha = self.smooth_alpha
                track.center = (1.0 - alpha) * track.center + alpha * center
                track.yaw = smooth_yaw(track.yaw, float(det["yaw"]), alpha=alpha)
                track.age += 1
                track.missed = 0
                unmatched_track_ids.remove(track_id)

            out = dict(det)
            track = self._tracks[track_id]
            out["track_id"] = track_id
            out["smoothed_center"] = track.center.astype(float).tolist()
            out["smoothed_yaw"] = float(track.yaw)
            out["track_stability"] = min(1.0, track.age / 4.0)
            assigned.append(out)

        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.missed += 1

        stale = [track_id for track_id, track in self._tracks.items() if track.missed > self.max_missed]
        for track_id in stale:
            del self._tracks[track_id]

        return assigned
