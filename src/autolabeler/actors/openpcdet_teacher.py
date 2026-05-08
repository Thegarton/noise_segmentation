from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


OPENPCDET_TEACHER_SOURCE = "openpcdet_centerpoint_pointpillar_nuscenes"

CLASS_REMAP = {
    "car": "CAR",
    "truck": "TRUCK_BUS",
    "bus": "TRUCK_BUS",
    "motorcycle": "MOTORCYCLE",
    "bicycle": "CYCLIST",
    "pedestrian": "PEDESTRIAN",
}


@dataclass(frozen=True)
class OpenPCDetPrediction:
    frame_id: str
    semantic_class: str
    center: list[float]
    size: list[float]
    yaw: float
    score: float
    raw_class_name: str
    source: str = OPENPCDET_TEACHER_SOURCE


class OpenPCDetPredictionStore:
    def __init__(self, predictions_by_frame: dict[str, list[OpenPCDetPrediction]] | None = None) -> None:
        self._predictions_by_frame = predictions_by_frame or {}

    @classmethod
    def from_jsonl(cls, path: str | None) -> "OpenPCDetPredictionStore":
        if path is None:
            return cls()

        by_frame: dict[str, list[OpenPCDetPrediction]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                frame_id = str(record["frame_id"])
                source = record.get("source", OPENPCDET_TEACHER_SOURCE)
                predictions: list[OpenPCDetPrediction] = []
                for item in record.get("predictions", []):
                    raw_class_name = str(item["class_name"])
                    semantic_class = remap_openpcdet_class(raw_class_name)
                    if semantic_class is None:
                        continue
                    box = item["box_3d"]
                    center = [float(x) for x in box["center"]]
                    size = [float(x) for x in box.get("size", [0.0, 0.0, 0.0])]
                    predictions.append(
                        OpenPCDetPrediction(
                            frame_id=frame_id,
                            semantic_class=semantic_class,
                            center=center,
                            size=size,
                            yaw=float(box["yaw"]),
                            score=float(item["score"]),
                            raw_class_name=raw_class_name,
                            source=source,
                        )
                    )
                by_frame[frame_id] = predictions
        return cls(by_frame)

    def get(self, frame_id: str) -> list[OpenPCDetPrediction]:
        return self._predictions_by_frame.get(frame_id, [])


def remap_openpcdet_class(class_name: str) -> str | None:
    return CLASS_REMAP.get(class_name.strip().lower())


def write_openpcdet_predictions_jsonl(path: str, records: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
