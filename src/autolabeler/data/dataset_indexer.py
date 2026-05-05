from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    lidar_path: str
    timestamp: float


def build_dataset_index(input_dir: str) -> list[FrameRecord]:
    paths = sorted(Path(input_dir).glob("*.bin"))
    records: list[FrameRecord] = []
    for i, p in enumerate(paths):
        records.append(FrameRecord(frame_id=p.stem, lidar_path=str(p), timestamp=float(i)))
    return records
