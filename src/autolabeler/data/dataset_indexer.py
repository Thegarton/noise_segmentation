from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    lidar_path: str
    timestamp: float


<<<<<<< codex/build-offline-auto-labeling-system-wl2yjl
def build_dataset_index(input_dir: str, input_format: str = "auto") -> list[FrameRecord]:
    base = Path(input_dir)
    if input_format == "auto":
        paths = sorted([*base.glob("*.bin"), *base.glob("*.csv")])
    elif input_format == "bin":
        paths = sorted(base.glob("*.bin"))
    elif input_format == "csv":
        paths = sorted(base.glob("*.csv"))
    else:
        raise ValueError(f"Unsupported input_format={input_format}")

=======
def build_dataset_index(input_dir: str) -> list[FrameRecord]:
    paths = sorted(Path(input_dir).glob("*.bin"))
>>>>>>> main
    records: list[FrameRecord] = []
    for i, p in enumerate(paths):
        records.append(FrameRecord(frame_id=p.stem, lidar_path=str(p), timestamp=float(i)))
    return records
