from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from autolabeler.data.csv_loader import load_csv


@dataclass(frozen=True)
class VideoTimestamp:
    frame_index: int
    timestamp_us: int


@dataclass(frozen=True)
class CameraFrameMatch:
    frame_id: str
    lidar_path: str
    image_path: str | None
    lidar_timestamp_us: int | None
    video_frame_index: int | None
    video_timestamp_us: int | None
    delta_ms: float | None
    synced: bool

    def to_jsonable(self) -> dict:
        return asdict(self)


TIMESTAMP_UNITS_TO_US = {
    "s": 1_000_000.0,
    "ms": 1_000.0,
    "us": 1.0,
    "ns": 0.001,
}


def read_video_timestamps_csv(
    path: str | Path,
    *,
    frame_index_col: str,
    timestamp_col: str,
    timestamp_unit: str,
) -> list[VideoTimestamp]:
    multiplier = TIMESTAMP_UNITS_TO_US[timestamp_unit]
    out: list[VideoTimestamp] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Video timestamp CSV {path} has no header")
        for row in reader:
            frame_index = int(float(row[frame_index_col]))
            timestamp_us = int(round(float(row[timestamp_col]) * multiplier))
            out.append(VideoTimestamp(frame_index=frame_index, timestamp_us=timestamp_us))
    if not out:
        raise ValueError(f"Video timestamp CSV is empty: {path}")
    return sorted(out, key=lambda x: x.timestamp_us)


def nearest_video_timestamp(timestamps: list[VideoTimestamp], lidar_timestamp_us: int) -> VideoTimestamp:
    if not timestamps:
        raise ValueError("Video timestamp list is empty")
    lo = 0
    hi = len(timestamps)
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid].timestamp_us < lidar_timestamp_us:
            lo = mid + 1
        else:
            hi = mid
    candidates = []
    if lo < len(timestamps):
        candidates.append(timestamps[lo])
    if lo > 0:
        candidates.append(timestamps[lo - 1])
    return min(candidates, key=lambda x: abs(x.timestamp_us - lidar_timestamp_us))


def build_camera_frame_matches(
    *,
    lidar_dir: str | Path,
    out_dir: str | Path,
    video_timestamps: list[VideoTimestamp],
    max_delta_ms: float,
) -> list[CameraFrameMatch]:
    lidar_dir = Path(lidar_dir)
    out_dir = Path(out_dir)
    matches: list[CameraFrameMatch] = []
    for csv_path in sorted(lidar_dir.glob("*.csv")):
        frame = load_csv(str(csv_path), frame_id=csv_path.stem)
        if frame.timestamp_us is None:
            matches.append(
                CameraFrameMatch(
                    frame_id=csv_path.stem,
                    lidar_path=str(csv_path),
                    image_path=None,
                    lidar_timestamp_us=None,
                    video_frame_index=None,
                    video_timestamp_us=None,
                    delta_ms=None,
                    synced=False,
                )
            )
            continue
        video_ts = nearest_video_timestamp(video_timestamps, frame.timestamp_us)
        delta_ms = abs(video_ts.timestamp_us - frame.timestamp_us) / 1000.0
        synced = delta_ms <= max_delta_ms
        matches.append(
            CameraFrameMatch(
                frame_id=csv_path.stem,
                lidar_path=str(csv_path),
                image_path=str(out_dir / f"{csv_path.stem}.jpg") if synced else None,
                lidar_timestamp_us=frame.timestamp_us,
                video_frame_index=video_ts.frame_index,
                video_timestamp_us=video_ts.timestamp_us,
                delta_ms=delta_ms,
                synced=synced,
            )
        )
    return matches


def extract_synced_camera_frames(
    *,
    video_path: str | Path,
    matches: list[CameraFrameMatch],
) -> None:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        for match in matches:
            if not match.synced or match.video_frame_index is None or match.image_path is None:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, match.video_frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read video frame {match.video_frame_index} from {video_path}")
            image_path = Path(match.image_path)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame):
                raise ValueError(f"Could not write image: {image_path}")
    finally:
        capture.release()


def write_camera_frame_manifest(path: str | Path, matches: list[CameraFrameMatch]) -> None:
    payload = {
        "version": 1,
        "frames": [match.to_jsonable() for match in matches],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("prepare_camera_frames.py requires OpenCV. Install opencv-python or use an env with cv2.") from exc
    return cv2

