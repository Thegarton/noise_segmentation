from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass, asdict
from math import cos, radians, sin
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InsPose:
    timestamp_us: int
    source_timestamp_us: int
    delta_us: int
    translation: list[float]
    rotation_matrix: list[list[float]]
    kitti_pose: list[float]
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None

    def to_jsonable(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InsSample:
    timestamp_us: int
    translation: list[float]
    rotation_matrix: list[list[float]]
    latitude: float | None
    longitude: float | None
    elevation: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None


class InsPoseIndex:
    def __init__(self, samples: list[InsSample]):
        if not samples:
            raise ValueError("INS pose index is empty")
        self.samples = sorted(samples, key=lambda sample: sample.timestamp_us)
        self.timestamps = [sample.timestamp_us for sample in self.samples]

    def nearest(self, timestamp_us: int) -> InsPose:
        pos = bisect.bisect_left(self.timestamps, timestamp_us)
        candidates = []
        if pos < len(self.samples):
            candidates.append(self.samples[pos])
        if pos > 0:
            candidates.append(self.samples[pos - 1])
        sample = min(candidates, key=lambda candidate: abs(candidate.timestamp_us - timestamp_us))
        delta_us = int(sample.timestamp_us - timestamp_us)
        return InsPose(
            timestamp_us=int(timestamp_us),
            source_timestamp_us=int(sample.timestamp_us),
            delta_us=delta_us,
            translation=sample.translation,
            rotation_matrix=sample.rotation_matrix,
            kitti_pose=_flatten_kitti_pose(sample.rotation_matrix, sample.translation),
            latitude=sample.latitude,
            longitude=sample.longitude,
            elevation=sample.elevation,
            yaw=sample.yaw,
            pitch=sample.pitch,
            roll=sample.roll,
        )


def load_ins_pose_index(path: str) -> InsPoseIndex:
    rows = _read_ins_rows(path)
    origin = _lat_lon_origin(rows)
    samples = [_row_to_sample(row, origin=origin) for row in rows]
    return InsPoseIndex(samples)


def write_kitti_pose(path: str, pose: InsPose) -> None:
    write_kitti_pose_values(path, pose.kitti_pose)


def write_kitti_pose_values(path: str, kitti_pose: list[float]) -> None:
    if len(kitti_pose) != 12:
        raise ValueError(f"KITTI pose must have 12 values, got {len(kitti_pose)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(_fmt(value) for value in kitti_pose) + "\n", encoding="utf-8")


def _read_ins_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"INS file {path} has no header")
        return [{_clean_key(key): value for key, value in row.items()} for row in reader]


def _row_to_sample(row: dict[str, str], *, origin: tuple[float, float, float] | None) -> InsSample:
    secs = _required_int(row, "secs")
    nsecs = _required_int(row, "nsecs")
    timestamp_us = secs * 1_000_000 + nsecs // 1000

    lat = _optional_float(row.get("latitude"))
    lon = _optional_float(row.get("longitude"))
    elevation = _optional_float(row.get("elevation"))
    translation = _translation_from_row(row, lat=lat, lon=lon, elevation=elevation, origin=origin)
    roll = _optional_float(row.get("attitude_X"))
    pitch = _optional_float(row.get("attitude_Y"))
    yaw = _optional_float(row.get("attitude_Z"))
    rotation = _rotation_matrix_zyx(roll or 0.0, pitch or 0.0, yaw or 0.0)
    return InsSample(
        timestamp_us=timestamp_us,
        translation=translation,
        rotation_matrix=rotation.tolist(),
        latitude=lat,
        longitude=lon,
        elevation=elevation,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def _translation_from_row(
    row: dict[str, str],
    *,
    lat: float | None,
    lon: float | None,
    elevation: float | None,
    origin: tuple[float, float, float] | None,
) -> list[float]:
    utm = [
        _optional_float(row.get("utmPosition_X")) or 0.0,
        _optional_float(row.get("utmPosition_Y")) or 0.0,
        _optional_float(row.get("utmPosition_Z")) or 0.0,
    ]
    if np.linalg.norm(np.asarray(utm, dtype=np.float64)) > 1e-6:
        return utm
    if origin is not None and lat is not None and lon is not None:
        return _lat_lon_to_local_enu(lat, lon, elevation or origin[2], origin)
    return [0.0, 0.0, elevation or 0.0]


def _lat_lon_origin(rows: list[dict[str, str]]) -> tuple[float, float, float] | None:
    for row in rows:
        lat = _optional_float(row.get("latitude"))
        lon = _optional_float(row.get("longitude"))
        elevation = _optional_float(row.get("elevation"))
        if lat is not None and lon is not None:
            return lat, lon, elevation or 0.0
    return None


def _lat_lon_to_local_enu(
    lat: float,
    lon: float,
    elevation: float,
    origin: tuple[float, float, float],
) -> list[float]:
    origin_lat, origin_lon, origin_elevation = origin
    earth_radius_m = 6_378_137.0
    east = radians(lon - origin_lon) * earth_radius_m * cos(radians(origin_lat))
    north = radians(lat - origin_lat) * earth_radius_m
    up = elevation - origin_elevation
    return [east, north, up]


def _rotation_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _flatten_kitti_pose(rotation: list[list[float]], translation: list[float]) -> list[float]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose[:3, :].reshape(-1).tolist()


def _required_int(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"INS row is missing required column {key!r}")
    return int(float(value))


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _clean_key(key: str | None) -> str:
    return "" if key is None else key.strip()


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"
