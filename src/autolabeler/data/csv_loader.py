from __future__ import annotations

import csv

import numpy as np

from .bin_loader import C, H, W
from .schemas import OrganizedLiDARFrame


REQUIRED_COLUMNS = ("x", "y", "z", "intensity")
DIS_COLS = tuple(f"dis{i}" for i in range(H))
INT_COLS = tuple(f"int{i}" for i in range(H))
REF_COLS = tuple(f"ref{i}" for i in range(H))

# Sensor geometry used by the raw packet CSV format in this dataset.
DEFAULT_ELEVATION_DEG = (7.85, -12.85)
DEFAULT_AZIMUTH_DEG = (60.0, -60.0)


def load_csv(path: str, frame_id: str) -> OrganizedLiDARFrame:
    rows, fieldnames = _read_csv_rows(path)
    if _has_columns(fieldnames, REQUIRED_COLUMNS):
        return _load_xyz_csv(rows, path=path, frame_id=frame_id)
    if _has_columns(fieldnames, DIS_COLS) and _has_columns(fieldnames, INT_COLS):
        return _load_raw_packet_csv(rows, path=path, frame_id=frame_id)
    raise ValueError(
        f"Unsupported CSV layout in {path}. Expected either {REQUIRED_COLUMNS} "
        f"or raw packet columns dis0..dis{H - 1}, int0..int{H - 1}."
    )


def _read_csv_rows(path: str) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header")
        fieldnames = tuple(name.strip() for name in reader.fieldnames)
        rows = [{_clean_key(k): v for k, v in row.items()} for row in reader]
    return rows, fieldnames


def _clean_key(key: str | None) -> str:
    return "" if key is None else key.strip()


def _has_columns(fieldnames: tuple[str, ...], required: tuple[str, ...]) -> bool:
    available = {name.strip() for name in fieldnames}
    return all(name in available for name in required)


def _load_xyz_csv(rows: list[dict[str, str]], *, path: str, frame_id: str) -> OrganizedLiDARFrame:
    xyz_i = []
    timestamp_s = None
    timestamp_u = None
    background_light_intensity = None

    for row in rows:
        xyz_i.append([_to_float(row["x"]), _to_float(row["y"]), _to_float(row["z"]), _to_float(row["intensity"])])
        if timestamp_s is None and row.get("timestamp_s") not in (None, ""):
            timestamp_s = int(float(row["timestamp_s"]))
        if timestamp_u is None and row.get("timestamp_u") not in (None, ""):
            timestamp_u = int(float(row["timestamp_u"]))
        if background_light_intensity is None and row.get("background_light_intensity") not in (None, ""):
            background_light_intensity = float(row["background_light_intensity"])

    arr = np.asarray(xyz_i, dtype=np.float32)
    if arr.shape[0] != H * W:
        raise ValueError(f"CSV {path} has {arr.shape[0]} points, expected {H * W} for organized {H}x{W} LiDAR")

    arr = arr.reshape(H, W, C)
    return _build_frame(
        frame_id=frame_id,
        arr=arr,
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        background_light_intensity=background_light_intensity,
        meta={"source_format": "csv", "csv_layout": "xyz"},
    )


def _load_raw_packet_csv(rows: list[dict[str, str]], *, path: str, frame_id: str) -> OrganizedLiDARFrame:
    if len(rows) != W:
        raise ValueError(f"Raw packet CSV {path} has {len(rows)} rows, expected {W} azimuth packets")

    rows = _sort_raw_rows(rows)
    distance = _matrix_from_columns(rows, DIS_COLS)
    reflectivity = _matrix_from_columns(rows, REF_COLS) if all(col in rows[0] for col in REF_COLS) else None
    intensity = _matrix_from_columns(rows, INT_COLS)

    np.clip(distance, 0.0, None, out=distance)
    valid = np.isfinite(distance) & (distance > 0.0)

    elevation = np.deg2rad(np.linspace(DEFAULT_ELEVATION_DEG[0], DEFAULT_ELEVATION_DEG[1], H, dtype=np.float32))
    azimuth = np.deg2rad(np.linspace(DEFAULT_AZIMUTH_DEG[0], DEFAULT_AZIMUTH_DEG[1], W, dtype=np.float32))

    cos_el = np.cos(elevation)[None, :]
    sin_el = np.sin(elevation)[None, :]
    cos_az = np.cos(azimuth)[:, None]
    sin_az = np.sin(azimuth)[:, None]

    x = distance * cos_el * cos_az
    y = distance * cos_el * sin_az
    z = distance * sin_el
    strength = reflectivity if reflectivity is not None else intensity

    # Raw packet matrices are [azimuth, beam]. The organized frame contract is [beam, azimuth, channel].
    arr = np.stack([x, y, z, strength], axis=-1).transpose(1, 0, 2).astype(np.float32, copy=False)
    arr[~valid.T, :3] = 0.0
    arr[~valid.T, 3] = 0.0

    timestamp_s, timestamp_u, timestamp_us = _timestamps_from_raw_rows(rows)
    background_light_intensity = float(np.nanmean(intensity[valid])) if np.any(valid) else None

    return _build_frame(
        frame_id=frame_id,
        arr=arr,
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        timestamp_us=timestamp_us,
        background_light_intensity=background_light_intensity,
        meta={
            "source_format": "csv",
            "csv_layout": "raw_packet",
            "distance_channel": "dis",
            "strength_channel": "ref" if reflectivity is not None else "int",
            "elevation_deg": list(DEFAULT_ELEVATION_DEG),
            "azimuth_deg": list(DEFAULT_AZIMUTH_DEG),
        },
    )


def _sort_raw_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if all(row.get("packSeqNum") not in (None, "") for row in rows):
        return sorted(rows, key=lambda row: int(float(row["packSeqNum"])))
    if all(row.get("axIdx0") not in (None, "") for row in rows):
        return sorted(rows, key=lambda row: int(float(row["axIdx0"])))
    return rows


def _matrix_from_columns(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[_to_float(row.get(col, "0")) for col in columns] for row in rows], dtype=np.float32)


def _timestamps_from_raw_rows(rows: list[dict[str, str]]) -> tuple[int | None, int | None, int | None]:
    first = rows[0] if rows else {}
    timestamp_s = _optional_int(first.get("timeStampData_s"))
    timestamp_u = _optional_int(first.get("timeStampData_u"))
    timestamp_us = None
    if timestamp_s is not None and timestamp_u is not None:
        timestamp_us = timestamp_s * 1_000_000 + timestamp_u
    return timestamp_s, timestamp_u, timestamp_us


def _build_frame(
    *,
    frame_id: str,
    arr: np.ndarray,
    timestamp_s: int | None,
    timestamp_u: int | None,
    background_light_intensity: float | None,
    meta: dict,
    timestamp_us: int | None = None,
) -> OrganizedLiDARFrame:
    if arr.shape != (H, W, C):
        raise ValueError(f"Organized frame must have shape {(H, W, C)}, got {arr.shape}")
    if timestamp_us is None and timestamp_s is not None and timestamp_u is not None:
        timestamp_us = timestamp_s * 1_000_000 + timestamp_u
    return OrganizedLiDARFrame(
        frame_id=frame_id,
        points_range=arr,
        points_flat=arr.reshape(-1, C),
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        timestamp_us=timestamp_us,
        background_light_intensity=background_light_intensity,
        meta=meta,
    )


def _to_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _optional_int(value: str | float | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))
