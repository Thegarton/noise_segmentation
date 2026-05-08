from __future__ import annotations

import csv
import numpy as np

from .bin_loader import H, W, C
from .schemas import OrganizedLiDARFrame


REQUIRED_COLUMNS = ("x", "y", "z", "intensity")


def load_csv(path: str, frame_id: str) -> OrganizedLiDARFrame:
    xyz_i = []
    timestamp_s = None
    timestamp_u = None
    background_light_intensity = None

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xyz_i.append([float(row["x"]), float(row["y"]), float(row["z"]), float(row["intensity"])])
            if timestamp_s is None and row.get("timestamp_s") not in (None, ""):
                timestamp_s = int(float(row["timestamp_s"]))
            if timestamp_u is None and row.get("timestamp_u") not in (None, ""):
                timestamp_u = int(float(row["timestamp_u"]))
            if background_light_intensity is None and row.get("background_light_intensity") not in (None, ""):
                background_light_intensity = float(row["background_light_intensity"])

    arr = np.asarray(xyz_i, dtype=np.float32)
    expected = H * W
    if arr.shape[0] != expected:
        raise ValueError(f"CSV {path} has {arr.shape[0]} points, expected {expected} for organized {H}x{W} LiDAR")

    arr = arr.reshape(H, W, C)
    flat = arr.reshape(-1, C)
    ts_us = None
    if timestamp_s is not None and timestamp_u is not None:
        ts_us = int(timestamp_s) * 1_000_000 + int(timestamp_u)

    return OrganizedLiDARFrame(
        frame_id=frame_id,
        points_range=arr,
        points_flat=flat,
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        timestamp_us=ts_us,
        background_light_intensity=background_light_intensity,
        meta={"source_format": "csv"},
    )
