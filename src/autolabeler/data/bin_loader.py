import numpy as np
from .schemas import OrganizedLiDARFrame

H, W, C = 192, 480, 4


<<<<<<< codex/build-offline-auto-labeling-system-wl2yjl
def load_bin(path: str, frame_id: str, *, timestamp_s: int | None = None, timestamp_u: int | None = None,
             background_light_intensity: float | None = None, meta: dict | None = None) -> OrganizedLiDARFrame:
    arr = np.fromfile(path, dtype=np.float32).reshape(H, W, C)
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
        meta=meta or {},
    )
=======
def load_bin(path: str, frame_id: str) -> OrganizedLiDARFrame:
    arr = np.fromfile(path, dtype=np.float32).reshape(H, W, C)
    flat = arr.reshape(-1, C)
    return OrganizedLiDARFrame(frame_id=frame_id, points_range=arr, points_flat=flat)
>>>>>>> main
