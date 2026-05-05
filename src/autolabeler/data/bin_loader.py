import numpy as np
from .schemas import OrganizedLiDARFrame

H, W, C = 192, 480, 4


def load_bin(path: str, frame_id: str) -> OrganizedLiDARFrame:
    arr = np.fromfile(path, dtype=np.float32).reshape(H, W, C)
    flat = arr.reshape(-1, C)
    return OrganizedLiDARFrame(frame_id=frame_id, points_range=arr, points_flat=flat)
