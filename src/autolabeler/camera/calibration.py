from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    T_lidar_to_camera: np.ndarray
    source_path: str


def load_camera_calibration(path: str | Path, *, extrinsic_direction: str) -> CameraCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    K = _find_matrix(payload, ("K", "intrinsic", "intrinsics", "camera_matrix"), shape=(3, 3))

    if extrinsic_direction == "lidar_to_camera":
        T = _find_matrix(
            payload,
            ("T_lidar_to_camera", "lidar_to_camera", "extrinsic", "extrinsics", "T"),
            shape=(4, 4),
            allow_3x4=True,
        )
    elif extrinsic_direction == "camera_to_lidar":
        T_camera_to_lidar = _find_matrix(
            payload,
            ("T_camera_to_lidar", "camera_to_lidar", "extrinsic", "extrinsics", "T"),
            shape=(4, 4),
            allow_3x4=True,
        )
        T = np.linalg.inv(T_camera_to_lidar)
    else:
        raise ValueError(f"Unsupported extrinsic direction: {extrinsic_direction}")

    return CameraCalibration(K=K.astype(np.float64), T_lidar_to_camera=T.astype(np.float64), source_path=str(path))


def _find_matrix(payload: dict[str, Any], keys: tuple[str, ...], *, shape: tuple[int, int], allow_3x4: bool = False) -> np.ndarray:
    for key in keys:
        value = _find_key_recursive(payload, key)
        if value is None:
            continue
        matrix = _as_matrix(value)
        if matrix.shape == shape:
            return matrix
        if allow_3x4 and matrix.shape == (3, 4) and shape == (4, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = matrix
            return out
        raise ValueError(f"Matrix {key} has shape {matrix.shape}, expected {shape}")
    raise KeyError(f"Could not find matrix with any key: {', '.join(keys)}")


def _find_key_recursive(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for nested in value.values():
            found = _find_key_recursive(nested, key)
            if found is not None:
                return found
    return None


def _as_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == 9:
            return arr.reshape(3, 3)
        if arr.size == 12:
            return arr.reshape(3, 4)
        if arr.size == 16:
            return arr.reshape(4, 4)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Could not parse matrix from value with shape {arr.shape}")

