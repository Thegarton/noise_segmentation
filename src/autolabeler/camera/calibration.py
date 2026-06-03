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
    source_image_shape: tuple[int, int] | None = None

    def K_for_image_shape(self, image_shape: tuple[int, int] | None) -> np.ndarray:
        if image_shape is None or self.source_image_shape is None:
            return self.K
        return scale_intrinsic_matrix(
            self.K,
            source_image_shape=self.source_image_shape,
            target_image_shape=image_shape,
        )


def load_camera_calibration(
    path: str | Path,
    *,
    extrinsic_direction: str,
    source_image_shape: tuple[int, int] | None = None,
) -> CameraCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    K = _find_matrix(payload, ("K", "intrinsic", "intrinsics", "camera_matrix"), shape=(3, 3))
    source_image_shape = source_image_shape or _find_source_image_shape(payload)

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

    return CameraCalibration(
        K=K.astype(np.float64),
        T_lidar_to_camera=T.astype(np.float64),
        source_path=str(path),
        source_image_shape=source_image_shape,
    )


def scale_intrinsic_matrix(
    K: np.ndarray,
    *,
    source_image_shape: tuple[int, int],
    target_image_shape: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_image_shape
    target_h, target_w = target_image_shape
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"source_image_shape must be positive, got {source_image_shape}")
    scaled = np.asarray(K, dtype=np.float64).copy()
    scaled[0, :] *= float(target_w) / float(source_w)
    scaled[1, :] *= float(target_h) / float(source_h)
    return scaled


def parse_image_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace(",", "x").replace("\t", "x").replace(" ", "x")
    tokens = [token for token in normalized.split("x") if token]
    if len(tokens) != 2:
        raise ValueError(f"Image size must be WIDTHxHEIGHT, got {value!r}")
    width, height = (int(float(token)) for token in tokens)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {value!r}")
    return height, width


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


def _find_source_image_shape(payload: dict[str, Any]) -> tuple[int, int] | None:
    image_shape = _find_key_recursive(payload, "image_shape")
    if image_shape is not None:
        return _as_image_shape(image_shape, value_order="height_width")

    for key in ("calibration_image_size", "source_image_size", "image_size", "resolution"):
        image_size = _find_key_recursive(payload, key)
        if image_size is not None:
            return _as_image_shape(image_size, value_order="width_height")
    return None


def _as_image_shape(value: Any, *, value_order: str) -> tuple[int, int]:
    if isinstance(value, dict):
        width = value.get("width", value.get("w"))
        height = value.get("height", value.get("h"))
        if width is None or height is None:
            raise ValueError(f"Image size dict must contain width/height, got {value}")
        width_i = int(width)
        height_i = int(height)
    else:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"Image size must have two values, got shape {arr.shape}")
        first, second = int(arr[0]), int(arr[1])
        if value_order == "height_width":
            height_i, width_i = first, second
        elif value_order == "width_height":
            width_i, height_i = first, second
        else:
            raise ValueError(f"Unsupported image size order: {value_order}")

    if width_i <= 0 or height_i <= 0:
        raise ValueError(f"Image size must be positive, got width={width_i} height={height_i}")
    return height_i, width_i


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
