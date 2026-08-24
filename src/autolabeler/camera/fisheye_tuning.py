"""Geometry helpers for interactive calibration-free fisheye rectification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fisheye import FISHEYE_MODELS, fisheye_radius_fraction

Projection = Literal["perspective", "cylindrical"]


@dataclass(frozen=True)
class TunableFisheyeRemap:
    """Inverse OpenCV remap and diagnostic information."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    projection: Projection
    horizontal_fov: float
    vertical_fov: float


def build_tunable_fisheye_remap(
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    source_cx: float,
    source_cy: float,
    source_radius_x: float,
    source_radius_y: float,
    source_fov: float,
    fisheye_model: str = "equalarea",
    projection: Projection = "cylindrical",
    horizontal_fov: float = 180.0,
    vertical_fov: float = 100.0,
    output_cx: float = 0.5,
    output_cy: float = 0.5,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
) -> TunableFisheyeRemap:
    """Build an inverse map from a virtual camera into a fisheye image.

    Coordinates use the camera convention X-right, Y-down, Z-forward. The
    yaw/pitch/roll rotation maps rays from the virtual output camera into the
    physical fisheye camera. ``output_cx`` and ``output_cy`` are normalized
    principal-point positions in the output image.

    ``k1`` through ``k4`` are optional normalized radial corrections applied
    after the selected fisheye model::

        r_corrected = r * (1 + k1*r^2 + k2*r^4 + k3*r^6 + k4*r^8)
    """

    input_height, input_width = _validate_shape(input_shape, name="input_shape")
    output_height, output_width = _validate_shape(output_shape, name="output_shape")
    _validate_parameters(
        source_radius_x=source_radius_x,
        source_radius_y=source_radius_y,
        source_fov=source_fov,
        fisheye_model=fisheye_model,
        projection=projection,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        output_cx=output_cx,
        output_cy=output_cy,
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(output_width, dtype=np.float64),
        np.arange(output_height, dtype=np.float64),
    )
    principal_x = float(output_cx) * float(output_width - 1)
    principal_y = float(output_cy) * float(output_height - 1)
    ray_x, ray_y, ray_z = build_output_rays(
        grid_x=grid_x,
        grid_y=grid_y,
        output_shape=(output_height, output_width),
        projection=projection,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        principal_x=principal_x,
        principal_y=principal_y,
    )
    ray_x, ray_y, ray_z = rotate_rays(
        ray_x,
        ray_y,
        ray_z,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )

    theta = np.arctan2(np.hypot(ray_x, ray_y), ray_z)
    phi = np.arctan2(ray_y, ray_x)
    theta_max = np.deg2rad(float(source_fov)) / 2.0
    radius_fraction = fisheye_radius_fraction(
        theta,
        theta_max=theta_max,
        dtype=fisheye_model,
    )
    radius_squared = np.square(radius_fraction)
    radial_scale = (
        1.0
        + float(k1) * radius_squared
        + float(k2) * np.square(radius_squared)
        + float(k3) * np.power(radius_squared, 3)
        + float(k4) * np.power(radius_squared, 4)
    )
    corrected_radius = radius_fraction * radial_scale
    map_x = float(source_cx) + float(source_radius_x) * corrected_radius * np.cos(phi)
    map_y = float(source_cy) + float(source_radius_y) * corrected_radius * np.sin(phi)

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(corrected_radius)
        & (theta <= theta_max + 1e-12)
        & (corrected_radius >= 0.0)
        & (corrected_radius <= 1.0)
        & (map_x >= 0.0)
        & (map_x <= float(input_width - 1))
        & (map_y >= 0.0)
        & (map_y <= float(input_height - 1))
    )
    return TunableFisheyeRemap(
        map_x=np.where(valid, map_x, -1.0).astype(np.float32),
        map_y=np.where(valid, map_y, -1.0).astype(np.float32),
        valid=valid,
        projection=projection,
        horizontal_fov=float(horizontal_fov),
        vertical_fov=float(vertical_fov),
    )


def build_output_rays(
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_shape: tuple[int, int],
    projection: Projection,
    horizontal_fov: float,
    vertical_fov: float,
    principal_x: float,
    principal_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct virtual-camera rays for perspective or cylindrical pixels."""

    output_height, output_width = _validate_shape(output_shape, name="output_shape")
    horizontal_radians = np.deg2rad(float(horizontal_fov))
    vertical_radians = np.deg2rad(float(vertical_fov))
    if projection == "perspective":
        focal_x = max(float(output_width - 1), 1.0) / (2.0 * np.tan(horizontal_radians / 2.0))
        focal_y = max(float(output_height - 1), 1.0) / (2.0 * np.tan(vertical_radians / 2.0))
        ray_x = (grid_x - float(principal_x)) / focal_x
        ray_y = (grid_y - float(principal_y)) / focal_y
        ray_z = np.ones_like(ray_x)
    elif projection == "cylindrical":
        focal_phi = max(float(output_width - 1), 1.0) / horizontal_radians
        focal_y = max(float(output_height - 1), 1.0) / (2.0 * np.tan(vertical_radians / 2.0))
        azimuth = (grid_x - float(principal_x)) / focal_phi
        ray_x = np.sin(azimuth)
        ray_y = (grid_y - float(principal_y)) / focal_y
        ray_z = np.cos(azimuth)
    else:
        raise ValueError(f"Unsupported projection {projection!r}; expected perspective or cylindrical")
    return ray_x, ray_y, ray_z


def rotate_rays(
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    ray_z: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    roll: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate virtual rays into the source camera using yaw, pitch, then roll."""

    rotation = rotation_matrix(yaw=yaw, pitch=pitch, roll=roll)
    rays = np.stack(np.broadcast_arrays(ray_x, ray_y, ray_z), axis=0)
    rotated = np.einsum("ij,j...->i...", rotation, rays)
    return rotated[0], rotated[1], rotated[2]


def rotation_matrix(*, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a yaw-then-pitch-then-roll matrix for RDF camera coordinates."""

    yaw_rad, pitch_rad, roll_rad = np.deg2rad([yaw, pitch, roll])
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
    yaw_matrix = np.asarray(
        [[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]],
        dtype=np.float64,
    )
    pitch_matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cos_pitch, sin_pitch], [0.0, -sin_pitch, cos_pitch]],
        dtype=np.float64,
    )
    roll_matrix = np.asarray(
        [[cos_roll, -sin_roll, 0.0], [sin_roll, cos_roll, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return roll_matrix @ pitch_matrix @ yaw_matrix


def remap_image(
    image: np.ndarray,
    remap: TunableFisheyeRemap,
    *,
    interpolation: str = "lanczos",
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply a tuning remap with OpenCV, imported only when rendering is used."""

    import cv2  # type: ignore  # noqa: WPS433

    interpolation_flags = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation not in interpolation_flags:
        raise ValueError(
            f"Unsupported interpolation {interpolation!r}; expected one of {tuple(interpolation_flags)}"
        )
    return cv2.remap(
        np.asarray(image),
        remap.map_x,
        remap.map_y,
        interpolation_flags[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def crop_and_stitch_three_views(
    views: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    overlap_pixels: int,
) -> np.ndarray:
    """Remove duplicated side regions and concatenate three rectifications."""

    arrays = tuple(np.asarray(view) for view in views)
    if len({view.shape for view in arrays}) != 1:
        raise ValueError(f"All views must have the same shape, got {[view.shape for view in arrays]}")
    if arrays[0].ndim not in (2, 3):
        raise ValueError(f"Views must be images with 2 or 3 dimensions, got {arrays[0].shape}")
    width = int(arrays[0].shape[1])
    overlap = int(overlap_pixels)
    if overlap < 0 or overlap >= width:
        raise ValueError(f"overlap_pixels must be in [0, {width}), got {overlap_pixels}")

    left_trim = overlap // 2
    right_trim = overlap - left_trim
    left = arrays[0][:, : width - right_trim] if right_trim else arrays[0]
    center = arrays[1][:, left_trim : width - right_trim if right_trim else width]
    right = arrays[2][:, left_trim:]
    return np.concatenate((left, center, right), axis=1)


def _validate_parameters(
    *,
    source_radius_x: float,
    source_radius_y: float,
    source_fov: float,
    fisheye_model: str,
    projection: str,
    horizontal_fov: float,
    vertical_fov: float,
    output_cx: float,
    output_cy: float,
) -> None:
    if source_radius_x <= 0.0 or source_radius_y <= 0.0:
        raise ValueError("source_radius_x and source_radius_y must be positive")
    if not 0.0 < source_fov <= 360.0:
        raise ValueError(f"source_fov must be in (0, 360], got {source_fov}")
    if fisheye_model not in FISHEYE_MODELS:
        raise ValueError(f"Unsupported fisheye_model {fisheye_model!r}; expected one of {FISHEYE_MODELS}")
    if projection not in ("perspective", "cylindrical"):
        raise ValueError(f"Unsupported projection {projection!r}")
    if not 0.0 < horizontal_fov < 360.0:
        raise ValueError(f"horizontal_fov must be in (0, 360), got {horizontal_fov}")
    if projection == "perspective" and horizontal_fov >= 180.0:
        raise ValueError("Perspective horizontal_fov must be below 180 degrees")
    if not 0.0 < vertical_fov < 180.0:
        raise ValueError(f"vertical_fov must be in (0, 180), got {vertical_fov}")
    if not 0.0 <= output_cx <= 1.0 or not 0.0 <= output_cy <= 1.0:
        raise ValueError("output_cx and output_cy must be normalized values in [0, 1]")


def _validate_shape(shape: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must contain height and width, got {shape}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {shape}")
    return height, width
