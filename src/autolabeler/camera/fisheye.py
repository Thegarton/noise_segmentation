"""OpenCV-compatible fisheye-to-flat remapping geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FISHEYE_MODELS = ("linear", "equalarea", "orthographic", "stereographic")
OUTPUT_PROJECTIONS = ("perspective", "cylindrical")
PFOV_AXES = ("horizontal", "vertical", "diagonal")


@dataclass(frozen=True)
class FisheyeRemap:
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    horizontal_fov: float
    vertical_fov: float


def build_fisheye_remap(
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    xcenter: float,
    ycenter: float,
    radius: float,
    fov: float = 180.0,
    pfov: float = 110.0,
    dtype: str = "linear",
    projection: str = "perspective",
    pfov_axis: str = "horizontal",
    angle: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> FisheyeRemap:
    """Build inverse maps from a flat output image into a circular fisheye.

    ``fov`` is the full angular diameter represented by the source circle.
    ``pfov`` is the output perspective FOV along ``pfov_axis``.
    """

    input_height, input_width = _validate_shape(input_shape, name="input_shape")
    output_height, output_width = _validate_shape(output_shape, name="output_shape")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if not 0.0 < fov <= 360.0:
        raise ValueError(f"fov must be in (0, 360], got {fov}")
    if not 0.0 < pfov < 180.0:
        raise ValueError(f"pfov must be in (0, 180), got {pfov}")
    if dtype not in FISHEYE_MODELS:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {FISHEYE_MODELS}")
    if projection not in OUTPUT_PROJECTIONS:
        raise ValueError(f"Unsupported projection {projection!r}; expected one of {OUTPUT_PROJECTIONS}")
    if pfov_axis not in PFOV_AXES:
        raise ValueError(f"Unsupported pfov_axis {pfov_axis!r}; expected one of {PFOV_AXES}")

    horizontal_fov, vertical_fov, focal = resolve_output_fov(
        output_shape=(output_height, output_width),
        pfov=pfov,
        pfov_axis=pfov_axis,
    )
    grid_x = np.arange(output_width, dtype=np.float32)[None, :]
    grid_y = np.arange(output_height, dtype=np.float32)[:, None]
    center_x = (float(output_width) - 1.0) / 2.0
    center_y = (float(output_height) - 1.0) / 2.0

    if projection == "perspective":
        ray_x = (grid_x - center_x) / focal
        ray_y = (grid_y - center_y) / focal
        ray_z: np.ndarray | float = 1.0
    else:
        longitude = (grid_x - center_x) * np.deg2rad(horizontal_fov) / max(float(output_width - 1), 1.0)
        latitude = (grid_y - center_y) * np.deg2rad(vertical_fov) / max(float(output_height - 1), 1.0)
        ray_x = np.sin(longitude)
        ray_y = np.tan(latitude)
        ray_z = np.cos(longitude)

    ray_x, ray_y, ray_z = _rotate_ray_components(ray_x, ray_y, ray_z, yaw=yaw, pitch=pitch)
    ray_xy = np.hypot(ray_x, ray_y)
    theta = np.arctan2(ray_xy, ray_z)
    phi = np.arctan2(ray_y, ray_x) + np.deg2rad(float(angle))

    theta_max = np.deg2rad(float(fov)) / 2.0
    radial_fraction = fisheye_radius_fraction(theta, theta_max=theta_max, dtype=dtype)
    source_radius = radial_fraction * float(radius)
    map_x = float(xcenter) + source_radius * np.cos(phi)
    map_y = float(ycenter) + source_radius * np.sin(phi)

    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (theta <= theta_max + 1e-12)
        & (map_x >= 0.0)
        & (map_x <= float(input_width - 1))
        & (map_y >= 0.0)
        & (map_y <= float(input_height - 1))
    )
    map_x = np.where(valid, map_x, -1.0).astype(np.float32)
    map_y = np.where(valid, map_y, -1.0).astype(np.float32)
    return FisheyeRemap(
        map_x=map_x,
        map_y=map_y,
        valid=valid,
        horizontal_fov=float(horizontal_fov),
        vertical_fov=float(vertical_fov),
    )


def resolve_output_fov(
    *,
    output_shape: tuple[int, int],
    pfov: float,
    pfov_axis: str,
) -> tuple[float, float, float]:
    """Resolve horizontal/vertical FOV and a square-pixel focal length."""

    height, width = _validate_shape(output_shape, name="output_shape")
    half_width = max((float(width) - 1.0) / 2.0, 0.5)
    half_height = max((float(height) - 1.0) / 2.0, 0.5)
    half_angle = np.deg2rad(float(pfov)) / 2.0
    if pfov_axis == "horizontal":
        focal = half_width / np.tan(half_angle)
    elif pfov_axis == "vertical":
        focal = half_height / np.tan(half_angle)
    elif pfov_axis == "diagonal":
        focal = np.hypot(half_width, half_height) / np.tan(half_angle)
    else:
        raise ValueError(f"Unsupported pfov_axis {pfov_axis!r}; expected one of {PFOV_AXES}")

    horizontal_fov = np.rad2deg(2.0 * np.arctan(half_width / focal))
    vertical_fov = np.rad2deg(2.0 * np.arctan(half_height / focal))
    return float(horizontal_fov), float(vertical_fov), float(focal)


def fisheye_radius_fraction(theta: np.ndarray, *, theta_max: float, dtype: str) -> np.ndarray:
    """Map incident ray angle to normalized radius for common fisheye models."""

    if theta_max <= 0.0:
        raise ValueError(f"theta_max must be positive, got {theta_max}")
    theta_arr = np.asarray(theta, dtype=np.float64)
    if dtype == "linear":
        return theta_arr / theta_max
    if dtype == "equalarea":
        denominator = np.sin(theta_max / 2.0)
        return np.sin(theta_arr / 2.0) / denominator
    if dtype == "orthographic":
        denominator = np.sin(theta_max)
        if abs(denominator) < 1e-12:
            raise ValueError("orthographic mapping is singular for this input fov")
        return np.sin(theta_arr) / denominator
    if dtype == "stereographic":
        denominator = np.tan(theta_max / 2.0)
        return np.tan(theta_arr / 2.0) / denominator
    raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {FISHEYE_MODELS}")


def _rotate_ray_components(
    ray_x: np.ndarray,
    ray_y: np.ndarray,
    ray_z: np.ndarray | float,
    *,
    yaw: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw_rad = np.deg2rad(float(yaw))
    pitch_rad = np.deg2rad(float(pitch))
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)

    pitched_y = cos_pitch * ray_y + sin_pitch * ray_z
    pitched_z = -sin_pitch * ray_y + cos_pitch * ray_z
    rotated_x = cos_yaw * ray_x + sin_yaw * pitched_z
    rotated_z = -sin_yaw * ray_x + cos_yaw * pitched_z
    output_shape = np.broadcast_shapes(rotated_x.shape, pitched_y.shape, rotated_z.shape)
    return (
        np.broadcast_to(rotated_x, output_shape),
        np.broadcast_to(pitched_y, output_shape),
        np.broadcast_to(rotated_z, output_shape),
    )


def _validate_shape(shape: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must contain height and width, got {shape}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {shape}")
    return height, width
