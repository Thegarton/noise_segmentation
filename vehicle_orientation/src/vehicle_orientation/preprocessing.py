"""Shared fisheye and mask-crop preprocessing for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FisheyeConfig:
    output_width: int = 3840
    output_height: int = 3060
    fisheye_format: str = "circular"
    dtype: str = "equalarea"
    projection: str = "cylindrical"
    fov: float = 190.0
    pfov: float = 140.0
    pfov_axis: str = "horizontal"
    xcenter: float = 960.0
    ycenter: float = 750.0
    radius: float = 1068.0
    interpolation: str = "lanczos"
    mask_radius: float = 860.0
    mask_lower_radius: float | None = None
    mask_center_y_offset: float = -125.0
    color_correction: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_FISHEYE_CONFIG = FisheyeConfig()


@dataclass(frozen=True)
class CropResult:
    bbox_xyxy: tuple[int, int, int, int]
    rgb: np.ndarray
    mask: np.ndarray
    masked_rgb: np.ndarray


@dataclass(frozen=True)
class FisheyeRemap:
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    horizontal_fov: float
    vertical_fov: float


class FisheyePreprocessor:
    """Prepare HL320 camera images while reusing expensive OpenCV remaps."""

    def __init__(self, config: FisheyeConfig = DEFAULT_FISHEYE_CONFIG) -> None:
        self.config = config
        self._remap_cache: dict[tuple[int, int, FisheyeConfig], FisheyeRemap] = {}

    @property
    def cache_size(self) -> int:
        return len(self._remap_cache)

    def prepare_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        cv2 = _import_cv2()
        image = _validate_rgb_like(image_bgr, name="image_bgr")
        height, width = image.shape[:2]
        corrected = simple_colour_correction(image) if self.config.color_correction else image.copy()
        masked = apply_circular_mask(
            corrected,
            center=(width / 2.0, height / 2.0 + self.config.mask_center_y_offset),
            radius=self.config.mask_radius,
            lower_radius=self.config.mask_lower_radius,
        )
        key = (height, width, self.config)
        remap = self._remap_cache.get(key)
        if remap is None:
            remap = build_fisheye_remap(
                input_shape=(height, width),
                output_shape=(self.config.output_height, self.config.output_width),
                xcenter=self.config.xcenter,
                ycenter=self.config.ycenter,
                radius=self.config.radius,
                fov=self.config.fov,
                pfov=self.config.pfov,
                dtype=self.config.dtype,
                projection=self.config.projection,
                pfov_axis=self.config.pfov_axis,
            )
            self._remap_cache[key] = remap
        output = cv2.remap(
            masked,
            remap.map_x,
            remap.map_y,
            interpolation=_interpolation_flag(cv2, self.config.interpolation),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return np.asarray(output, dtype=np.uint8)


def simple_colour_correction(image: np.ndarray, *, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Apply a conservative per-channel percentile stretch without annotations."""

    arr = _validate_rgb_like(image, name="image").astype(np.float32)
    output = np.empty_like(arr)
    for channel in range(3):
        plane = arr[..., channel]
        low = float(np.percentile(plane, low_percentile))
        high = float(np.percentile(plane, high_percentile))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
            output[..., channel] = plane
            continue
        output[..., channel] = (plane - low) * (255.0 / (high - low))
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def apply_circular_mask(
    image: np.ndarray,
    *,
    center: tuple[float, float],
    radius: float,
    lower_radius: float | None = None,
) -> np.ndarray:
    arr = _validate_rgb_like(image, name="image")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    height, width = arr.shape[:2]
    inside = build_split_circular_mask(
        shape=(height, width),
        center=center,
        upper_radius=radius,
        lower_radius=lower_radius,
    )
    output = np.zeros_like(arr)
    output[inside] = arr[inside]
    return output


def build_split_circular_mask(
    *,
    shape: tuple[int, int],
    center: tuple[float, float],
    upper_radius: float,
    lower_radius: float | None = None,
) -> np.ndarray:
    """Build a mask whose upper and lower image halves use different radii."""
    height, width = _validate_shape(shape, name="shape")
    upper = float(upper_radius)
    lower = upper if lower_radius is None else float(lower_radius)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError(f"upper_radius must be positive and finite, got {upper_radius}")
    if not np.isfinite(lower) or lower <= 0.0:
        raise ValueError(f"lower_radius must be positive and finite, got {lower_radius}")

    center_x, center_y = (float(value) for value in center)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError(f"center must be finite, got {center}")
    yy, xx = np.ogrid[:height, :width]
    distance_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    upper_half = yy <= center_y
    return np.where(
        upper_half,
        distance_squared <= upper**2,
        distance_squared <= lower**2,
    )


def extract_mask_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    padding: float = 0.12,
    background: int | tuple[int, int, int] = 127,
) -> CropResult:
    image = _validate_rgb_like(image_rgb, name="image_rgb")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != image.shape[:2]:
        raise ValueError(f"mask shape {binary.shape} does not match image shape {image.shape[:2]}")
    if not 0.0 <= float(padding) <= 1.0:
        raise ValueError(f"padding must be in [0,1], got {padding}")
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("Cannot crop an empty mask")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = int(np.ceil((x1 - x0) * float(padding)))
    pad_y = int(np.ceil((y1 - y0) * float(padding)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(image.shape[1], x1 + pad_x)
    y1 = min(image.shape[0], y1 + pad_y)

    rgb = image[y0:y1, x0:x1].copy()
    crop_mask = binary[y0:y1, x0:x1].copy()
    fill = np.asarray(background, dtype=np.uint8)
    if fill.ndim == 0:
        fill = np.repeat(fill, 3)
    if fill.shape != (3,):
        raise ValueError(f"background must be a scalar or RGB triplet, got shape {fill.shape}")
    masked = np.broadcast_to(fill, rgb.shape).copy()
    masked[crop_mask] = rgb[crop_mask]
    return CropResult(
        bbox_xyxy=(x0, y0, x1, y1),
        rgb=rgb,
        mask=crop_mask,
        masked_rgb=masked,
    )


def deduplicate_mask_indices(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    *,
    iou_threshold: float = 0.8,
) -> list[int]:
    if len(masks) != len(scores):
        raise ValueError(f"masks/scores length mismatch: {len(masks)} vs {len(scores)}")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"iou_threshold must be in [0,1], got {iou_threshold}")
    order = sorted(range(len(masks)), key=lambda index: (-float(scores[index]), index))
    keep: list[int] = []
    normalized = [np.asarray(mask, dtype=bool) for mask in masks]
    for index in order:
        candidate = normalized[index]
        if any(mask_iou(candidate, normalized[kept]) > iou_threshold for kept in keep):
            continue
        keep.append(index)
    return keep


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape:
        raise ValueError(f"Mask shape mismatch: {left_mask.shape} vs {right_mask.shape}")
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(left_mask & right_mask))
    return float(intersection) / float(union)


def letterbox_rgb(image_rgb: np.ndarray, *, size: int = 224, fill: int = 127) -> np.ndarray:
    image = _validate_rgb_like(image_rgb, name="image_rgb")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    from PIL import Image, ImageOps  # noqa: WPS433

    pil = Image.fromarray(image, mode="RGB")
    contained = ImageOps.contain(pil, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(fill, fill, fill))
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def classifier_tensor_array(image_rgb: np.ndarray, *, size: int = 224) -> np.ndarray:
    image = letterbox_rgb(image_rgb, size=size).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    normalized = (image - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def build_fisheye_remap(
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    xcenter: float,
    ycenter: float,
    radius: float,
    fov: float,
    pfov: float,
    dtype: str,
    projection: str,
    pfov_axis: str,
) -> FisheyeRemap:
    input_height, input_width = _validate_shape(input_shape, name="input_shape")
    output_height, output_width = _validate_shape(output_shape, name="output_shape")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if not 0.0 < fov <= 360.0:
        raise ValueError(f"fov must be in (0,360], got {fov}")
    if not 0.0 < pfov < 180.0:
        raise ValueError(f"pfov must be in (0,180), got {pfov}")
    if dtype not in {"linear", "equalarea", "orthographic", "stereographic"}:
        raise ValueError(f"Unsupported fisheye dtype: {dtype}")
    if projection not in {"perspective", "cylindrical"}:
        raise ValueError(f"Unsupported output projection: {projection}")

    horizontal_fov, vertical_fov, focal = _resolve_output_fov(
        output_shape=(output_height, output_width),
        pfov=pfov,
        axis=pfov_axis,
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

    ray_xy = np.hypot(ray_x, ray_y)
    theta = np.arctan2(ray_xy, ray_z)
    phi = np.arctan2(ray_y, ray_x)
    theta_max = np.deg2rad(float(fov)) / 2.0
    radial_fraction = _fisheye_radius_fraction(theta, theta_max=theta_max, dtype=dtype)
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
    return FisheyeRemap(
        map_x=np.where(valid, map_x, -1.0).astype(np.float32),
        map_y=np.where(valid, map_y, -1.0).astype(np.float32),
        valid=valid,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
    )


def _resolve_output_fov(*, output_shape: tuple[int, int], pfov: float, axis: str) -> tuple[float, float, float]:
    height, width = _validate_shape(output_shape, name="output_shape")
    half_width = max((float(width) - 1.0) / 2.0, 0.5)
    half_height = max((float(height) - 1.0) / 2.0, 0.5)
    half_angle = np.deg2rad(float(pfov)) / 2.0
    if axis == "horizontal":
        focal = half_width / np.tan(half_angle)
    elif axis == "vertical":
        focal = half_height / np.tan(half_angle)
    elif axis == "diagonal":
        focal = np.hypot(half_width, half_height) / np.tan(half_angle)
    else:
        raise ValueError(f"Unsupported pfov axis: {axis}")
    horizontal = np.rad2deg(2.0 * np.arctan(half_width / focal))
    vertical = np.rad2deg(2.0 * np.arctan(half_height / focal))
    return float(horizontal), float(vertical), float(focal)


def _fisheye_radius_fraction(theta: np.ndarray, *, theta_max: float, dtype: str) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    if dtype == "linear":
        return values / theta_max
    if dtype == "equalarea":
        return np.sin(values / 2.0) / np.sin(theta_max / 2.0)
    if dtype == "orthographic":
        denominator = np.sin(theta_max)
        if abs(denominator) < 1e-12:
            raise ValueError("orthographic mapping is singular for this fov")
        return np.sin(values) / denominator
    if dtype == "stereographic":
        return np.tan(values / 2.0) / np.tan(theta_max / 2.0)
    raise ValueError(f"Unsupported fisheye dtype: {dtype}")


def _validate_shape(shape: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must contain height and width, got {shape}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {shape}")
    return height, width


def _validate_rgb_like(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H,W,3], got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Vehicle fisheye preprocessing requires opencv-python") from exc
    return cv2


def _interpolation_flag(cv2: Any, value: str) -> int:
    flags = {
        "linear": int(cv2.INTER_LINEAR),
        "cubic": int(cv2.INTER_CUBIC),
        "lanczos": int(cv2.INTER_LANCZOS4),
    }
    if value not in flags:
        raise ValueError(f"Unsupported interpolation: {value}")
    return flags[value]
