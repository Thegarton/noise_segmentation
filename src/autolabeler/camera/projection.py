from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import CameraCalibration


INVALID_PIXEL = -1


def project_points_to_image(
    points_xyz: np.ndarray,
    calibration: CameraCalibration,
    *,
    image_shape: tuple[int, int] | None = None,
    use_z_buffer: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape [N,3], got {points.shape}")

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points, ones], axis=1)
    camera_points = (calibration.T_lidar_to_camera @ points_h.T).T[:, :3]
    depth = camera_points[:, 2].astype(np.float32)

    valid = np.isfinite(camera_points).all(axis=1) & (depth > 0.0)
    uvw = (calibration.K @ camera_points.T).T
    uv = np.full((points.shape[0], 2), INVALID_PIXEL, dtype=np.int32)
    projected_float = np.full((points.shape[0], 2), np.nan, dtype=np.float32)

    positive = valid & (np.abs(uvw[:, 2]) > 1e-9)
    projected_float[positive, 0] = (uvw[positive, 0] / uvw[positive, 2]).astype(np.float32)
    projected_float[positive, 1] = (uvw[positive, 1] / uvw[positive, 2]).astype(np.float32)
    rounded = np.zeros((points.shape[0], 2), dtype=np.int64)
    rounded[positive] = np.rint(projected_float[positive]).astype(np.int64, copy=False)

    if image_shape is not None:
        image_h, image_w = image_shape
        inside = (
            positive
            & (rounded[:, 0] >= 0)
            & (rounded[:, 0] < image_w)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < image_h)
        )
    else:
        inside = positive

    if use_z_buffer and image_shape is not None:
        visible = _z_buffer_visible(rounded, depth, inside, image_shape=image_shape)
    else:
        visible = inside

    uv[visible, 0] = rounded[visible, 0]
    uv[visible, 1] = rounded[visible, 1]
    return uv, depth, visible


def save_projection_arrays(
    output_dir: str | Path,
    *,
    point_to_pixel: np.ndarray,
    point_camera_depth: np.ndarray,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_path = out_dir / "point_to_pixel.npy"
    depth_path = out_dir / "point_camera_depth.npy"
    np.save(pixel_path, np.asarray(point_to_pixel, dtype=np.int32))
    np.save(depth_path, np.asarray(point_camera_depth, dtype=np.float32))
    return {"point_to_pixel": str(pixel_path), "point_camera_depth": str(depth_path)}


def save_projection_overlay(
    image_path: str | Path,
    output_path: str | Path,
    *,
    point_to_pixel: np.ndarray,
    depth: np.ndarray,
    point_radius: int = 1,
) -> None:
    cv2 = _import_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    overlay = image.copy()
    valid = np.asarray(point_to_pixel[:, 0] >= 0)
    if np.any(valid):
        valid_depth = np.asarray(depth[valid], dtype=np.float32)
        min_depth = float(np.nanmin(valid_depth))
        max_depth = float(np.nanmax(valid_depth))
        denom = max(max_depth - min_depth, 1e-6)
        colors = ((valid_depth - min_depth) / denom * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(colors.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        pixels = point_to_pixel[valid]
        for (u, v), color in zip(pixels, color_map):
            cv2.circle(overlay, (int(u), int(v)), point_radius, color.tolist(), thickness=-1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), overlay):
        raise ValueError(f"Could not write overlay: {out}")


def _z_buffer_visible(
    rounded: np.ndarray,
    depth: np.ndarray,
    inside: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    image_h, image_w = image_shape
    flat_pixel = rounded[:, 1] * image_w + rounded[:, 0]
    z = np.full(image_h * image_w, np.inf, dtype=np.float32)
    visible = np.zeros_like(inside, dtype=bool)
    for idx in np.where(inside)[0]:
        pixel = int(flat_pixel[idx])
        d = float(depth[idx])
        if d < z[pixel]:
            z[pixel] = d
    for idx in np.where(inside)[0]:
        pixel = int(flat_pixel[idx])
        visible[idx] = np.isclose(depth[idx], z[pixel])
    return visible


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Projection overlay requires OpenCV. Install opencv-python or use an env with cv2.") from exc
    return cv2
