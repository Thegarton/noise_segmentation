"""In-memory colour correction matching the main HL320 camera pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


def fix_cast_gray_world(image_bgr: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    blue, green, red = cv2.split(np.asarray(image_bgr, dtype=np.float32))
    mean_blue, mean_green, mean_red = (
        float(np.mean(blue)),
        float(np.mean(green)),
        float(np.mean(red)),
    )
    mean_gray = (mean_blue + mean_green + mean_red) / 3.0
    blue = np.clip(blue * (mean_gray / (mean_blue + 1e-5)), 0, 255)
    green = np.clip(green * (mean_gray / (mean_green + 1e-5)), 0, 255)
    red = np.clip(red * (mean_gray / (mean_red + 1e-5)), 0, 255)
    return cv2.merge([blue, green, red]).astype(np.uint8)


def simple_wb(image_bgr: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    xphoto = getattr(cv2, "xphoto", None)
    if xphoto is None or not hasattr(xphoto, "createSimpleWB"):
        raise RuntimeError(
            "--colour-correction requires cv2.xphoto; install opencv-contrib-python"
        )
    white_balance = xphoto.createSimpleWB()
    white_balance.setP(0.5)
    return white_balance.balanceWhite(image_bgr)


def simple_colour_correction(image_bgr: np.ndarray) -> np.ndarray:
    return fix_cast_gray_world(simple_wb(image_bgr))


def simple_colour_correction_rgb(image_rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(image_rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {array.shape}")
    image_bgr = np.ascontiguousarray(array[..., ::-1])
    corrected_bgr = simple_colour_correction(image_bgr)
    return np.ascontiguousarray(corrected_bgr[..., ::-1])


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Colour correction requires opencv-contrib-python"
        ) from exc
    return cv2
