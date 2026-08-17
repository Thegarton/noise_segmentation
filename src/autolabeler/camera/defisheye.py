#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autolabeler.camera.fisheye import (  # noqa: E402
    FISHEYE_MODELS,
    OUTPUT_PROJECTIONS,
    PFOV_AXES,
    build_fisheye_remap,
)


@dataclass(frozen=True)
class Circle:
    xcenter: float
    ycenter: float
    radius: float
    method: str


def defisheye(
    image: str,
    output: str,
    output_size: str = None,
    format: str = "circular",
    dtype: str = "linear",
    projection: str = "perspective",
    fov: float = 180.0,
    pfov: float = 110.0,
    pfov_axis: str = "horizontal",
    xcenter: float = None,
    ycenter: float = None,
    radius: float = None,
    auto_circle: bool = False,
    black_threshold: int = 12,
    angle: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    interpolation: str = "lanczos",
    save_map: bool = False,
):
    cv2 = import_cv2()
    image_path = Path(image).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    input_height, input_width = image.shape[:2]
    output_width, output_height = parse_size(output_size) if output_size else (input_width, input_height)
    circle = resolve_circle(
        cv2=cv2,
        image=image,
        fisheye_format=format,
        xcenter=xcenter,
        ycenter=ycenter,
        radius=radius,
        auto_circle=auto_circle,
        black_threshold=black_threshold,
    )

    remap = build_fisheye_remap(
        input_shape=(input_height, input_width),
        output_shape=(output_height, output_width),
        xcenter=circle.xcenter,
        ycenter=circle.ycenter,
        radius=circle.radius,
        fov=fov,
        pfov=pfov,
        dtype=dtype,
        projection=projection,
        pfov_axis=pfov_axis,
        angle=angle,
        yaw=yaw,
        pitch=pitch,
    )
    interpolation = interpolation_flag(cv2, interpolation)
    output = cv2.remap(
        image,
        remap.map_x,
        remap.map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise OSError(f"OpenCV could not write output image: {output_path}")



def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("This script requires OpenCV. Install with: pip install opencv-python") from exc
    return cv2


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.strip().lower().replace(",", "x").replace(" ", "x")
    tokens = [token for token in normalized.split("x") if token]
    if len(tokens) != 2:
        raise ValueError(f"Output size must be WIDTHxHEIGHT, got {value!r}")
    width, height = (int(token) for token in tokens)
    if width <= 0 or height <= 0:
        raise ValueError(f"Output size must be positive, got {value!r}")
    return width, height


def resolve_circle(
    *,
    cv2: Any,
    image: np.ndarray,
    fisheye_format: str,
    xcenter: float | None,
    ycenter: float | None,
    radius: float | None,
    auto_circle: bool,
    black_threshold: int,
) -> Circle:
    height, width = image.shape[:2]
    default_radius = min(width, height) / 2.0
    if fisheye_format == "fullframe":
        default_radius = float(np.hypot(width / 2.0, height / 2.0))
    default = Circle(
        xcenter=width / 2.0 if xcenter is None else float(xcenter),
        ycenter=height / 2.0 if ycenter is None else float(ycenter),
        radius=default_radius if radius is None else float(radius),
        method="manual_or_default",
    )
    if not auto_circle:
        return default

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > int(black_threshold)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return default
    contour = max(contours, key=cv2.contourArea)
    (detected_x, detected_y), detected_radius = cv2.minEnclosingCircle(contour)
    return Circle(
        xcenter=float(detected_x) if xcenter is None else float(xcenter),
        ycenter=float(detected_y) if ycenter is None else float(ycenter),
        radius=float(detected_radius) if radius is None else float(radius),
        method="auto_non_black_min_enclosing_circle",
    )


def interpolation_flag(cv2: Any, value: str) -> int:
    return {
        "linear": int(cv2.INTER_LINEAR),
        "cubic": int(cv2.INTER_CUBIC),
        "lanczos": int(cv2.INTER_LANCZOS4),
    }[value]
