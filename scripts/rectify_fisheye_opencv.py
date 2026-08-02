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


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    input_height, input_width = image.shape[:2]
    output_width, output_height = parse_size(args.output_size) if args.output_size else (input_width, input_height)
    circle = resolve_circle(
        cv2=cv2,
        image=image,
        fisheye_format=args.format,
        xcenter=args.xcenter,
        ycenter=args.ycenter,
        radius=args.radius,
        auto_circle=args.auto_circle,
        black_threshold=args.black_threshold,
    )

    remap = build_fisheye_remap(
        input_shape=(input_height, input_width),
        output_shape=(output_height, output_width),
        xcenter=circle.xcenter,
        ycenter=circle.ycenter,
        radius=circle.radius,
        fov=args.fov,
        pfov=args.pfov,
        dtype=args.dtype,
        projection=args.projection,
        pfov_axis=args.pfov_axis,
        angle=args.angle,
        yaw=args.yaw,
        pitch=args.pitch,
    )
    interpolation = interpolation_flag(cv2, args.interpolation)
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

    map_path = None
    if args.save_map:
        map_path = output_path.with_suffix(".remap.npz")
        np.savez_compressed(map_path, map_x=remap.map_x, map_y=remap.map_y, valid=remap.valid)

    metadata = {
        "version": 1,
        "input_image": str(image_path),
        "output_image": str(output_path),
        "input_size": [int(input_width), int(input_height)],
        "output_size": [int(output_width), int(output_height)],
        "output_aspect_ratio": float(output_width) / float(output_height),
        "format": args.format,
        "dtype": args.dtype,
        "projection": args.projection,
        "fov": float(args.fov),
        "pfov": float(args.pfov),
        "pfov_axis": args.pfov_axis,
        "effective_horizontal_fov": remap.horizontal_fov,
        "effective_vertical_fov": remap.vertical_fov,
        "circle": asdict(circle),
        "angle": float(args.angle),
        "yaw": float(args.yaw),
        "pitch": float(args.pitch),
        "interpolation": args.interpolation,
        "valid_output_pixels": int(np.count_nonzero(remap.valid)),
        "valid_output_fraction": float(np.mean(remap.valid)),
        "remap_file": None if map_path is None else str(map_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a circular fisheye image to a high-resolution flat image with OpenCV remap."
    )
    parser.add_argument("--image", required=True, help="Input fisheye image.")
    parser.add_argument("--output", required=True, help="Output JPG/PNG path.")
    parser.add_argument(
        "--output-size",
        default=None,
        help="Flat output size as WIDTHxHEIGHT, for example 3840x2160. Default: input size.",
    )
    parser.add_argument("--format", choices=("circular", "fullframe"), default="circular")
    parser.add_argument("--dtype", choices=FISHEYE_MODELS, default="linear")
    parser.add_argument("--projection", choices=OUTPUT_PROJECTIONS, default="perspective")
    parser.add_argument("--fov", type=float, default=180.0, help="Full angular diameter of the source fisheye circle.")
    parser.add_argument("--pfov", type=float, default=110.0, help="Field of view of the flat output along --pfov-axis.")
    parser.add_argument("--pfov-axis", choices=PFOV_AXES, default="horizontal")
    parser.add_argument("--xcenter", type=float, default=None)
    parser.add_argument("--ycenter", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--auto-circle", action="store_true", help="Estimate circle from the non-black image region.")
    parser.add_argument("--black-threshold", type=int, default=12)
    parser.add_argument("--angle", type=float, default=0.0, help="Rotate the source fisheye around its optical axis.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Move output view center horizontally, in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Move output view center vertically, in degrees; positive is down.")
    parser.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="lanczos")
    parser.add_argument("--save-map", action="store_true", help="Save map_x/map_y for reuse on frames from the same camera.")
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
