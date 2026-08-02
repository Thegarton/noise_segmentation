#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FOVS = (160.0, 170.0, 180.0)
DEFAULT_PFOVS = (70.0, 85.0, 100.0, 115.0, 130.0)
DEFAULT_DTYPES = ("linear", "equalarea", "stereographic")


@dataclass(frozen=True)
class CircleEstimate:
    xcenter: float
    ycenter: float
    radius: float
    method: str


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2 = import_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image.shape[:2]

    circle = resolve_circle(
        image=image,
        xcenter=args.xcenter,
        ycenter=args.ycenter,
        radius=args.radius,
        auto_circle=args.auto_circle,
        black_threshold=args.black_threshold,
    )
    fovs = parse_float_list(args.fovs, default=DEFAULT_FOVS)
    pfovs = parse_float_list(args.pfovs, default=DEFAULT_PFOVS)
    dtypes = parse_str_list(args.dtypes, default=DEFAULT_DTYPES)

    variants = []
    for dtype in dtypes:
        for fov in fovs:
            for pfov in pfovs:
                variant = run_defisheye_variant(
                    image_path=image_path,
                    out_dir=out_dir,
                    dtype=dtype,
                    fisheye_format=args.format,
                    fov=fov,
                    pfov=pfov,
                    circle=circle,
                    angle=args.angle,
                    pad=args.pad,
                )
                variants.append(variant)

    contact_sheet_path = out_dir / "defisheye_grid.jpg"
    build_contact_sheet(
        cv2=cv2,
        variants=variants,
        output_path=contact_sheet_path,
        tile_width=args.tile_width,
        label_height=args.label_height,
        columns=args.columns,
    )
    manifest = {
        "version": 1,
        "input_image": str(image_path),
        "image_size": [int(width), int(height)],
        "format": args.format,
        "auto_circle": bool(args.auto_circle),
        "circle": {
            "xcenter": float(circle.xcenter),
            "ycenter": float(circle.ycenter),
            "radius": float(circle.radius),
            "method": circle.method,
        },
        "angle": float(args.angle),
        "pad": int(args.pad),
        "fovs": fovs,
        "pfovs": pfovs,
        "dtypes": dtypes,
        "variants": variants,
        "contact_sheet": str(contact_sheet_path),
    }
    manifest_path = out_dir / "defisheye_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"variants": len(variants), "contact_sheet": str(contact_sheet_path), "manifest": str(manifest_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a defisheye parameter grid for quick visual tuning.")
    parser.add_argument("--image", required=True, help="Input fisheye image.")
    parser.add_argument("--out-dir", required=True, help="Directory for variants and contact sheet.")
    parser.add_argument("--format", choices=("circular", "fullframe"), default="circular")
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES), help="Comma-separated: linear,equalarea,orthographic,stereographic")
    parser.add_argument("--fovs", default=",".join(str(x) for x in DEFAULT_FOVS), help="Comma-separated input fisheye FOV values.")
    parser.add_argument("--pfovs", default=",".join(str(x) for x in DEFAULT_PFOVS), help="Comma-separated output perspective FOV values.")
    parser.add_argument("--xcenter", type=float, default=None)
    parser.add_argument("--ycenter", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--auto-circle", action="store_true", help="Estimate circular fisheye center/radius from non-black pixels.")
    parser.add_argument("--black-threshold", type=int, default=12, help="Pixel threshold for --auto-circle black border detection.")
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--label-height", type=int, default=46)
    parser.add_argument("--columns", type=int, default=5)
    return parser.parse_args()


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("This script requires opencv-python. Install it in the same env as defisheye.") from exc
    return cv2


def import_defisheye():
    try:
        from defisheye import Defisheye  # type: ignore
    except ImportError as exc:
        raise ImportError("This script requires duducosmos/defisheye. Install with: pip install defisheye") from exc
    return Defisheye


def resolve_circle(
    *,
    image: np.ndarray,
    xcenter: float | None,
    ycenter: float | None,
    radius: float | None,
    auto_circle: bool,
    black_threshold: int,
) -> CircleEstimate:
    height, width = image.shape[:2]
    default = CircleEstimate(
        xcenter=float(width) / 2.0 if xcenter is None else float(xcenter),
        ycenter=float(height) / 2.0 if ycenter is None else float(ycenter),
        radius=float(min(width, height)) / 2.0 if radius is None else float(radius),
        method="manual_or_default",
    )
    if not auto_circle:
        return default

    cv2 = import_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > int(black_threshold)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return default
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), detected_radius = cv2.minEnclosingCircle(contour)
    return CircleEstimate(
        xcenter=float(xcenter) if xcenter is not None else float(cx),
        ycenter=float(ycenter) if ycenter is not None else float(cy),
        radius=float(radius) if radius is not None else float(detected_radius),
        method="auto_non_black_min_enclosing_circle",
    )


def run_defisheye_variant(
    *,
    image_path: Path,
    out_dir: Path,
    dtype: str,
    fisheye_format: str,
    fov: float,
    pfov: float,
    circle: CircleEstimate,
    angle: float,
    pad: int,
) -> dict[str, Any]:
    Defisheye = import_defisheye()
    output_name = (
        f"{image_path.stem}"
        f"__dtype-{dtype}"
        f"__format-{fisheye_format}"
        f"__fov-{fov:g}"
        f"__pfov-{pfov:g}.jpg"
    )
    output_path = out_dir / output_name
    obj = Defisheye(
        str(image_path),
        dtype=dtype,
        format=fisheye_format,
        fov=float(fov),
        pfov=float(pfov),
        xcenter=float(circle.xcenter),
        ycenter=float(circle.ycenter),
        radius=float(circle.radius),
        angle=float(angle),
        pad=int(pad),
    )
    obj.convert(outfile=str(output_path))
    return {
        "path": str(output_path),
        "dtype": dtype,
        "format": fisheye_format,
        "fov": float(fov),
        "pfov": float(pfov),
        "xcenter": float(circle.xcenter),
        "ycenter": float(circle.ycenter),
        "radius": float(circle.radius),
        "angle": float(angle),
        "pad": int(pad),
    }


def build_contact_sheet(
    *,
    cv2: Any,
    variants: list[dict[str, Any]],
    output_path: Path,
    tile_width: int,
    label_height: int,
    columns: int,
) -> None:
    if not variants:
        raise ValueError("No defisheye variants were produced")
    columns = max(1, int(columns))
    tiles = []
    for variant in variants:
        image = cv2.imread(str(variant["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read generated variant: {variant['path']}")
        height, width = image.shape[:2]
        scale = float(tile_width) / float(width)
        tile_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        label = np.full((label_height, tile_width, 3), 255, dtype=np.uint8)
        text_1 = f"{variant['dtype']}  fov={variant['fov']:g}  pfov={variant['pfov']:g}"
        text_2 = f"r={variant['radius']:.1f}  c=({variant['xcenter']:.1f},{variant['ycenter']:.1f})"
        cv2.putText(label, text_1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(label, text_2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(np.vstack([resized, label]))

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = int(tile_width)
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns
        y = row * tile_h
        x = col * tile_w
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def parse_float_list(value: str, *, default: tuple[float, ...]) -> list[float]:
    if not value.strip():
        return [float(x) for x in default]
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def parse_str_list(value: str, *, default: tuple[str, ...]) -> list[str]:
    if not value.strip():
        return [str(x) for x in default]
    return [token.strip() for token in value.split(",") if token.strip()]


if __name__ == "__main__":
    main()
