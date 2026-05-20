#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.bin_loader import C, H, W
from autolabeler.data.frame_loader import load_frame


def main() -> None:
    args = parse_args()
    frame_id = args.frame_id or Path(args.points).stem
    points, intensity = load_points(args.points, args.input_format, frame_id)
    point_count = len(points)

    semantic = load_semantic_labels(
        mask_path=args.semantic_mask,
        labels_jsonl=args.labels_jsonl,
        frame_id=frame_id,
        point_count=point_count,
        height=args.height,
        width=args.width,
        allow_resize=args.allow_range_resize,
    )
    confidence = load_confidence(
        args.confidence,
        point_count=point_count,
        height=args.height,
        width=args.width,
        allow_resize=args.allow_range_resize,
    )

    keep = np.isfinite(points).all(axis=1)
    if args.drop_zero_points:
        keep &= np.linalg.norm(points, axis=1) > 0.0
    points = points[keep]
    semantic = semantic[keep]
    confidence = confidence[keep] if confidence is not None else None
    intensity = intensity[keep] if intensity is not None else None

    show_point_cloud(
        points=points,
        semantic=semantic,
        confidence=confidence,
        intensity=intensity,
        point_size=args.point_size,
        color_by=args.color_by,
        screenshot=args.screenshot,
        background=args.background,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize point-wise semantic segmentation with PyVista.")
    p.add_argument(
        "--points",
        required=True,
        help="Path to organized frame (.bin/.csv) or point cloud .npy with shape (N, 4), (H, W, 4), or flat float32.",
    )
    p.add_argument("--input-format", choices=["auto", "bin", "csv", "npy"], default="auto")
    p.add_argument("--frame-id", default=None, help="Frame id for labels.jsonl lookup. Defaults to points file stem.")
    p.add_argument("--semantic-mask", default=None, help="Path to semantic_mask.npy or raw mask file.")
    p.add_argument("--confidence", default=None, help="Optional confidence .npy/raw file.")
    p.add_argument("--labels-jsonl", default=None, help="Optional labels.jsonl from run_full_autolabeling.py.")
    p.add_argument("--height", type=int, default=H)
    p.add_argument("--width", type=int, default=W)
    p.add_argument(
        "--allow-range-resize",
        action="store_true",
        help="Nearest-neighbor resize for range masks, e.g. 192x120 -> 192x480.",
    )
    p.add_argument(
        "--color-by",
        choices=["semantic", "confidence", "intensity"],
        default="semantic",
        help="Scalar field used for point coloring.",
    )
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--background", default="black")
    p.add_argument("--screenshot", default=None, help="Optional output image path. Uses off-screen rendering.")
    p.add_argument(
        "--drop-zero-points",
        action="store_true",
        help="Hide points at exactly (0, 0, 0), useful for padded frames.",
    )
    return p.parse_args()


def load_points(path: str, input_format: str, frame_id: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    p = Path(path)
    fmt = p.suffix.lstrip(".").lower() if input_format == "auto" else input_format
    frame_id = frame_id or p.stem

    if fmt in {"bin", "csv"}:
        frame = load_frame(str(p), frame_id, input_format=fmt)
        flat = np.asarray(frame.points_flat, dtype=np.float32)
    elif fmt == "npy":
        arr = np.load(p, allow_pickle=False)
        flat = normalize_point_array(arr, source=str(p))
    else:
        raise ValueError(f"Unsupported point input format: {input_format} for {path}")

    if flat.shape[1] < 3:
        raise ValueError(f"Point cloud must have at least xyz columns, got shape {flat.shape}")
    points = flat[:, :3].astype(np.float32, copy=False)
    intensity = flat[:, 3].astype(np.float32, copy=False) if flat.shape[1] > 3 else None
    return points, intensity


def normalize_point_array(arr: np.ndarray, *, source: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr.reshape(-1, arr.shape[-1]).astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        if arr.size % C != 0:
            raise ValueError(f"{source} has flat size {arr.size}, not divisible by {C}")
        return arr.reshape(-1, C).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported point array shape in {source}: {arr.shape}")


def load_semantic_labels(
    *,
    mask_path: str | None,
    labels_jsonl: str | None,
    frame_id: str | None,
    point_count: int,
    height: int,
    width: int,
    allow_resize: bool,
) -> np.ndarray:
    if mask_path:
        mask = load_mask_array(mask_path)
        return normalize_scalar_array(
            mask,
            point_count=point_count,
            height=height,
            width=width,
            allow_resize=allow_resize,
            name="semantic mask",
        ).astype(np.int32, copy=False)

    if labels_jsonl:
        if not frame_id:
            raise ValueError("--frame-id is required when --labels-jsonl is used")
        return labels_jsonl_to_semantic(labels_jsonl, frame_id, point_count)

    raise ValueError("Pass either --semantic-mask or --labels-jsonl")


def load_confidence(
    path: str | None,
    *,
    point_count: int,
    height: int,
    width: int,
    allow_resize: bool,
) -> np.ndarray | None:
    if not path:
        return None
    arr = load_mask_array(path)
    return normalize_scalar_array(
        arr,
        point_count=point_count,
        height=height,
        width=width,
        allow_resize=allow_resize,
        name="confidence",
    ).astype(np.float32, copy=False)


def load_mask_array(path: str) -> np.ndarray:
    p = Path(path)
    try:
        return np.load(p, allow_pickle=False)
    except ValueError:
        dtype = guess_raw_dtype(p)
        return np.fromfile(p, dtype=dtype)


def guess_raw_dtype(path: Path) -> np.dtype:
    name = path.name.lower()
    if "conf" in name:
        return np.dtype(np.float16)
    return np.dtype(np.uint8)


def normalize_scalar_array(
    arr: np.ndarray,
    *,
    point_count: int,
    height: int,
    width: int,
    allow_resize: bool,
    name: str,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    arr = strip_possible_raw_header(
        arr,
        point_count=point_count,
        height=height,
        width=width,
        allow_resize=allow_resize,
        name=name,
    )

    if arr.ndim == 2:
        if arr.shape == (height, width):
            return arr.reshape(-1)
        if allow_resize:
            resized = resize_range_mask_nearest(arr, height=height, width=width, name=name)
            return resized.reshape(-1)
        raise ValueError(
            f"{name} shape is {arr.shape}, expected {(height, width)}. "
            "Use --allow-range-resize for nearest-neighbor range-view resize."
        )

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape}")
    if arr.size == point_count:
        return arr
    if arr.size == height * width:
        if point_count != height * width:
            raise ValueError(
                f"{name} has organized size {height * width}, but point cloud has {point_count} points."
            )
        return arr
    if allow_resize and arr.size % height == 0:
        low_width = arr.size // height
        resized = resize_range_mask_nearest(arr.reshape(height, low_width), height=height, width=width, name=name)
        if resized.size != point_count:
            raise ValueError(f"Resized {name} has {resized.size} values, but point cloud has {point_count} points.")
        return resized.reshape(-1)
    raise ValueError(
        f"{name} has {arr.size} values, but point cloud has {point_count}; "
        f"expected {height * width} for {height}x{width}."
    )


def strip_possible_raw_header(
    arr: np.ndarray,
    *,
    point_count: int,
    height: int,
    width: int,
    allow_resize: bool,
    name: str,
) -> np.ndarray:
    if arr.ndim != 1:
        return arr

    expected_sizes = {point_count, height * width}
    if allow_resize:
        expected_sizes.update({height * low_width for low_width in range(1, width + 1) if width % low_width == 0})

    for extra in (4, 8, 16, 32, 64, 128):
        payload_size = arr.size - extra
        if payload_size in expected_sizes:
            print(f"{name}: stripped {extra} leading raw values that look like a file header")
            return arr[extra:]
    return arr


def resize_range_mask_nearest(arr: np.ndarray, *, height: int, width: int, name: str) -> np.ndarray:
    if arr.shape[0] != height:
        raise ValueError(f"{name} height is {arr.shape[0]}, expected {height}")
    if width % arr.shape[1] != 0:
        raise ValueError(f"{name} width {arr.shape[1]} cannot be repeated to target width {width}")
    repeat = width // arr.shape[1]
    return np.repeat(arr, repeat, axis=1)


def labels_jsonl_to_semantic(path: str, frame_id: str, point_count: int) -> np.ndarray:
    frame = find_jsonl_frame(path, frame_id)
    labels = np.zeros(point_count, dtype=np.int32)
    class_to_id: dict[str, int] = {}
    next_id = 1
    for item in frame.get("labels", []):
        semantic_class = str(item.get("semantic_class", "unknown"))
        if semantic_class not in class_to_id:
            class_to_id[semantic_class] = next_id
            next_id += 1
        class_id = class_to_id[semantic_class]
        point_indices = np.asarray(item.get("point_indices", []), dtype=np.int64)
        point_indices = point_indices[(point_indices >= 0) & (point_indices < point_count)]
        labels[point_indices] = class_id
    print("labels.jsonl class ids:", {"background": 0, **class_to_id})
    return labels


def find_jsonl_frame(path: str, frame_id: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("frame_id")) == str(frame_id):
                return item
    raise ValueError(f"Frame id {frame_id!r} was not found in {path}")


def show_point_cloud(
    *,
    points: np.ndarray,
    semantic: np.ndarray,
    confidence: np.ndarray | None,
    intensity: np.ndarray | None,
    point_size: float,
    color_by: str,
    screenshot: str | None,
    background: str,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit("PyVista is not installed. Install it with: pip install pyvista") from exc

    if color_by == "confidence" and confidence is None:
        raise ValueError("--color-by confidence requires --confidence")
    if color_by == "intensity" and intensity is None:
        raise ValueError("--color-by intensity requires a point cloud with intensity column")

    cloud = pv.PolyData(points)
    cloud["semantic"] = semantic
    if confidence is not None:
        cloud["confidence"] = confidence
    if intensity is not None:
        cloud["intensity"] = intensity

    plotter = pv.Plotter(off_screen=screenshot is not None)
    plotter.set_background(background)
    plotter.add_mesh(
        cloud,
        scalars=color_by,
        render_points_as_spheres=True,
        point_size=point_size,
        cmap="tab20" if color_by == "semantic" else "turbo",
        show_scalar_bar=True,
        scalar_bar_args={"title": color_by},
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = "xy"
    title = f"points={len(points)} semantic_classes={sorted(np.unique(semantic).astype(int).tolist())}"
    plotter.add_text(title, position="upper_left", font_size=10, color="white")

    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=screenshot, auto_close=True)
        print(f"screenshot={screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
