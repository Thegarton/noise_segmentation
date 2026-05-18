#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np


DEFAULT_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
        [255, 255, 255],
        [110, 70, 20],
        [0, 100, 0],
        [90, 20, 120],
        [30, 30, 30],
        [200, 120, 40],
        [40, 120, 200],
        [120, 40, 200],
        [200, 40, 120],
        [80, 160, 60],
    ],
    dtype=np.uint8,
)


LITEPT_NUSCENES_CLASS_NAMES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize semantic_mask.npy as a color PNG")
    parser.add_argument("--mask", required=True, help="Path to semantic_mask.npy")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--confidence", default=None, help="Optional confidence.npy path")
    parser.add_argument("--metadata", default=None, help="Optional metadata.json with class_names")
    parser.add_argument("--legend", default=None, help="Optional legend JSON path")
    parser.add_argument("--scale", type=int, default=3, help="Nearest-neighbor pixel scale")
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()

    mask = np.load(args.mask)
    if mask.ndim != 2:
        raise SystemExit(f"Mask must be 2D, got shape={mask.shape}")
    if args.scale <= 0:
        raise SystemExit(f"--scale must be positive, got {args.scale}")

    metadata = _load_metadata(args.metadata)
    class_names = metadata.get("class_names") or LITEPT_NUSCENES_CLASS_NAMES

    rgb = colorize_mask(mask, ignore_index=args.ignore_index)
    if args.confidence:
        confidence = np.load(args.confidence)
        rgb = apply_confidence(rgb, confidence)
    if args.scale != 1:
        rgb = np.repeat(np.repeat(rgb, args.scale, axis=0), args.scale, axis=1)

    write_png(args.output, rgb)

    if args.legend:
        legend = build_legend(mask, class_names=class_names, ignore_index=args.ignore_index)
        Path(args.legend).parent.mkdir(parents=True, exist_ok=True)
        Path(args.legend).write_text(json.dumps(legend, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(Path(args.output).resolve()), "shape": list(mask.shape)}, indent=2))


def colorize_mask(mask: np.ndarray, *, ignore_index: int = 255) -> np.ndarray:
    labels = np.asarray(mask)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels != ignore_index
    if valid.any():
        rgb[valid] = DEFAULT_COLORS[labels[valid].astype(np.int64) % len(DEFAULT_COLORS)]
    rgb[~valid] = np.array([15, 15, 15], dtype=np.uint8)
    return rgb


def apply_confidence(rgb: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    conf = np.asarray(confidence, dtype=np.float32)
    if conf.shape != rgb.shape[:2]:
        raise SystemExit(f"Confidence shape must match mask shape, got {conf.shape} vs {rgb.shape[:2]}")
    conf = np.clip(conf, 0.0, 1.0)
    # Low confidence becomes darker, but keeps the semantic color.
    factor = 0.25 + 0.75 * conf[..., None]
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def build_legend(mask: np.ndarray, *, class_names: list[str], ignore_index: int = 255) -> list[dict]:
    labels, counts = np.unique(mask, return_counts=True)
    legend = []
    for label, count in zip(labels.tolist(), counts.tolist()):
        if int(label) == ignore_index:
            name = "ignore"
            color = [15, 15, 15]
        else:
            name = class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
            color = DEFAULT_COLORS[int(label) % len(DEFAULT_COLORS)].tolist()
        legend.append({"id": int(label), "name": name, "count": int(count), "color": color})
    return legend


def write_png(path: str, rgb: np.ndarray) -> None:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"PNG image must have shape [H,W,3], got {image.shape}")

    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _load_metadata(path: str | None) -> dict:
    if not path:
        return {}
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
