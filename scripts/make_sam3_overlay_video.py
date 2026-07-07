#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402


DEFAULT_CLASS_IDS = (2, 5, 10, 11, 13, 15)
DEFAULT_FPS = 10
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    class_ids = parse_class_ids(args.class_ids)
    id_to_name = load_class_names(args.classes_yaml)
    frames = discover_frames(input_dir, image_name=args.image_name, max_frames=args.max_frames)
    if not frames:
        raise FileNotFoundError(f"No frame folders with {args.image_name!r} found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = Image.open(frames[0]["image"]).convert("RGB")
    frame_size = first.size

    writer = open_video_writer(output_path, fps=args.fps, frame_size=frame_size)
    written = 0
    try:
        for item in frames:
            image = Image.open(item["image"]).convert("RGB")
            if image.size != frame_size:
                image = image.resize(frame_size, Image.Resampling.LANCZOS)
            semantic_mask = load_semantic_mask(item["dir"] / "semantic_mask.npy")
            shown_ids = class_ids_for_legend(
                selected_ids=class_ids,
                semantic_mask=semantic_mask,
                legend_mode=args.legend_mode,
            )
            image = draw_legend(
                image,
                class_ids=shown_ids,
                id_to_name=id_to_name,
                frame_id=item["frame_id"],
                position=args.legend_position,
            )
            write_frame(writer, image)
            written += 1
            if written % 25 == 0 or written == len(frames):
                print(f"[{written}/{len(frames)}] wrote {item['frame_id']}", flush=True)
    finally:
        close_video_writer(writer)

    manifest = {
        "version": 1,
        "input_dir": str(input_dir),
        "output": str(output_path),
        "fps": args.fps,
        "image_name": args.image_name,
        "legend_mode": args.legend_mode,
        "legend_position": args.legend_position,
        "class_ids": class_ids,
        "classes": [{"id": class_id, "name": id_to_name.get(class_id, f"id_{class_id}")} for class_id in class_ids],
        "frame_count": written,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(output_path), "frames": written, "manifest": str(manifest_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MP4 from SAM3 per-frame overlays and draw a class legend.")
    parser.add_argument("--input-dir", required=True, help="Directory with per-frame SAM3 folders, e.g. output/sam3_single_image_folder_1090_1245.")
    parser.add_argument("--output", required=True, help="Output video path, usually .mp4.")
    parser.add_argument("--classes-yaml", default=None, help="YAML with semantic_classes. Used for class names.")
    parser.add_argument("--image-name", default="overlay.jpg", help="Image inside each frame folder. Default: overlay.jpg.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--class-ids",
        default=",".join(str(value) for value in DEFAULT_CLASS_IDS),
        help="Comma-separated class ids to show in the legend. Default: 2,5,10,11,13,15.",
    )
    parser.add_argument(
        "--legend-mode",
        choices=("all", "present"),
        default="all",
        help="all: always show selected classes. present: show only selected classes present in semantic_mask.npy.",
    )
    parser.add_argument(
        "--legend-position",
        choices=("top-right", "top-left"),
        default="top-right",
    )
    return parser.parse_args()


def parse_class_ids(value: str) -> list[int]:
    ids = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if token:
            ids.append(int(token))
    if not ids:
        raise ValueError("--class-ids must contain at least one id")
    return ids


def load_class_names(classes_yaml: str | None) -> dict[int, str]:
    defaults = {
        2: "CAR",
        5: "PEDESTRIAN",
        10: "traffic_sign",
        11: "roadblock",
        13: "tire",
        15: "traffic_cone",
    }
    if not classes_yaml:
        return defaults
    path = Path(classes_yaml).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Classes YAML does not exist: {path}")
    class_to_id = load_semantic_classes(str(path))
    id_to_name = {int(class_id): str(name) for name, class_id in class_to_id.items()}
    for class_id, name in defaults.items():
        id_to_name.setdefault(class_id, name)
    return id_to_name


def discover_frames(input_dir: Path, *, image_name: str, max_frames: int | None) -> list[dict[str, Any]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")
    frames = []
    for frame_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        image_path = frame_dir / image_name
        if not image_path.is_file():
            image_path = find_first_image(frame_dir)
        if image_path is None:
            continue
        frames.append({"frame_id": frame_dir.name, "dir": frame_dir, "image": image_path})
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        frames = frames[:max_frames]
    return frames


def find_first_image(frame_dir: Path) -> Path | None:
    for stem in ("overlay", "preview", "mask_projection", "image"):
        for suffix in IMAGE_SUFFIXES:
            candidate = frame_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def load_semantic_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def class_ids_for_legend(
    *,
    selected_ids: list[int],
    semantic_mask: np.ndarray | None,
    legend_mode: str,
) -> list[int]:
    if legend_mode == "all" or semantic_mask is None:
        return selected_ids
    present = {int(value) for value in np.unique(semantic_mask)}
    return [class_id for class_id in selected_ids if class_id in present]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(class_id) * 1009 + 17)
    values = rng.integers(40, 240, size=3, dtype=np.uint8)
    return int(values[0]), int(values[1]), int(values[2])


def draw_legend(
    image: Image.Image,
    *,
    class_ids: list[int],
    id_to_name: dict[int, str],
    frame_id: str,
    position: str,
) -> Image.Image:
    if not class_ids:
        return image

    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(max(16, image.width // 90))
    title_font = load_font(max(18, image.width // 80))
    margin = max(18, image.width // 90)
    pad = max(10, image.width // 160)
    swatch = max(16, image.width // 95)
    row_gap = max(7, image.height // 170)

    title = f"{frame_id}"
    rows = [(class_id, id_to_name.get(class_id, f"id_{class_id}")) for class_id in class_ids]
    text_widths = [text_bbox_width(draw, name, font) for _, name in rows]
    title_width = text_bbox_width(draw, title, title_font)
    panel_width = max(title_width, swatch + pad + max(text_widths, default=0)) + 2 * pad
    title_height = text_bbox_height(draw, title, title_font)
    row_height = max(swatch, text_bbox_height(draw, "Ag", font))
    panel_height = 2 * pad + title_height + row_gap + len(rows) * row_height + max(0, len(rows) - 1) * row_gap

    if position == "top-left":
        x0 = margin
    else:
        x0 = image.width - margin - panel_width
    y0 = margin
    x1 = x0 + panel_width
    y1 = y0 + panel_height

    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(6, pad // 2), fill=(0, 0, 0, 170))
    draw.text((x0 + pad, y0 + pad), title, font=title_font, fill=(255, 255, 255, 245))

    y = y0 + pad + title_height + row_gap
    for class_id, name in rows:
        color = color_for_class(class_id)
        draw.rounded_rectangle(
            (x0 + pad, y + max(0, (row_height - swatch) // 2), x0 + pad + swatch, y + max(0, (row_height - swatch) // 2) + swatch),
            radius=max(2, swatch // 8),
            fill=(*color, 255),
            outline=(255, 255, 255, 180),
            width=1,
        )
        draw.text((x0 + pad + swatch + pad, y), name, font=font, fill=(255, 255, 255, 245))
        y += row_height + row_gap
    return image


def text_bbox_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left)


def text_bbox_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    _left, top, _right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(bottom - top)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def open_video_writer(path: Path, *, fps: float, frame_size: tuple[int, int]) -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("OpenCV is required to write the video. Install opencv-python in this env.") from exc

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def write_frame(writer: Any, image: Image.Image) -> None:
    import cv2  # type: ignore

    rgb = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    writer.write(bgr)


def close_video_writer(writer: Any) -> None:
    writer.release()


if __name__ == "__main__":
    main()
