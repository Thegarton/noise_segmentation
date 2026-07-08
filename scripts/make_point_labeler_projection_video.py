#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IGNORE_IDS = {0, 255}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_FPS = 10
DEFAULT_POINT_SIZE = 10
DEFAULT_CLASS_IDS = (2, 5, 10, 11, 13, 15, 18, 19, 20, 21)


def main() -> None:
    args = parse_args()
    labeler_dir = Path(args.labeler_dir).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    class_ids = parse_class_ids(args.class_ids)
    id_to_name, id_to_color = read_labels_xml(labeler_dir / "labels.xml")
    bridge_manifest = read_json(labeler_dir / "bridge_manifest.json")
    frame_manifest_by_id = {str(frame.get("frame_id")): frame for frame in bridge_manifest.get("frames", [])}

    frame_ids = discover_frame_ids(export_dir, max_frames=args.max_frames)
    if not frame_ids:
        raise FileNotFoundError(f"No exported frame folders found in {export_dir}")

    first_frame = render_frame(
        frame_ids[0],
        labeler_dir=labeler_dir,
        export_dir=export_dir,
        frame_manifest_by_id=frame_manifest_by_id,
        id_to_name=id_to_name,
        id_to_color=id_to_color,
        selected_class_ids=class_ids,
        point_size=args.point_size,
        legend_position=args.legend_position,
        legend_mode=args.legend_mode,
    )
    frame_size = first_frame.size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_path, fps=args.fps, frame_size=frame_size)

    written = 0
    try:
        write_frame(writer, first_frame)
        written += 1
        for frame_id in frame_ids[1:]:
            image = render_frame(
                frame_id,
                labeler_dir=labeler_dir,
                export_dir=export_dir,
                frame_manifest_by_id=frame_manifest_by_id,
                id_to_name=id_to_name,
                id_to_color=id_to_color,
                selected_class_ids=class_ids,
                point_size=args.point_size,
                legend_position=args.legend_position,
                legend_mode=args.legend_mode,
            )
            if image.size != frame_size:
                image = image.resize(frame_size, Image.Resampling.LANCZOS)
            write_frame(writer, image)
            written += 1
            if written % 25 == 0 or written == len(frame_ids):
                print(f"[{written}/{len(frame_ids)}] wrote {frame_id}", flush=True)
    finally:
        close_video_writer(writer)

    manifest = {
        "version": 1,
        "labeler_dir": str(labeler_dir),
        "export_dir": str(export_dir),
        "output": str(output_path),
        "fps": float(args.fps),
        "point_size": float(args.point_size),
        "legend_mode": args.legend_mode,
        "legend_position": args.legend_position,
        "class_ids": class_ids,
        "ignored_class_ids": sorted(IGNORE_IDS),
        "frame_count": written,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "classes": [
            {"id": int(class_id), "name": id_to_name.get(class_id, f"id_{class_id}"), "color": id_to_color[class_id].tolist()}
            for class_id in class_ids
            if class_id in id_to_color and class_id not in IGNORE_IDS
        ],
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"video": str(output_path), "frames": written, "manifest": str(manifest_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project point_labeler exported flat masks to camera images and build a video with a class legend."
    )
    parser.add_argument("--labeler-dir", required=True, help="point_labeler dataset directory with labels.xml, image_2/, bridge_manifest.json.")
    parser.add_argument("--export-dir", required=True, help="Export directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--output", required=True, help="Output video path, usually .mp4.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--class-ids",
        default=",".join(str(value) for value in DEFAULT_CLASS_IDS),
        help="Comma-separated class ids to draw and show in the legend. Default: 2,5,10,11,13,15,18,19,20,21.",
    )
    parser.add_argument(
        "--legend-mode",
        choices=("all", "present"),
        default="all",
        help="present: show only classes visible in the current frame. all: show all non-background/non-ignore classes from labels.xml.",
    )
    parser.add_argument(
        "--legend-position",
        choices=("top-right", "top-left"),
        default="top-right",
    )
    return parser.parse_args()


def parse_class_ids(value: str) -> list[int]:
    class_ids = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if token:
            class_id = int(token)
            if class_id not in IGNORE_IDS:
                class_ids.append(class_id)
    if not class_ids:
        raise ValueError("--class-ids must contain at least one non-background/non-ignore class id")
    return class_ids


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_labels_xml(path: Path) -> tuple[dict[int, str], dict[int, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"labels.xml does not exist: {path}")
    root = ET.parse(path).getroot()
    id_to_name: dict[int, str] = {}
    id_to_color: dict[int, np.ndarray] = {}
    for label in root.findall("label"):
        raw_id = (label.findtext("id") or "").strip()
        name = (label.findtext("name") or "").strip()
        if not raw_id or not name:
            continue
        label_id = int(raw_id)
        id_to_name[label_id] = name
        raw_color = (label.findtext("color") or "").strip()
        if raw_color:
            values = [int(float(token)) for token in raw_color.replace(",", " ").split()[:3]]
            if len(values) == 3:
                id_to_color[label_id] = np.asarray(values, dtype=np.uint8)
    id_to_name.setdefault(0, "background")
    id_to_name.setdefault(255, "ignore")
    id_to_color.setdefault(0, np.asarray([32, 32, 32], dtype=np.uint8))
    id_to_color.setdefault(255, np.asarray([125, 125, 125], dtype=np.uint8))
    return id_to_name, id_to_color


def discover_frame_ids(export_dir: Path, *, max_frames: int | None) -> list[str]:
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")
    frame_ids = sorted(path.name for path in export_dir.iterdir() if path.is_dir() and (path / "semantic_mask.npy").is_file())
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def render_frame(
    frame_id: str,
    *,
    labeler_dir: Path,
    export_dir: Path,
    frame_manifest_by_id: dict[str, dict[str, Any]],
    id_to_name: dict[int, str],
    id_to_color: dict[int, np.ndarray],
    selected_class_ids: list[int],
    point_size: float,
    legend_position: str,
    legend_mode: str,
) -> Image.Image:
    mask = np.load(export_dir / frame_id / "semantic_mask.npy", allow_pickle=False).reshape(-1)
    csv_cols = load_csv_columns(csv_path_for_frame(frame_id, labeler_dir=labeler_dir, frame_manifest_by_id=frame_manifest_by_id))
    image_path = image_path_for_frame(frame_id, labeler_dir=labeler_dir, frame_manifest_by_id=frame_manifest_by_id)
    image = load_image(image_path)

    labels, csv_cols = align_mask_and_csv(mask, csv_cols)
    image = draw_projected_points(
        image,
        labels=labels,
        cxd=csv_cols["cxd"],
        cyd=csv_cols["cyd"],
        id_to_color=id_to_color,
        selected_class_ids=selected_class_ids,
        point_size=point_size,
    )
    legend_ids = legend_class_ids(labels, selected_class_ids=selected_class_ids, mode=legend_mode)
    return draw_legend(
        image,
        class_ids=legend_ids,
        id_to_name=id_to_name,
        id_to_color=id_to_color,
        frame_id=frame_id,
        position=legend_position,
    )


def csv_path_for_frame(
    frame_id: str,
    *,
    labeler_dir: Path,
    frame_manifest_by_id: dict[str, dict[str, Any]],
) -> Path:
    frame = frame_manifest_by_id.get(frame_id, {})
    raw = frame.get("source_csv")
    if raw and Path(raw).is_file():
        return Path(raw)
    for folder in ("csv", "CSV"):
        candidate = labeler_dir / folder / f"{frame_id}.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"CSV for frame {frame_id} was not found in bridge_manifest or {labeler_dir}/csv")


def image_path_for_frame(
    frame_id: str,
    *,
    labeler_dir: Path,
    frame_manifest_by_id: dict[str, dict[str, Any]],
) -> Path:
    frame = frame_manifest_by_id.get(frame_id, {})
    for key in ("image", "source_image"):
        raw = frame.get(key)
        if raw and Path(raw).is_file():
            return Path(raw)
    for suffix in IMAGE_SUFFIXES:
        candidate = labeler_dir / "image_2" / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image for frame {frame_id} was not found in bridge_manifest or {labeler_dir}/image_2")


def split_table_row(line: str) -> list[str]:
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return line.replace("\t", " ").split()


def load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = None
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                header = split_table_row(stripped)
                break
        if header is None:
            raise ValueError(f"CSV is empty: {path}")
        column_map = {name.strip().casefold(): index for index, name in enumerate(header)}
        required = {
            "cxd": require_column(column_map, "cxd", path),
            "cyd": require_column(column_map, "cyd", path),
        }
        max_column = max(required.values())
        values = {name: [] for name in required}
        for line_number, raw_line in enumerate(handle, start=2):
            stripped = raw_line.strip()
            if not stripped:
                continue
            tokens = split_table_row(stripped)
            if len(tokens) <= max_column:
                raise ValueError(f"{path}:{line_number}: expected at least {max_column + 1} columns, got {len(tokens)}")
            for name, column in required.items():
                values[name].append(parse_float(tokens[column], path, line_number, name))
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def require_column(column_map: dict[str, int], name: str, source: Path) -> int:
    key = name.casefold()
    if key not in column_map:
        raise ValueError(f"CSV {source} is missing required column {name!r}")
    return int(column_map[key])


def parse_float(value: str, source: Path, line_number: int, column: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: cannot parse {column}={value!r} as float") from exc


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def align_mask_and_csv(mask: np.ndarray, csv_cols: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = [int(mask.size), int(csv_cols["cxd"].size), int(csv_cols["cyd"].size)]
    target = min(lengths)
    if len(set(lengths)) != 1:
        print(
            f"WARNING: mask/CSV length mismatch: mask={mask.size} cxd={csv_cols['cxd'].size} cyd={csv_cols['cyd'].size}; using first {target}",
            file=sys.stderr,
        )
    return mask[:target], {key: values[:target] for key, values in csv_cols.items()}


def draw_projected_points(
    image: Image.Image,
    *,
    labels: np.ndarray,
    cxd: np.ndarray,
    cyd: np.ndarray,
    id_to_color: dict[int, np.ndarray],
    selected_class_ids: list[int],
    point_size: float,
) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    radius = max(1.0, float(point_size) / 2.0)
    height = image.height
    width = image.width
    valid = np.isfinite(cxd) & np.isfinite(cyd)
    valid &= (cxd >= 0.0) & (cxd < width) & (cyd >= 0.0) & (cyd < height)
    valid &= ~np.isin(labels, list(IGNORE_IDS))
    valid &= np.isin(labels, selected_class_ids)

    for label_id in sorted(int(value) for value in np.unique(labels[valid])):
        color = id_to_color.get(label_id, stable_color(label_id))
        rgb = tuple(int(value) for value in color.tolist())
        class_mask = valid & (labels == label_id)
        xs = cxd[class_mask]
        ys = cyd[class_mask]
        for x, y in zip(xs, ys):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*rgb, 230))
    return image


def stable_color(label_id: int) -> np.ndarray:
    rng = np.random.default_rng((int(label_id) * 1009 + 17) & 0xFFFFFFFF)
    return rng.integers(40, 240, size=3, dtype=np.uint8)


def legend_class_ids(labels: np.ndarray, *, selected_class_ids: list[int], mode: str) -> list[int]:
    available = [class_id for class_id in selected_class_ids if class_id not in IGNORE_IDS]
    if mode == "all":
        return available
    present = {int(value) for value in np.unique(labels) if int(value) not in IGNORE_IDS}
    return [class_id for class_id in available if class_id in present]


def draw_legend(
    image: Image.Image,
    *,
    class_ids: list[int],
    id_to_name: dict[int, str],
    id_to_color: dict[int, np.ndarray],
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

    title = frame_id
    rows = [(class_id, id_to_name.get(class_id, f"id_{class_id}")) for class_id in class_ids]
    text_widths = [text_bbox_width(draw, name, font) for _, name in rows]
    title_width = text_bbox_width(draw, title, title_font)
    panel_width = max(title_width, swatch + pad + max(text_widths, default=0)) + 2 * pad
    title_height = text_bbox_height(draw, title, title_font)
    row_height = max(swatch, text_bbox_height(draw, "Ag", font))
    panel_height = 2 * pad + title_height + row_gap + len(rows) * row_height + max(0, len(rows) - 1) * row_gap

    x0 = margin if position == "top-left" else image.width - margin - panel_width
    y0 = margin
    x1 = x0 + panel_width
    y1 = y0 + panel_height
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(6, pad // 2), fill=(0, 0, 0, 170))
    draw.text((x0 + pad, y0 + pad), title, font=title_font, fill=(255, 255, 255, 245))

    y = y0 + pad + title_height + row_gap
    for class_id, name in rows:
        color = id_to_color.get(class_id, stable_color(class_id))
        rgb = tuple(int(value) for value in color.tolist())
        y_swatch = y + max(0, (row_height - swatch) // 2)
        draw.rounded_rectangle(
            (x0 + pad, y_swatch, x0 + pad + swatch, y_swatch + swatch),
            radius=max(2, swatch // 8),
            fill=(*rgb, 255),
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
        raise ImportError("OpenCV is required to write video. Install opencv-python in this env.") from exc

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
