#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from make_point_labeler_projection_video import (  # noqa: E402
    DEFAULT_CLASS_IDS,
    DEFAULT_FPS,
    DEFAULT_POINT_SIZE,
    IGNORE_IDS,
    IMAGE_SUFFIXES,
    align_mask_and_csv,
    close_video_writer,
    draw_legend,
    draw_projected_points,
    load_csv_columns,
    open_video_writer,
    parse_class_ids,
    read_labels_xml,
    write_frame,
)


DEFAULT_CLASS_NAMES = {
    2: "CAR",
    5: "PEDESTRIAN",
    10: "traffic_sign",
    11: "roadblock",
    13: "tire",
    15: "traffic_cone",
    18: "adhesive_noise",
    19: "Long_distance_noise",
    20: "underground_noise",
    21: "Multiple_Distances",
}

DEFAULT_CLASS_COLORS = {
    2: [146, 113, 58],
    5: [220, 48, 79],
    10: [112, 84, 238],
    11: [191, 171, 235],
    13: [76, 68, 117],
    15: [66, 165, 42],
    18: [85, 255, 0],
    19: [255, 255, 0],
    20: [0, 0, 127],
    21: [255, 85, 0],
}


def main() -> None:
    args = parse_args()
    inference_dir = Path(args.inference_dir).expanduser().resolve()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    class_ids = parse_class_ids(args.class_ids)
    id_to_name, id_to_color = load_classes(args.labels_xml)

    frame_ids = discover_frame_ids(inference_dir, max_frames=args.max_frames)
    if not frame_ids:
        raise FileNotFoundError(f"No inference frame folders with semantic_mask.npy found in {inference_dir}")

    first_frame = render_frame(
        frame_ids[0],
        inference_dir=inference_dir,
        csv_dir=csv_dir,
        image_dir=image_dir,
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
                inference_dir=inference_dir,
                csv_dir=csv_dir,
                image_dir=image_dir,
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
        "mode": "litept_flat_csv_inference_projection_video",
        "inference_dir": str(inference_dir),
        "csv_dir": str(csv_dir),
        "image_dir": str(image_dir),
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
        description="Project LitePT flat CSV inference masks to camera images and build a video with a fixed class legend."
    )
    parser.add_argument("--inference-dir", required=True, help="Directory with <frame>/semantic_mask.npy from run_litept_flat_csv_inference.py.")
    parser.add_argument("--csv-dir", required=True, help="Directory with matching <frame>.csv files containing Cxd/Cyd.")
    parser.add_argument("--image-dir", required=True, help="Directory with matching <frame>.jpg images.")
    parser.add_argument("--output", required=True, help="Output video path, usually .mp4.")
    parser.add_argument("--labels-xml", default=None, help="Optional labels.xml used for class names/colors.")
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
        help="all: always show selected classes. present: show only selected classes visible in the current frame.",
    )
    parser.add_argument(
        "--legend-position",
        choices=("top-right", "top-left"),
        default="top-right",
    )
    return parser.parse_args()


def load_classes(labels_xml: str | None) -> tuple[dict[int, str], dict[int, np.ndarray]]:
    id_to_name = dict(DEFAULT_CLASS_NAMES)
    id_to_color = {class_id: np.asarray(color, dtype=np.uint8) for class_id, color in DEFAULT_CLASS_COLORS.items()}
    if labels_xml:
        xml_names, xml_colors = read_labels_xml(Path(labels_xml).expanduser().resolve())
        id_to_name.update(xml_names)
        id_to_color.update(xml_colors)
    return id_to_name, id_to_color


def discover_frame_ids(inference_dir: Path, *, max_frames: int | None) -> list[str]:
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference dir does not exist: {inference_dir}")
    frame_ids = sorted(
        path.name
        for path in inference_dir.iterdir()
        if path.is_dir() and (path / "semantic_mask.npy").is_file()
    )
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def render_frame(
    frame_id: str,
    *,
    inference_dir: Path,
    csv_dir: Path,
    image_dir: Path,
    id_to_name: dict[int, str],
    id_to_color: dict[int, np.ndarray],
    selected_class_ids: list[int],
    point_size: float,
    legend_position: str,
    legend_mode: str,
) -> Image.Image:
    mask = np.load(inference_dir / frame_id / "semantic_mask.npy", allow_pickle=False).reshape(-1)
    csv_cols = load_csv_columns(csv_path_for_frame(frame_id, csv_dir=csv_dir))
    image = Image.open(image_path_for_frame(frame_id, image_dir=image_dir)).convert("RGB")
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


def csv_path_for_frame(frame_id: str, *, csv_dir: Path) -> Path:
    path = csv_dir / f"{frame_id}.csv"
    if path.is_file():
        return path
    raise FileNotFoundError(f"CSV for frame {frame_id} does not exist: {path}")


def image_path_for_frame(frame_id: str, *, image_dir: Path) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(
        path
        for path in image_dir.glob(f"{frame_id}*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image for frame {frame_id} was not found in {image_dir}")


def legend_class_ids(labels: np.ndarray, *, selected_class_ids: list[int], mode: str) -> list[int]:
    if mode == "all":
        return selected_class_ids
    present = {int(value) for value in np.unique(labels) if int(value) not in IGNORE_IDS}
    return [class_id for class_id in selected_class_ids if class_id in present]


if __name__ == "__main__":
    main()
