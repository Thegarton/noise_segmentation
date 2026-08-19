#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from run_sam3_single_image_folder import (
    FolderDetectionSummary,
    collect_images,
    sam3_single_image_folder,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS_ROOT = REPO_ROOT / "configs"
LEGACY_CSV_TAGS_CONFIG = CONFIGS_ROOT / "sam3_csv_class_tags_zh_en.yaml"
SENSOR_CONFIG_FILENAMES = {
    "prompt_config": "sam3_text_prompts_v2.yaml",
    "classes_yaml": "classes_v2.yaml",
    "label_min_scores": "sam3_label_min_scores.yaml",
    "csv_tags_yaml": "sam3_csv_class_tags_zh_en.yaml",
}
DEFAULT_SAM3_ROOT = (REPO_ROOT / "../.." / "sam3").resolve()
DEFAULT_VEHICLE_ORIENTATION_ROOT = Path(
    os.environ.get("VEHICLE_ORIENTATION_ROOT", REPO_ROOT / "vehicle_orientation")
).expanduser().resolve()
DEFAULT_VEHICLE_ORIENTATION_CHECKPOINT = (
    DEFAULT_VEHICLE_ORIENTATION_ROOT
    / "output"
    / "vehicle_orientation_efficientnet_b2"
    / "model_best.pth"
)


def load_csv_class_tags(path: str | Path) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"CSV class-tags config does not exist: {config_path}")

    tags: dict[str, str] = {}
    in_class_tags = False
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        line = line_without_comment.strip()

        if indent == 0:
            in_class_tags = line == "class_tags:"
            continue
        if not in_class_tags:
            continue
        if indent != 2 or ":" not in line:
            raise ValueError(
                f"Expected '  english_label: bilingual tag' in "
                f"{config_path}:{line_number}, got {raw_line!r}"
            )

        label, tag = (value.strip() for value in line.split(":", 1))
        if len(tag) >= 2 and tag[0] in {"'", '"'} and tag[-1] == tag[0]:
            tag = tag[1:-1]
        if not label or not tag:
            raise ValueError(
                f"Empty class label or CSV tag in {config_path}:{line_number}"
            )
        if label in tags:
            raise ValueError(
                f"Duplicate class label {label!r} in {config_path}:{line_number}"
            )
        tags[label] = tag

    if not tags:
        raise ValueError(f"No entries found under class_tags in {config_path}")
    return tags


def format_csv_tags(classes: list[str], *, class_tags: dict[str, str]) -> str:
    return "; ".join(class_tags.get(label, label) for label in classes)


def find_sensor_config_dir(
    sensor_version: str,
    *,
    configs_root: str | Path = CONFIGS_ROOT,
) -> Path:
    version = str(sensor_version).strip()
    if not version:
        raise ValueError("--sensor-version must not be empty")
    if Path(version).name != version or "/" in version or "\\" in version:
        raise ValueError(
            f"--sensor-version must be a directory name, not a path: {sensor_version!r}"
        )

    root = Path(configs_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Configs directory does not exist: {root}")
    normalized = version.casefold()
    matches = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.strip().casefold() == normalized
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"Sensor version {version!r} is ambiguous under {root}: "
            f"{[path.name for path in matches]!r}"
        )
    available = sorted(path.name.strip() for path in root.iterdir() if path.is_dir())
    raise FileNotFoundError(
        f"Config directory for sensor version {version!r} was not found under {root}; "
        f"available versions={available!r}"
    )


def resolve_sensor_config_paths(
    args: argparse.Namespace,
    *,
    configs_root: str | Path = CONFIGS_ROOT,
) -> argparse.Namespace:
    sensor_dir = find_sensor_config_dir(args.sensor_version, configs_root=configs_root)
    for attribute, filename in SENSOR_CONFIG_FILENAMES.items():
        explicit_path = getattr(args, attribute, None)
        config_path = (
            Path(explicit_path).expanduser().resolve()
            if explicit_path
            else sensor_dir / filename
        )
        if not config_path.is_file():
            source = f"explicit --{attribute.replace('_', '-')}" if explicit_path else args.sensor_version
            raise FileNotFoundError(
                f"Missing {attribute.replace('_', ' ')} config for {source}: {config_path}"
            )
        setattr(args, attribute, str(config_path))
    args.sensor_config_dir = str(sensor_dir)
    return args


@dataclass(frozen=True)
class MatchedFrame:
    frame_id: str
    image_path: Path
    image_reference: str
    image_name: str
    image_timestamp: int
    diff_ms: float

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare sensor images and run SAM3.1 on one selected GPU."
    )
    parser.add_argument(
        "--sensor-version",
        required=True,
        help="Sensor config directory name under configs/, for example HL320.",
    )
    parser.add_argument(
        "--image-path",
        "--image-dir",
        dest="image_dir",
        required=True,
        help="Directory with prepared camera images.",
    )
    parser.add_argument(
        "--output-path",
        "--out-dir",
        dest="out_dir",
        required=True,
        help="Output directory; only sam3_run_summary.csv is retained.",
    )
    parser.add_argument(
        "--data-name",
        required=True,
        help="Dataset name written to sam3_run_summary.csv.",
    )
    parser.add_argument(
        "--csv-tags-yaml",
        default=None,
        help=(
            "Optional override for the CSV tag mapping. Default: "
            "configs/<sensor-version>/sam3_csv_class_tags_zh_en.yaml."
        ),
    )
    parser.add_argument(
        "--img-time-map",
        required=True,
        help=(
            "TXT with '<image-name>>><timestamp>' rows. The zero-based row index may be "
            "referenced by the second column of --img-match."
        ),
    )
    parser.add_argument(
        "--img-match",
        required=True,
        help="TXT with '<frame-id>>><image-index-or-name>>><diff-ms>' rows.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="First frame id from --img-match to process (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Last frame id from --img-match to process (inclusive).",
    )
    parser.add_argument(
        "--prompt-config",
        default=None,
        help="Optional override; default is sam3_text_prompts_v2.yaml in the sensor config directory.",
    )
    parser.add_argument(
        "--classes-yaml",
        default=None,
        help="Optional override; default is classes_v2.yaml in the sensor config directory.",
    )
    parser.add_argument("--sam3-root", default=str(DEFAULT_SAM3_ROOT))
    parser.add_argument("--sam3-model-path", default=str(DEFAULT_SAM3_ROOT / "sam3.1"))
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument(
        "--label-min-score",
        "--label-min-scores",
        dest="label_min_scores",
        default=None,
        help=(
            "Optional override for per-label thresholds. Default: "
            "sam3_label_min_scores.yaml in the sensor config directory."
        ),
    )
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--prompt-log", action="store_true")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-mask-size", type=int, default=500)
    parser.add_argument(
        "--class-min-frames",
        type=int,
        default=2,
        help=(
            "Write a class to sam3_run_summary.csv only when it is present in at "
            "least this many processed images."
        ),
    )
    parser.add_argument(
        "--class-min-consecutive-frames",
        type=int,
        default=2,
        help=(
            "Additionally require a class to be present in at least this many "
            "consecutive processed images."
        ),
    )

    parser.add_argument(
        "--vehicle-orientation-checkpoint",
        default=str(DEFAULT_VEHICLE_ORIENTATION_CHECKPOINT),
    )
    parser.add_argument("--vehicle-prompt-label", default="vehicle")
    parser.add_argument("--vehicle-orientation-device", default="cuda")
    parser.add_argument("--vehicle-orientation-min-confidence", type=float, default=0.60)
    parser.add_argument("--vehicle-orientation-min-margin", type=float, default=0.10)
    parser.add_argument(
        "--mask-nms-iou",
        "--vehicle-orientation-nms-iou",
        dest="vehicle_orientation_nms_iou",
        type=float,
        default=0.80,
    )
    parser.add_argument("--sam3-only", action="store_true")

    parser.add_argument(
        "--gpu-num",
        "--gpu-id",
        type=int,
        dest="gpu_id",
        required=True,
        help=(
            "Physical GPU id for this process. Sets CUDA_VISIBLE_DEVICES before loading "
            "SAM3; inside the process the selected card is cuda:0."
        ),
    )
    parser.add_argument(
        "--inference-precision",
        default="auto",
        choices=("auto", "fp16", "bf16", "fp32"),
        help="auto selects FP16 on T4 and BF16 on GPUs with native BF16 support.",
    )
    parser.add_argument(
        "--cache-visual-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute the single-image visual backbone once and reuse it for all text prompts.",
    )
    parser.add_argument(
        "--save-intermediate-outputs",
        action="store_true",
        help=(
            "Save per-frame masks, images, JSON metadata, and the SAM3 manifest. "
            "By default only sam3_run_summary.csv is written."
        ),
    )
    return parser


def configure_gpu(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    if gpu_id < 0:
        raise ValueError(f"--gpu-id must be non-negative, got {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["SAM3_PHYSICAL_GPU_ID"] = str(gpu_id)


def parse_delimited_rows(path: str | Path, *, columns: int) -> list[list[str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mapping file does not exist: {source}")

    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        values = [value.strip() for value in line.split(">>")]
        if len(values) != columns or any(not value for value in values):
            raise ValueError(
                f"Expected {columns} non-empty '>>'-separated values in "
                f"{source}:{line_number}, got {raw_line!r}"
            )
        rows.append(values)
    if not rows:
        raise ValueError(f"Mapping file contains no data rows: {source}")
    return rows


def index_images(image_dir: Path, *, recursive: bool = False) -> dict[str, Path]:
    image_paths = collect_images(image_dir, recursive=recursive)
    by_stem: dict[str, Path] = {}
    for image_path in image_paths:
        existing = by_stem.get(image_path.stem)
        if existing is not None:
            raise ValueError(
                f"Image stem {image_path.stem!r} is ambiguous: {existing} and {image_path}"
            )
        by_stem[image_path.stem] = image_path
    return by_stem


def load_matched_frames(
    *,
    image_dir: str | Path,
    img_time_map: str | Path,
    img_match: str | Path,
    start_frame: int | None,
    end_frame: int | None,
    recursive: bool = False,
) -> list[MatchedFrame]:
    if start_frame is not None and start_frame < 0:
        raise ValueError(f"--start-frame must be non-negative, got {start_frame}")
    if end_frame is not None and end_frame < 0:
        raise ValueError(f"--end-frame must be non-negative, got {end_frame}")
    if start_frame is not None and end_frame is not None and start_frame > end_frame:
        raise ValueError(
            f"--start-frame ({start_frame}) must not exceed --end-frame ({end_frame})"
        )

    time_rows = parse_delimited_rows(img_time_map, columns=2)
    time_entries: list[tuple[str, int]] = []
    time_by_name: dict[str, int] = {}
    for image_name, raw_timestamp in time_rows:
        if image_name in time_by_name:
            raise ValueError(f"Duplicate image name {image_name!r} in {img_time_map}")
        try:
            timestamp = int(raw_timestamp)
        except ValueError as exc:
            raise ValueError(
                f"Invalid timestamp {raw_timestamp!r} for image {image_name!r} in {img_time_map}"
            ) from exc
        time_entries.append((image_name, timestamp))
        time_by_name[image_name] = timestamp

    source_dir = Path(image_dir).expanduser().resolve()
    images_by_stem = index_images(source_dir, recursive=recursive)
    match_rows = parse_delimited_rows(img_match, columns=3)
    reference_mode = resolve_image_reference_mode(
        match_rows,
        time_by_name=time_by_name,
        time_entry_count=len(time_entries),
        source=Path(img_match).expanduser().resolve(),
    )
    matches: list[MatchedFrame] = []
    seen_frames: set[int] = set()
    for frame_id, image_reference, raw_diff_ms in match_rows:
        try:
            numeric_frame_id = int(frame_id)
        except ValueError as exc:
            raise ValueError(f"Frame id must be numeric in {img_match}, got {frame_id!r}") from exc
        if start_frame is not None and numeric_frame_id < start_frame:
            continue
        if end_frame is not None and numeric_frame_id > end_frame:
            continue
        if numeric_frame_id in seen_frames:
            raise ValueError(f"Duplicate frame id {frame_id!r} in {img_match}")
        seen_frames.add(numeric_frame_id)

        if reference_mode == "name":
            image_name = image_reference
            image_timestamp = time_by_name[image_name]
        else:
            image_index = int(image_reference)
            image_name, image_timestamp = time_entries[image_index]

        image_path = images_by_stem.get(image_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Image {image_name!r} selected for frame {frame_id} was not found in {source_dir}"
            )
        try:
            diff_ms = float(raw_diff_ms)
        except ValueError as exc:
            raise ValueError(
                f"Invalid diff_ms {raw_diff_ms!r} for frame {frame_id} in {img_match}"
            ) from exc
        matches.append(
            MatchedFrame(
                frame_id=frame_id,
                image_path=image_path,
                image_reference=image_reference,
                image_name=image_name,
                image_timestamp=image_timestamp,
                diff_ms=diff_ms,
            )
        )

    if not matches:
        range_description = f"[{start_frame or 0}, {end_frame if end_frame is not None else 'end'}]"
        raise ValueError(f"No matched frames selected from {img_match} in range {range_description}")
    return sorted(matches, key=lambda item: int(item.frame_id))


def resolve_image_reference_mode(
    match_rows: list[list[str]],
    *,
    time_by_name: dict[str, int],
    time_entry_count: int,
    source: Path,
) -> str:
    """Resolve the second imgMatch column once, avoiding per-row mixed modes."""
    references = [row[1] for row in match_rows]
    all_names = all(reference in time_by_name for reference in references)

    parsed_indices: list[int] = []
    all_indices = True
    for reference in references:
        try:
            image_index = int(reference)
        except ValueError:
            all_indices = False
            break
        if image_index < 0 or image_index >= time_entry_count:
            all_indices = False
            break
        parsed_indices.append(image_index)

    # An exact image-name mapping wins when every reference exists by name.
    # Otherwise all references must consistently be zero-based row indices.
    if all_names:
        return "name"
    if all_indices and len(parsed_indices) == len(references):
        return "index"

    invalid = [
        reference
        for reference in references
        if reference not in time_by_name
        and not (
            reference.lstrip("+").isdigit()
            and 0 <= int(reference) < time_entry_count
        )
    ]
    raise ValueError(
        f"The second column of {source} cannot be resolved consistently as image "
        f"names or zero-based imgTimeMap indices; invalid references={invalid[:5]!r}"
    )


def selected_frame_matches(args: argparse.Namespace) -> list[MatchedFrame] | None:
    mapping_values = (args.img_time_map, args.img_match)
    if any(mapping_values) and not all(mapping_values):
        raise ValueError("--img-time-map and --img-match must be provided together")
    if not all(mapping_values):
        if args.start_frame is not None or args.end_frame is not None:
            raise ValueError("--start-frame/--end-frame require --img-time-map and --img-match")
        return None
    return load_matched_frames(
        image_dir=args.image_dir,
        img_time_map=args.img_time_map,
        img_match=args.img_match,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recursive=False,
    )


def write_run_summary_csv(
    args: argparse.Namespace,
    *,
    detected_classes: set[str],
    frame_class_presence: tuple[frozenset[str], ...] | list[set[str]] | None = None,
    matches: list[MatchedFrame] | None,
) -> Path:
    out_dir = Path(args.out_dir).expanduser().resolve()
    if matches is not None:
        frame_ids = [match.frame_id for match in matches]
        start_timestamp: int | str = matches[0].image_timestamp
        end_timestamp: int | str = matches[-1].image_timestamp
    else:
        image_paths = collect_images(
            Path(args.image_dir).expanduser().resolve(),
            recursive=False,
        )
        frame_ids = [image_path.stem for image_path in image_paths]
        start_timestamp = ""
        end_timestamp = ""
    if not frame_ids:
        raise ValueError("Cannot write run summary because no input frames were selected")

    min_frames = int(getattr(args, "class_min_frames", 1))
    min_consecutive_frames = int(getattr(args, "class_min_consecutive_frames", 1))
    if min_frames <= 0:
        raise ValueError(f"--class-min-frames must be positive, got {min_frames}")
    if min_consecutive_frames <= 0:
        raise ValueError(
            "--class-min-consecutive-frames must be positive, "
            f"got {min_consecutive_frames}"
        )

    if frame_class_presence is None:
        frame_class_presence = [set(detected_classes)]
    class_stats = summarize_class_presence(frame_class_presence)
    filtered_classes = sorted(
        label
        for label in detected_classes
        if class_stats.get(label, {}).get("frame_count", 0) >= min_frames
        and class_stats.get(label, {}).get("max_consecutive_frames", 0)
        >= min_consecutive_frames
    )

    data_name = str(args.data_name).strip() if args.data_name is not None else ""
    if not data_name:
        data_name = "data_name"
    class_tags = load_csv_class_tags(
        getattr(args, "csv_tags_yaml", LEGACY_CSV_TAGS_CONFIG)
    )

    summary_path = out_dir / "sam3_run_summary.csv"
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "tags",
                "data_name",
                "start_frame",
                "end_frame",
                "start_timestamp",
                "end_timestamp",
                "frame_num",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tags": format_csv_tags(filtered_classes, class_tags=class_tags),
                "data_name": data_name,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "frame_num": len(frame_ids),
            }
        )
    temporary_path.replace(summary_path)
    return summary_path


def summarize_class_presence(
    frame_class_presence: tuple[frozenset[str], ...] | list[set[str]],
) -> dict[str, dict[str, int]]:
    labels = sorted({label for frame_classes in frame_class_presence for label in frame_classes})
    summary: dict[str, dict[str, int]] = {}
    for label in labels:
        frame_count = 0
        current_run = 0
        max_run = 0
        for frame_classes in frame_class_presence:
            if label in frame_classes:
                frame_count += 1
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        summary[label] = {
            "frame_count": frame_count,
            "max_consecutive_frames": max_run,
        }
    return summary


def run_sam3(
    args: argparse.Namespace,
    *,
    matches: list[MatchedFrame] | None,
) -> FolderDetectionSummary:
    return sam3_single_image_folder(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        prompt_config=args.prompt_config,
        classes_yaml=args.classes_yaml,
        sam3_root=args.sam3_root,
        sam3_model_path=args.sam3_model_path,
        min_score=args.min_score,
        label_min_scores=args.label_min_scores,
        recursive=False,
        max_images=None,
        max_prompts=None,
        use_fa3=args.use_fa3,
        overwrite=args.overwrite,
        validate=False,
        log_json=args.log_json,
        min_mask_size=args.min_mask_size,
        vehicle_orientation_checkpoint=args.vehicle_orientation_checkpoint,
        vehicle_orientation_device=args.vehicle_orientation_device,
        vehicle_orientation_min_confidence=args.vehicle_orientation_min_confidence,
        vehicle_orientation_min_margin=args.vehicle_orientation_min_margin,
        vehicle_orientation_nms_iou=args.vehicle_orientation_nms_iou,
        vehicle_prompt_label=args.vehicle_prompt_label,
        sam3_only=args.sam3_only,
        prompt_log=args.prompt_log,
        inference_precision=args.inference_precision,
        cache_visual_features=args.cache_visual_features,
        save_outputs=args.save_intermediate_outputs,
        frame_image_pairs=None
        if matches is None
        else [(match.frame_id, match.image_path) for match in matches],
    )


def prepare_csv_only_output(
    output_dir: str | Path,
    *,
    image_dir: str | Path,
    overwrite: bool,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    source = Path(image_dir).expanduser().resolve()
    if destination == source or destination in source.parents:
        raise ValueError(
            f"Output directory must not be the image directory or its parent: {destination}"
        )
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {destination}")

    existing = list(destination.iterdir()) if destination.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}; enable --overwrite"
        )
    if overwrite:
        for path in existing:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    resolve_sensor_config_paths(args)

    configure_gpu(args.gpu_id)
    matches = selected_frame_matches(args)
    assert matches is not None
    if args.save_intermediate_outputs:
        prepare_csv_only_output(
            args.out_dir,
            image_dir=args.image_dir,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "matched_frames": len(matches),
                "first_frame": matches[0].frame_id,
                "last_frame": matches[-1].frame_id,
                "first_image": matches[0].image_name,
                "last_image": matches[-1].image_name,
                "first_timestamp": matches[0].image_timestamp,
                "last_timestamp": matches[-1].image_timestamp,
                "sensor_version": args.sensor_version,
                "sensor_config_dir": args.sensor_config_dir,
            },
            indent=2,
        )
    )
    detection_summary = run_sam3(args, matches=matches)
    if not args.save_intermediate_outputs:
        prepare_csv_only_output(
            args.out_dir,
            image_dir=args.image_dir,
            overwrite=args.overwrite,
        )
    summary_path = write_run_summary_csv(
        args,
        detected_classes=set(detection_summary.detected_classes),
        frame_class_presence=detection_summary.frame_class_presence,
        matches=matches,
    )
    print(json.dumps({"run_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
