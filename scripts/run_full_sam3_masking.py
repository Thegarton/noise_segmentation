#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from run_sam3_single_image_folder import collect_images, sam3_single_image_folder


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        description="Prepare HL320 images and run SAM3.1 on one selected GPU."
    )
    parser.add_argument("-i", "--folder_path", help="Path to the folder with source .bin files.")
    parser.add_argument("-I", "--image-folder-path", help="Path used by image-only conversion.")
    parser.add_argument("-o", "--output_folder_name", help="Output directory for converted data.")
    parser.add_argument("--skip-conversion", action="store_true")
    parser.add_argument("--without-bin", action="store_true")

    parser.add_argument("--image-dir", required=True, help="Directory with prepared camera images.")
    parser.add_argument("--out-dir", required=True, help="Output directory for per-image SAM3 results.")
    parser.add_argument(
        "--data-name",
        default=None,
        help="Dataset name written to sam3_run_summary.csv. Default: data_name.",
    )
    parser.add_argument(
        "--img-time-map",
        default=None,
        help=(
            "TXT with '<image-name>>><timestamp>' rows. The zero-based row index may be "
            "referenced by the second column of --img-match."
        ),
    )
    parser.add_argument(
        "--img-match",
        default=None,
        help="TXT with '<frame-id>>><image-index-or-name>>><diff-ms>' rows.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame id from --img-match to process (inclusive).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame id from --img-match to process (inclusive).",
    )
    parser.add_argument(
        "--prompt-config",
        default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"),
    )
    parser.add_argument(
        "--classes-yaml",
        default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"),
    )
    parser.add_argument("--sam3-root", default=str(REPO_ROOT / "../sam3"))
    parser.add_argument("--sam3-model-path", default=str(REPO_ROOT / "../sam3/sam3.1"))
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--projection-dir", default=None)
    parser.add_argument("--require-projection", action="store_true")
    parser.add_argument("--label-min-scores", default=None)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--prompt-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--min-mask-size", type=int, default=30)
    parser.add_argument("--defisheye", action="store_true")

    parser.add_argument("--vehicle-orientation-checkpoint", default=None)
    parser.add_argument("--vehicle-prompt-label", default="vehicle")
    parser.add_argument("--vehicle-orientation-device", default="auto")
    parser.add_argument("--vehicle-orientation-min-confidence", type=float, default=0.75)
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
        "--gpu-id",
        type=int,
        default=None,
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
        action="store_true",
        help="Compute the single-image visual backbone once and reuse it for all text prompts.",
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
        if not line or line.startswith("#"):
            continue
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


def index_images(image_dir: Path, *, recursive: bool) -> dict[str, Path]:
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
    recursive: bool,
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
    matches: list[MatchedFrame] = []
    seen_frames: set[str] = set()
    for frame_id, image_reference, raw_diff_ms in match_rows:
        try:
            numeric_frame_id = int(frame_id)
        except ValueError as exc:
            raise ValueError(f"Frame id must be numeric in {img_match}, got {frame_id!r}") from exc
        if start_frame is not None and numeric_frame_id < start_frame:
            continue
        if end_frame is not None and numeric_frame_id > end_frame:
            continue
        if frame_id in seen_frames:
            raise ValueError(f"Duplicate frame id {frame_id!r} in {img_match}")
        seen_frames.add(frame_id)

        if image_reference in time_by_name:
            image_name = image_reference
            image_timestamp = time_by_name[image_name]
        else:
            try:
                image_index = int(image_reference)
            except ValueError as exc:
                raise ValueError(
                    f"Image reference {image_reference!r} for frame {frame_id} is neither "
                    "an image name nor a zero-based numeric index"
                ) from exc
            if image_index < 0 or image_index >= len(time_entries):
                raise IndexError(
                    f"Image index {image_index} for frame {frame_id} is outside "
                    f"--img-time-map range [0, {len(time_entries)})"
                )
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
    return matches


def selected_frame_matches(args: argparse.Namespace) -> list[MatchedFrame] | None:
    mapping_values = (args.img_time_map, args.img_match)
    if any(mapping_values) and not all(mapping_values):
        raise ValueError("--img-time-map and --img-match must be provided together")
    if not all(mapping_values):
        if args.start_frame is not None or args.end_frame is not None:
            raise ValueError("--start-frame/--end-frame require --img-time-map and --img-match")
        return None
    matches = load_matched_frames(
        image_dir=args.image_dir,
        img_time_map=args.img_time_map,
        img_match=args.img_match,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        recursive=args.recursive,
    )
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        matches = matches[: args.max_images]
    return matches


def write_match_manifest(args: argparse.Namespace, matches: list[MatchedFrame]) -> Path:
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "image_dir": str(Path(args.image_dir).expanduser().resolve()),
        "img_time_map": str(Path(args.img_time_map).expanduser().resolve()),
        "img_match": str(Path(args.img_match).expanduser().resolve()),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "frames": [
            {
                **asdict(match),
                "image_path": str(match.image_path),
            }
            for match in matches
        ],
    }
    path = output_dir / "sam3_image_match_manifest.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def write_run_summary_csv(
    args: argparse.Namespace,
    *,
    detected_classes: set[str],
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
            recursive=args.recursive,
        )
        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]
        frame_ids = [image_path.stem for image_path in image_paths]
        start_timestamp = ""
        end_timestamp = ""
    if not frame_ids:
        raise ValueError("Cannot write run summary because no input frames were selected")

    data_name = str(args.data_name).strip() if args.data_name is not None else ""
    if not data_name:
        data_name = "data_name"

    summary_path = out_dir / "sam3_run_summary.csv"
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "classes",
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
                "classes": json.dumps(sorted(detected_classes), ensure_ascii=False),
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


def run_conversion(args: argparse.Namespace) -> None:
    if args.skip_conversion:
        return
    try:
        from autolabeler.data.HL320_cameracalibration import (  # type: ignore
            save_converted_data,
            save_converted_data_without_bin,
        )
    except ImportError as exc:
        raise ImportError(
            "HL320 conversion helpers are unavailable. Use --skip-conversion when --image-dir "
            "already contains prepared images."
        ) from exc

    if args.without_bin:
        if not args.image_folder_path:
            raise ValueError("--image-folder-path is required with --without-bin")
        save_converted_data_without_bin(args.image_folder_path, args.defisheye)
    else:
        if not args.folder_path or not args.output_folder_name:
            raise ValueError("--folder_path and --output_folder_name are required for BIN conversion")
        save_converted_data(args.folder_path, args.output_folder_name)


def run_sam3(args: argparse.Namespace, *, matches: list[MatchedFrame] | None) -> set[str]:
    return sam3_single_image_folder(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        prompt_config=args.prompt_config,
        classes_yaml=args.classes_yaml,
        sam3_root=args.sam3_root,
        sam3_model_path=args.sam3_model_path,
        min_score=args.min_score,
        projection_dir=args.projection_dir,
        label_min_scores=args.label_min_scores,
        recursive=args.recursive,
        max_images=args.max_images,
        max_prompts=args.max_prompts,
        use_fa3=args.use_fa3,
        overwrite=args.overwrite,
        require_projection=args.require_projection,
        validate=args.validate,
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
        frame_image_pairs=None
        if matches is None
        else [(match.frame_id, match.image_path) for match in matches],
    )


def collect_review_images(out_dir: str | Path) -> None:
    try:
        from autolabeler.data.collect_image_in_one_folder import (  # type: ignore
            collect_combine_image_in_one_folder,
            collect_overlay_image_in_one_folder,
        )
    except ImportError:
        print(
            "Review-image collection helpers are unavailable; per-frame outputs are complete.",
            file=sys.stderr,
            flush=True,
        )
        return
    collect_overlay_image_in_one_folder(out_dir)
    collect_combine_image_in_one_folder(out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    configure_gpu(args.gpu_id)
    run_conversion(args)
    matches = selected_frame_matches(args)
    if matches is not None:
        match_manifest = write_match_manifest(args, matches)
        print(
            json.dumps(
                {
                    "matched_frames": len(matches),
                    "first_frame": matches[0].frame_id,
                    "last_frame": matches[-1].frame_id,
                    "match_manifest": str(match_manifest),
                },
                indent=2,
            )
        )
    detected_classes = run_sam3(args, matches=matches)
    summary_path = write_run_summary_csv(
        args,
        detected_classes=detected_classes,
        matches=matches,
    )
    print(json.dumps({"run_summary": str(summary_path)}, indent=2))
    collect_review_images(args.out_dir)


if __name__ == "__main__":
    main()
