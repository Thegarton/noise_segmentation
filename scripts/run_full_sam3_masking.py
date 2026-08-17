#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
        description="Prepare HL320 images and run SAM3.1 on one or more GPUs."
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
        "--gpu-ids",
        default=None,
        help=(
            "Comma-separated physical GPU ids, for example 0,1,2,3. Each GPU gets an "
            "independent SAM3 process and a disjoint image shard."
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

    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def parse_gpu_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    gpu_ids = [token.strip() for token in value.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one id")
    if any(not token.isdigit() for token in gpu_ids):
        raise ValueError(f"--gpu-ids must contain non-negative integers, got {value!r}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicate ids: {value!r}")
    return gpu_ids


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


def collect_run_classes(out_dir: Path, *, frame_ids: list[str]) -> list[str]:
    classes: list[str] = []
    seen: set[str] = set()
    for frame_id in frame_ids:
        classes_log_path = out_dir / frame_id / "classes_log.json"
        if not classes_log_path.is_file():
            continue
        payload = json.loads(classes_log_path.read_text(encoding="utf-8"))
        class_list = payload.get("class_list", [])
        if not isinstance(class_list, list):
            raise ValueError(f"class_list must be a list in {classes_log_path}")
        for raw_class_name in class_list:
            class_name = str(raw_class_name)
            if class_name not in seen:
                seen.add(class_name)
                classes.append(class_name)
    return classes


def write_run_summary_csv(
    args: argparse.Namespace,
    *,
    matches: list[MatchedFrame] | None,
) -> Path:
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = out_dir / "sam3_single_image_folder_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing final SAM3 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_ids = [str(frame["frame_id"]) for frame in manifest.get("frames", [])]
    if not frame_ids:
        raise ValueError(f"Final SAM3 manifest contains no frames: {manifest_path}")

    classes = collect_run_classes(out_dir, frame_ids=frame_ids)
    matches_by_frame = {} if matches is None else {match.frame_id: match for match in matches}
    first_match = matches_by_frame.get(frame_ids[0])
    last_match = matches_by_frame.get(frame_ids[-1])
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
                "classes": json.dumps(classes, ensure_ascii=False),
                "data_name": data_name,
                "start_frame": frame_ids[0],
                "end_frame": frame_ids[-1],
                "start_timestamp": "" if first_match is None else first_match.image_timestamp,
                "end_timestamp": "" if last_match is None else last_match.image_timestamp,
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


def count_selected_images(args: argparse.Namespace) -> int:
    matches = selected_frame_matches(args)
    paths = (
        [match.image_path for match in matches]
        if matches is not None
        else collect_images(Path(args.image_dir).expanduser().resolve(), recursive=args.recursive)
    )
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        paths = paths[: args.max_images]
    if not paths:
        raise ValueError(f"No images found in {args.image_dir}")
    return len(paths)


def run_sam3_worker(args: argparse.Namespace) -> None:
    worker_index = 0 if args.worker_index is None else int(args.worker_index)
    matches = selected_frame_matches(args)
    sam3_single_image_folder(
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
        worker_index=worker_index,
        num_workers=args.num_workers,
        frame_image_pairs=None
        if matches is None
        else [(match.frame_id, match.image_path) for match in matches],
    )


def build_worker_command(
    original_argv: list[str],
    *,
    worker_index: int,
    num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *original_argv,
        "--skip-conversion",
        "--worker-index",
        str(worker_index),
        "--num-workers",
        str(num_workers),
    ]


def launch_gpu_workers(args: argparse.Namespace, gpu_ids: list[str], original_argv: list[str]) -> int:
    image_count = count_selected_images(args)
    num_workers = min(len(gpu_ids), image_count)
    active_gpu_ids = gpu_ids[:num_workers]
    if num_workers < len(gpu_ids):
        print(
            f"Using {num_workers} of {len(gpu_ids)} requested GPUs for {image_count} images",
            file=sys.stderr,
            flush=True,
        )

    processes: list[tuple[int, str, subprocess.Popen[Any]]] = []
    for worker_index, gpu_id in enumerate(active_gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["SAM3_WORKER_INDEX"] = str(worker_index)
        env["SAM3_NUM_WORKERS"] = str(num_workers)
        env["SAM3_PHYSICAL_GPU_ID"] = gpu_id
        command = build_worker_command(
            original_argv,
            worker_index=worker_index,
            num_workers=num_workers,
        )
        print(
            f"Starting SAM3 worker {worker_index + 1}/{num_workers} on physical GPU {gpu_id}",
            file=sys.stderr,
            flush=True,
        )
        processes.append((worker_index, gpu_id, subprocess.Popen(command, env=env)))

    unfinished = list(processes)
    failure: tuple[int, str, int] | None = None
    while unfinished and failure is None:
        for item in list(unfinished):
            worker_index, gpu_id, process = item
            return_code = process.poll()
            if return_code is None:
                continue
            unfinished.remove(item)
            if return_code != 0:
                failure = (worker_index, gpu_id, return_code)
                break
        if unfinished and failure is None:
            time.sleep(0.2)

    if failure is not None:
        for _, _, process in unfinished:
            process.terminate()
        for _, _, process in unfinished:
            process.wait()
        worker_index, gpu_id, return_code = failure
        raise RuntimeError(
            f"SAM3 worker {worker_index} on GPU {gpu_id} exited with code {return_code}"
        )
    return num_workers


def merge_worker_manifests(out_dir: str | Path, *, num_workers: int) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    manifests: list[dict[str, Any]] = []
    worker_paths: list[Path] = []
    for worker_index in range(num_workers):
        path = output_dir / f"sam3_single_image_folder_manifest.worker_{worker_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SAM3 worker manifest: {path}")
        worker_paths.append(path)
        manifests.append(json.loads(path.read_text(encoding="utf-8")))

    combined = dict(manifests[0])
    frames = [frame for manifest in manifests for frame in manifest.get("frames", [])]
    frames.sort(key=lambda item: (str(item.get("frame_id", "")), str(item.get("image", ""))))
    combined["images"] = len(frames)
    combined["frames"] = frames
    combined.pop("worker", None)
    combined["workers"] = [
        {
            "index": index,
            "manifest": str(path),
            "images": int(manifest.get("images", 0)),
            "physical_gpu_id": manifest.get("worker", {}).get("physical_gpu_id"),
            "sam3_runtime": manifest.get("sam3_runtime"),
        }
        for index, (path, manifest) in enumerate(zip(worker_paths, manifests))
    ]
    combined["multi_gpu"] = True
    combined["num_workers"] = num_workers
    runtime = dict(manifests[0].get("sam3_runtime") or {})
    runtime["visual_feature_cache_hits"] = sum(
        int((manifest.get("sam3_runtime") or {}).get("visual_feature_cache_hits", 0))
        for manifest in manifests
    )
    runtime["visual_feature_cache_misses"] = sum(
        int((manifest.get("sam3_runtime") or {}).get("visual_feature_cache_misses", 0))
        for manifest in manifests
    )
    combined["sam3_runtime"] = runtime

    final_path = output_dir / "sam3_single_image_folder_manifest.json"
    temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(final_path)
    return final_path


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
    original_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(original_argv)

    if args.worker_index is not None:
        run_sam3_worker(args)
        return

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
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids:
        num_workers = launch_gpu_workers(args, gpu_ids, original_argv)
        manifest_path = merge_worker_manifests(args.out_dir, num_workers=num_workers)
        print(json.dumps({"workers": num_workers, "manifest": str(manifest_path)}, indent=2))
    else:
        run_sam3_worker(args)
    collect_review_images(args.out_dir)
    summary_path = write_run_summary_csv(args, matches=matches)
    print(json.dumps({"run_summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
