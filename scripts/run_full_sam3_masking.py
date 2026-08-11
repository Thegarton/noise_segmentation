#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_sam3_single_image_folder import collect_images, sam3_single_image_folder


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    paths = collect_images(Path(args.image_dir).expanduser().resolve(), recursive=args.recursive)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError(f"--max-images must be positive, got {args.max_images}")
        paths = paths[: args.max_images]
    if not paths:
        raise ValueError(f"No images found in {args.image_dir}")
    return len(paths)


def run_sam3_worker(args: argparse.Namespace) -> None:
    worker_index = 0 if args.worker_index is None else int(args.worker_index)
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
    frames.sort(key=lambda item: (str(item.get("image", "")), str(item.get("frame_id", ""))))
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
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids:
        num_workers = launch_gpu_workers(args, gpu_ids, original_argv)
        manifest_path = merge_worker_manifests(args.out_dir, num_workers=num_workers)
        print(json.dumps({"workers": num_workers, "manifest": str(manifest_path)}, indent=2))
    else:
        run_sam3_worker(args)
    collect_review_images(args.out_dir)


if __name__ == "__main__":
    main()
