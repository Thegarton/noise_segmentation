#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.class_config import load_semantic_classes  # noqa: E402
from autolabeler.teachers.sam3_video_adapter import (  # noqa: E402
    Sam3VideoFrame,
    class_names_from_mapping,
    copy_frame_image,
    labels_to_ids_from_prompt_config,
    load_camera_frame_manifest,
    load_sam3_video_result,
    run_sam3_video_teacher,
    sam3_video_to_semantic_mask,
    save_semantic_overlay,
    write_frames_json,
)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_camera_frame_manifest(args.camera_frame_manifest)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {args.max_frames}")
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError(f"No synced frames found in {args.camera_frame_manifest}")

    raw_dir = out_dir / "_sam3_video_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_to_run = [
        frame
        for frame in frames
        if args.overwrite or (not final_outputs_exist(out_dir, frame) and not resolve_raw_npz(raw_dir, frame).is_file())
    ]

    frames_json = out_dir / "sam3_video_frames.json"
    if frames_to_run:
        write_frames_json(frames_json, frames_to_run)
        sam3_extra_args = []
        if args.sam3_root is not None:
            sam3_extra_args.extend(["--sam3-root", args.sam3_root])
        if args.sam3_model_path is not None:
            sam3_extra_args.extend(["--sam3-model-path", args.sam3_model_path])
        sam3_extra_args.extend(args.sam3_wrapper_arg)
        run_sam3_video_teacher(
            video=args.video,
            prompt_config=args.prompt_config,
            frames_json=frames_json,
            output_dir=raw_dir,
            sam3_video_script=args.sam3_video_script,
            conda_env=args.sam3_conda_env,
            conda_prefix=args.sam3_conda_prefix,
            extra_args=sam3_extra_args,
        )
    elif not frames_json.is_file():
        write_frames_json(frames_json, frames)

    project_classes = load_semantic_classes(args.classes_yaml)
    label_to_id = labels_to_ids_from_prompt_config(args.classes_yaml, args.prompt_config)
    class_names = class_names_from_mapping(project_classes)

    exported = []
    for frame in frames:
        frame_out = out_dir / frame.frame_id
        semantic_path = frame_out / "semantic_mask.npy"
        confidence_path = frame_out / "confidence.npy"
        metadata_path = frame_out / "metadata.json"
        if final_outputs_exist(out_dir, frame) and not args.overwrite:
            exported.append({"frame_id": frame.frame_id, "status": "exists", "metadata": str(metadata_path)})
            continue

        raw_npz = resolve_raw_npz(raw_dir, frame)
        if not raw_npz.is_file():
            raise FileNotFoundError(
                f"SAM3 video wrapper did not write a result for frame_id={frame.frame_id}. "
                f"Expected {raw_dir / (frame.frame_id + '.npz')} or {raw_dir / frame.frame_id / 'sam3_video.npz'}"
            )

        result = load_sam3_video_result(raw_npz)
        if args.validate:
            load_sam3_video_result(raw_npz)
        semantic = sam3_video_to_semantic_mask(result, label_to_id=label_to_id, min_score=args.min_score)

        frame_out.mkdir(parents=True, exist_ok=True)
        final_npz = frame_out / "sam3_video.npz"
        shutil.copy2(raw_npz, final_npz)
        np.save(semantic_path, semantic.semantic_mask.astype(np.uint16, copy=False))
        np.save(confidence_path, semantic.confidence.astype(np.float32, copy=False))

        image_copy = copy_frame_image(frame.image_path, frame_out / "image.jpg")
        overlay_path = frame_out / "overlay.jpg"
        overlay_written = False
        overlay_error = None
        if frame.image_path is not None:
            try:
                overlay_written = save_semantic_overlay(frame.image_path, overlay_path, semantic_mask=semantic.semantic_mask)
            except ImportError as exc:
                overlay_error = str(exc)

        metadata = {
            "version": 1,
            "frame_id": frame.frame_id,
            "provenance": "sam3_video_teacher",
            "pseudo_label_version": "sam3_video_teacher_v1",
            "video": str(Path(args.video)),
            "video_frame_index": frame.video_frame_index,
            "lidar_timestamp_us": frame.lidar_timestamp_us,
            "video_timestamp_us": frame.video_timestamp_us,
            "delta_ms": frame.delta_ms,
            "image_path": frame.image_path,
            "image_copy": image_copy,
            "sam3_video_npz": str(final_npz),
            "semantic_mask": str(semantic_path),
            "confidence_mask": str(confidence_path),
            "overlay": str(overlay_path) if overlay_written else None,
            "overlay_error": overlay_error,
            "prompt_config": str(Path(args.prompt_config)),
            "classes_yaml": str(Path(args.classes_yaml)),
            "semantic_classes": project_classes,
            "class_names": class_names,
            "min_score": float(args.min_score),
            "accepted_instances": semantic.accepted_instances,
            "ignored_instances": semantic.ignored_instances,
            "unknown_labels": semantic.unknown_labels,
            "class_pixel_counts": semantic.class_pixel_counts,
            "sam3_conda_env": args.sam3_conda_env if args.sam3_conda_prefix is None else None,
            "sam3_conda_prefix": args.sam3_conda_prefix,
            "sam3_root": args.sam3_root,
            "sam3_model_path": args.sam3_model_path,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append({"frame_id": frame.frame_id, "status": "created", "metadata": str(metadata_path)})

    manifest = {
        "version": 1,
        "video": str(Path(args.video)),
        "camera_frame_manifest": str(Path(args.camera_frame_manifest)),
        "prompt_config": str(Path(args.prompt_config)),
        "classes_yaml": str(Path(args.classes_yaml)),
        "out_dir": str(out_dir),
        "frames_json": str(frames_json),
        "raw_dir": str(raw_dir),
        "synced_frames": len(frames),
        "max_frames": args.max_frames,
        "processed_frames": len(exported),
        "frames": exported,
    }
    manifest_path = out_dir / "sam3_video_teacher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(exported), "manifest": str(manifest_path)}, indent=2))


def resolve_raw_npz(raw_dir: Path, frame: Sam3VideoFrame) -> Path:
    flat = raw_dir / f"{frame.frame_id}.npz"
    if flat.is_file():
        return flat
    nested = raw_dir / frame.frame_id / "sam3_video.npz"
    if nested.is_file():
        return nested
    return flat


def final_outputs_exist(out_dir: Path, frame: Sam3VideoFrame) -> bool:
    frame_out = out_dir / frame.frame_id
    return (
        (frame_out / "sam3_video.npz").is_file()
        and (frame_out / "semantic_mask.npy").is_file()
        and (frame_out / "confidence.npy").is_file()
        and (frame_out / "metadata.json").is_file()
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAM3 as a video teacher and export synced camera semantic masks.")
    p.add_argument("--video", required=True)
    p.add_argument("--camera-frame-manifest", required=True)
    p.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    p.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sam3-video-script", required=True)
    p.add_argument("--sam3-conda-env", default="sam3")
    p.add_argument("--sam3-conda-prefix", default=None)
    p.add_argument("--sam3-root", default=None, help="Optional path to the SAM3 repository, passed to the wrapper.")
    p.add_argument("--sam3-model-path", default=None, help="Optional local facebook/sam3.1 model directory, passed to the wrapper.")
    p.add_argument("--sam3-wrapper-arg", action="append", default=[], help="Extra argument passed to the SAM3 video wrapper.")
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N synced frames from camera_frame_manifest.json.")
    p.add_argument("--min-score", type=float, default=0.7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
