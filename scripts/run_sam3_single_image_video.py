#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_IMAGE_SCRIPT = REPO_ROOT / "scripts" / "run_sam3_single_image_folder.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: str
    video_frame_index: int
    image_path: Path


@dataclass(frozen=True)
class ExtractionResult:
    video_path: Path
    frames_dir: Path
    start_frame: int
    frame_step: int
    max_frames: int | None
    source_frame_count: int | None
    source_fps: float | None
    frames: list[ExtractedFrame]


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else out_dir / "_video_frames"
    single_image_script = Path(args.single_image_script).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not single_image_script.is_file():
        raise FileNotFoundError(f"SAM3 single-image script does not exist: {single_image_script}")
    validate_args(args)

    out_dir.mkdir(parents=True, exist_ok=True)
    extraction = extract_video_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    manifest_path = write_extraction_manifest(
        extraction,
        out_dir=out_dir,
        single_image_script=single_image_script,
        args=args,
    )

    if args.extract_only:
        print(json.dumps({"frames": len(extraction.frames), "manifest": str(manifest_path)}, indent=2))
        return

    if args.per_frame_process:
        commands = run_per_frame(
            args=args,
            extraction=extraction,
            out_dir=out_dir,
            single_image_script=single_image_script,
        )
    else:
        command = build_single_image_command(
            args=args,
            image_dir=frames_dir,
            out_dir=out_dir,
            single_image_script=single_image_script,
            max_images=None,
        )
        if args.dry_run:
            commands = [command]
        else:
            print_command(command)
            subprocess.run(command, check=True, env=build_subprocess_env())
            commands = [command]

    run_manifest = {
        "version": 1,
        "video": str(video_path),
        "frames_dir": str(frames_dir),
        "out_dir": str(out_dir),
        "extraction_manifest": str(manifest_path),
        "per_frame_process": bool(args.per_frame_process),
        "dry_run": bool(args.dry_run),
        "commands": commands,
    }
    run_manifest_path = out_dir / "sam3_single_image_video_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(extraction.frames), "manifest": str(run_manifest_path)}, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.sam3_conda_prefix and args.sam3_conda_env:
        raise ValueError("Use only one of --sam3-conda-prefix or --sam3-conda-env")
    if args.start_frame < 0:
        raise ValueError(f"--start-frame must be non-negative, got {args.start_frame}")
    if args.frame_step <= 0:
        raise ValueError(f"--frame-step must be positive, got {args.frame_step}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError(f"--max-frames must be positive, got {args.max_frames}")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError(f"--jpeg-quality must be in [1,100], got {args.jpeg_quality}")


def extract_video_frames(
    *,
    video_path: Path,
    frames_dir: Path,
    start_frame: int,
    max_frames: int | None,
    frame_step: int,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> ExtractionResult:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing_images = list_image_files(frames_dir)
    if existing_images and not overwrite:
        raise FileExistsError(f"Frame directory already contains images: {frames_dir}. Use --overwrite to rebuild it.")
    if overwrite:
        clean_frame_dir(frames_dir)

    cv2 = import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_frame_count = read_int_capture_property(capture, cv2, "CAP_PROP_FRAME_COUNT")
    source_fps = read_float_capture_property(capture, cv2, "CAP_PROP_FPS")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    frames: list[ExtractedFrame] = []
    frame_index = int(start_frame)
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if (frame_index - start_frame) % frame_step == 0:
                frame_id = f"{frame_index:06d}"
                frame_path = frames_dir / f"{frame_id}.{image_ext}"
                write_frame(cv2, frame_path=frame_path, frame_bgr=frame_bgr, image_ext=image_ext, jpeg_quality=jpeg_quality)
                frames.append(ExtractedFrame(frame_id=frame_id, video_frame_index=frame_index, image_path=frame_path))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return ExtractionResult(
        video_path=video_path,
        frames_dir=frames_dir,
        start_frame=start_frame,
        frame_step=frame_step,
        max_frames=max_frames,
        source_frame_count=source_frame_count,
        source_fps=source_fps,
        frames=frames,
    )


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("OpenCV is required to split video frames. Install opencv-python in this env.") from exc
    return cv2


def list_image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def clean_frame_dir(path: Path) -> None:
    for item in list_image_files(path):
        item.unlink()
    manifest = path / "video_frames_manifest.json"
    if manifest.exists():
        manifest.unlink()


def read_int_capture_property(capture: Any, cv2: Any, name: str) -> int | None:
    value = read_float_capture_property(capture, cv2, name)
    if value is None or value <= 0:
        return None
    return int(round(value))


def read_float_capture_property(capture: Any, cv2: Any, name: str) -> float | None:
    prop = getattr(cv2, name, None)
    if prop is None:
        return None
    try:
        value = float(capture.get(prop))
    except Exception:
        return None
    if not (value > 0.0):
        return None
    return value


def write_frame(cv2: Any, *, frame_path: Path, frame_bgr: Any, image_ext: str, jpeg_quality: int) -> None:
    params: list[int] = []
    if image_ext in {"jpg", "jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok = bool(cv2.imwrite(str(frame_path), frame_bgr, params))
    if not ok:
        raise RuntimeError(f"Could not write extracted frame: {frame_path}")


def write_extraction_manifest(
    extraction: ExtractionResult,
    *,
    out_dir: Path,
    single_image_script: Path,
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "version": 1,
        "video": str(extraction.video_path),
        "frames_dir": str(extraction.frames_dir),
        "single_image_script": str(single_image_script),
        "start_frame": int(extraction.start_frame),
        "frame_step": int(extraction.frame_step),
        "max_frames": extraction.max_frames,
        "source_frame_count": extraction.source_frame_count,
        "source_fps": extraction.source_fps,
        "image_ext": args.image_ext,
        "jpeg_quality": int(args.jpeg_quality),
        "frames": [
            {
                "frame_id": item.frame_id,
                "video_frame_index": int(item.video_frame_index),
                "image_path": str(item.image_path),
            }
            for item in extraction.frames
        ],
    }
    path = extraction.frames_dir / "video_frames_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "video_frames_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_single_image_command(
    *,
    args: argparse.Namespace,
    image_dir: Path,
    out_dir: Path,
    single_image_script: Path,
    max_images: int | None,
) -> list[str]:
    command = build_python_launcher(args)
    command.extend(
        [
            str(single_image_script),
            "--image-dir",
            str(image_dir),
            "--out-dir",
            str(out_dir),
            "--prompt-config",
            str(Path(args.prompt_config).expanduser()),
            "--classes-yaml",
            str(Path(args.classes_yaml).expanduser()),
            "--min-score",
            str(args.min_score),
            "--min-mask-size",
            str(args.min_mask_size),
        ]
    )
    optional_path_args = [
        ("--sam3-root", args.sam3_root),
        ("--sam3-model-path", args.sam3_model_path),
        ("--projection-dir", args.projection_dir),
        ("--label-min-scores", args.label_min_scores),
        ("--vehicle-orientation-checkpoint", getattr(args, "vehicle_orientation_checkpoint", None)),
    ]
    for flag, value in optional_path_args:
        if value:
            command.extend([flag, str(Path(value).expanduser())])
    for suffix in args.projection_stem_suffix:
        command.extend(["--projection-stem-suffix", suffix])
    if args.require_projection:
        command.append("--require-projection")
    if args.use_fa3:
        command.append("--use-fa3")
    if args.prompt_log:
        command.append("--prompt-log")
    if args.overwrite:
        command.append("--overwrite")
    if args.validate:
        command.append("--validate")
    if getattr(args, "sam3_only", False):
        command.append("--sam3-only")
    if max_images is not None:
        command.extend(["--max-images", str(max_images)])
    if args.max_prompts is not None:
        command.extend(["--max-prompts", str(args.max_prompts)])
    if getattr(args, "vehicle_orientation_checkpoint", None):
        command.extend(
            [
                "--vehicle-prompt-label",
                str(getattr(args, "vehicle_prompt_label", "vehicle")),
                "--vehicle-orientation-device",
                str(getattr(args, "vehicle_orientation_device", "auto")),
                "--vehicle-orientation-min-confidence",
                str(getattr(args, "vehicle_orientation_min_confidence", 0.70)),
                "--vehicle-orientation-min-margin",
                str(getattr(args, "vehicle_orientation_min_margin", 0.10)),
                "--vehicle-orientation-nms-iou",
                str(getattr(args, "vehicle_orientation_nms_iou", 0.80)),
            ]
        )
    return command


def build_python_launcher(args: argparse.Namespace) -> list[str]:
    if args.sam3_conda_prefix:
        return ["conda", "run", "-p", str(Path(args.sam3_conda_prefix).expanduser()), "python"]
    if args.sam3_conda_env:
        return ["conda", "run", "-n", str(args.sam3_conda_env), "python"]
    return [sys.executable]


def run_per_frame(
    *,
    args: argparse.Namespace,
    extraction: ExtractionResult,
    out_dir: Path,
    single_image_script: Path,
) -> list[list[str]]:
    scratch_root = out_dir / "_single_frame_inputs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    for index, frame in enumerate(extraction.frames, start=1):
        single_dir = scratch_root / frame.frame_id
        single_dir.mkdir(parents=True, exist_ok=True)
        linked_frame = single_dir / frame.image_path.name
        if linked_frame.exists() or linked_frame.is_symlink():
            linked_frame.unlink()
        try:
            os.symlink(frame.image_path, linked_frame)
        except OSError:
            import shutil  # noqa: WPS433

            shutil.copy2(frame.image_path, linked_frame)

        command = build_single_image_command(
            args=args,
            image_dir=single_dir,
            out_dir=out_dir,
            single_image_script=single_image_script,
            max_images=1,
        )
        commands.append(command)
        if args.dry_run:
            continue
        print(f"[{index:04d}/{len(extraction.frames):04d}] {frame.frame_id}", file=sys.stderr, flush=True)
        print_command(command)
        subprocess.run(command, check=True, env=build_subprocess_env())
    return commands


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    return env


def print_command(command: list[str]) -> None:
    print(" ".join(shell_quote(part) for part in command), file=sys.stderr, flush=True)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/.:=+"
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a video into images and run run_sam3_single_image_folder.py on the extracted frames.",
    )
    parser.add_argument("--video", required=True, help="Input video, for example .avi or .mp4.")
    parser.add_argument("--out-dir", required=True, help="SAM3 output directory. Per-frame outputs are written here.")
    parser.add_argument("--frames-dir", default=None, help="Where extracted frames are stored. Default: <out-dir>/_video_frames.")
    parser.add_argument(
        "--single-image-script",
        default=str(DEFAULT_SINGLE_IMAGE_SCRIPT),
        help="Path to run_sam3_single_image_folder.py.",
    )
    parser.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes_pointwise_v1.yaml"))
    parser.add_argument("--sam3-root", default=None)
    parser.add_argument("--sam3-model-path", default=None)
    parser.add_argument(
        "--sam3-only",
        action="store_true",
        help="Disable EfficientNet and use SAM3 prompt labels as semantic classes directly.",
    )
    parser.add_argument("--vehicle-orientation-checkpoint", default=None)
    parser.add_argument("--vehicle-prompt-label", default="vehicle")
    parser.add_argument("--vehicle-orientation-device", default="auto")
    parser.add_argument("--vehicle-orientation-min-confidence", type=float, default=0.70)
    parser.add_argument("--vehicle-orientation-min-margin", type=float, default=0.10)
    parser.add_argument("--vehicle-orientation-nms-iou", type=float, default=0.80)
    parser.add_argument("--sam3-conda-env", default=None, help="Run SAM3 script through `conda run -n ENV python ...`.")
    parser.add_argument("--sam3-conda-prefix", default=None, help="Run SAM3 script through `conda run -p PREFIX python ...`.")
    parser.add_argument("--projection-dir", default=None, help="Optional projection image directory passed through.")
    parser.add_argument(
        "--projection-stem-suffix",
        action="append",
        default=[],
        help="Optional suffix stripped by the single-image script when matching projection images.",
    )
    parser.add_argument("--require-projection", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--min-mask-size", type=int, default=30)
    parser.add_argument(
        "--label-min-scores",
        default=None,
        help="Optional per-label detection threshold YAML/JSON passed to the single-image runner.",
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--image-ext", choices=["jpg", "jpeg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--prompt-log", action="store_true")
    parser.add_argument("--per-frame-process", action="store_true", help="Run a separate SAM3 subprocess for each frame.")
    parser.add_argument("--extract-only", action="store_true", help="Only split video frames; do not run SAM3.")
    parser.add_argument("--dry-run", action="store_true", help="Build manifests and print commands without running SAM3.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
