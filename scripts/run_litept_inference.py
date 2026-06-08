#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autolabeler.export.semantic_mask_exporter import export_semantic_segmentation_result
from autolabeler.teachers.litept_adapter import (
    LitePTUnavailableError,
    build_litept_inference_plan,
    run_litept_inference,
)

def main() -> None:
    p = argparse.ArgumentParser(description="Run LitePT point-wise semantic segmentation over organized LiDAR frames")
    p.add_argument("--litept-root", required=True, help="Path to the cloned LitePT repository")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--litept-dataset", choices=["nuscenes", "waymo", "custom"], default="nuscenes")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override. Required for custom; otherwise defaults to pth/<dataset>/model_best.pth",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/classes.yaml")
    p.add_argument(
        "--litept-config",
        default=None,
        help="LitePT-native config override. Required for custom; otherwise uses the dataset preset.",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    p.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:1, or cpu")
    p.add_argument("--ins-path", default=None, help="Optional INS file path. Defaults to <input-dir>/ins when it exists.")
    p.add_argument("--skip-pose-export", action="store_true", help="Do not match INS ego poses or write pose.txt files.")
    p.add_argument(
        "--force-torch-pointrope",
        action="store_true",
        help="Use LitePT's pure PyTorch PointROPE fallback instead of the compiled CUDA extension",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate paths and frame discovery without importing LitePT")
    args = p.parse_args()

    if args.dry_run:
        plan = build_litept_inference_plan(
            litept_root=args.litept_root,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            input_format=args.input_format,
            litept_dataset=args.litept_dataset,
            litept_config=args.litept_config,
            validate_checkpoint=False,
            max_frames=args.max_frames,
        )
        print(json.dumps(plan.__dict__, ensure_ascii=False, indent=2))
        return

    ins_path = _resolve_ins_path(args.input_dir, args.ins_path, args.skip_pose_export)
    try:
        results = run_litept_inference(
            litept_root=args.litept_root,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            input_format=args.input_format,
            config_path=args.config,
            litept_dataset=args.litept_dataset,
            litept_config=args.litept_config,
            max_frames=args.max_frames,
            device=args.device,
            force_torch_pointrope=args.force_torch_pointrope,
            ins_path=ins_path,
            skip_pose_export=args.skip_pose_export,
        )
    except LitePTUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for result in results:
        export_semantic_segmentation_result(args.output_dir, result)
    print(
        json.dumps(
            {"exported_frames": len(results), "output_dir": str(Path(args.output_dir).resolve())},
            ensure_ascii=False,
            indent=2,
        )
    )


def _resolve_ins_path(input_dir: str, ins_path: str | None, skip_pose_export: bool) -> str | None:
    if skip_pose_export:
        return None
    if ins_path:
        return ins_path
    candidate = Path(input_dir) / "ins"
    return str(candidate) if candidate.exists() else None


if __name__ == "__main__":
    main()
