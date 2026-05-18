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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/classes.yaml")
    p.add_argument("--litept-config", default=None, help="Optional LitePT-native config file")
    p.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames")
    p.add_argument("--dry-run", action="store_true", help="Validate paths and frame discovery without importing LitePT")
    args = p.parse_args()

    if args.dry_run:
        plan = build_litept_inference_plan(
            litept_root=args.litept_root,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            input_format=args.input_format,
            validate_checkpoint=False,
            max_frames=args.max_frames,
        )
        print(json.dumps(plan.__dict__, ensure_ascii=False, indent=2))
        return

    try:
        results = run_litept_inference(
            litept_root=args.litept_root,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            input_format=args.input_format,
            config_path=args.config,
            litept_config=args.litept_config,
            max_frames=args.max_frames,
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


if __name__ == "__main__":
    main()
