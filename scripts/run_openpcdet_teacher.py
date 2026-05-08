#!/usr/bin/env python3
from __future__ import annotations

import argparse

from autolabeler.teachers.openpcdet_adapter import prepare_openpcdet_points, run_openpcdet_inference


def main() -> None:
    p = argparse.ArgumentParser(description="Run OpenPCDet CenterPoint/PointPillars as an offline actor teacher")
    p.add_argument("--input-dir", default="./data")
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--prepared-dir", default="./out/openpcdet")
    p.add_argument("--output", default="./out/openpcdet_predictions.jsonl")
    p.add_argument("--openpcdet-root", required=True, help="Path to cloned open-mmlab/OpenPCDet repository")
    p.add_argument("--cfg-file", required=True, help="OpenPCDet model config, e.g. cfgs/nuscenes_models/cbgs_centerpoint.yaml")
    p.add_argument("--ckpt", required=True, help="OpenPCDet pretrained checkpoint .pth")
    p.add_argument("--score-threshold", type=float, default=0.15)
    args = p.parse_args()

    prepared = prepare_openpcdet_points(
        input_dir=args.input_dir,
        output_dir=args.prepared_dir,
        input_format=args.input_format,
    )
    run_openpcdet_inference(
        openpcdet_root=args.openpcdet_root,
        cfg_file=args.cfg_file,
        ckpt=args.ckpt,
        prepared_frames=prepared,
        output_jsonl=args.output,
        score_threshold=args.score_threshold,
    )
    print(f"teacher_predictions={args.output} prepared_frames={len(prepared)}")


if __name__ == "__main__":
    main()
