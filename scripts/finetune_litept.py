#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from autolabeler.teachers.litept_finetune import (
    build_finetune_plan,
    build_training_command,
    prepare_finetune_run,
    run_training,
)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.resume and args.prepare_only:
        raise SystemExit("--resume and --prepare-only cannot be used together")

    try:
        plan = build_finetune_plan(
            litept_root=args.litept_root,
            export_dir=args.export_dir,
            labeler_dir=args.labeler_dir,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            val_ratio=args.val_ratio,
            seed=args.seed,
            grid_size=args.grid_size,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            class_weighting=args.class_weighting,
            max_class_weight=args.max_class_weight,
            noise_frame_repeat=args.noise_frame_repeat,
            force_torch_pointrope=args.force_torch_pointrope,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        payload = plan.to_jsonable()
        payload["training_command"] = None
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    try:
        if not args.resume:
            prepare_finetune_run(plan, overwrite=args.overwrite)
        command = build_training_command(plan, resume=args.resume)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": True,
                    "output_dir": plan.output_dir,
                    "config": plan.config_path,
                    "training_command": command,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    return_code = run_training(plan, resume=args.resume)
    if return_code != 0:
        raise SystemExit(return_code)
    print(
        json.dumps(
            {
                "completed": True,
                "experiment_dir": plan.experiment_dir,
                "best_checkpoint": f"{plan.experiment_dir}/model/model_best.pth",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and fine-tune LitePT on masks exported from point_labeler."
    )
    parser.add_argument("--litept-root", required=True)
    parser.add_argument("--export-dir", required=True, help="Output of export_from_point_labeler.py")
    parser.add_argument("--labeler-dir", required=True, help="Point-labeler dataset containing velodyne/ and labels.xml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Defaults to <litept-root>/pth/waymo/model_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument(
        "--class-weighting",
        choices=("sqrt_inverse", "none"),
        default="sqrt_inverse",
        help="CrossEntropy class weighting computed from valid training points",
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=10.0,
        help="Cap used by inverse-square-root class weighting",
    )
    parser.add_argument(
        "--noise-frame-repeat",
        type=int,
        default=4,
        help="How many times to include train frames containing classes whose name contains 'noise'",
    )
    parser.add_argument(
        "--force-torch-pointrope",
        action="store_true",
        help="Use LitePT's pure PyTorch PointROPE fallback during training",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing files or importing torch")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data/config/checkpoint without starting training")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated data and experiment output")
    parser.add_argument("--resume", action="store_true", help="Resume from experiment/model/model_last.pth")
    return parser.parse_args()


if __name__ == "__main__":
    main()
