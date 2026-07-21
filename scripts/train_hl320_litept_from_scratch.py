#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.hl320.litept_training import (  # noqa: E402
    build_training_command,
    build_training_plan,
    prepare_training_run,
    run_training,
)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    if args.resume and args.prepare_only:
        raise SystemExit("--resume and --prepare-only cannot be used together")

    try:
        plan = build_training_plan(
            litept_root=args.litept_root,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            grid_size=args.grid_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weighting=args.class_weighting,
            max_class_weight=args.max_class_weight,
            noise_frame_repeat=args.noise_frame_repeat,
            seed=args.seed,
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
            prepare_training_run(plan, overwrite=args.overwrite)
        command = build_training_command(plan, resume=args.resume)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
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
    parser = argparse.ArgumentParser(description="Train LitePT from scratch on the clean HL320 dataset format.")
    parser.add_argument("--litept-root", required=True)
    parser.add_argument("--dataset-root", required=True, help="Output root from scripts/build_hl320_dataset.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weighting",
        choices=("sqrt_inverse", "none"),
        default="sqrt_inverse",
    )
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    parser.add_argument("--noise-frame-repeat", type=int, default=4)
    parser.add_argument("--force-torch-pointrope", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without writing files or importing torch")
    parser.add_argument("--prepare-only", action="store_true", help="Write config/manifests but do not start training")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from <output-dir>/experiment/model/model_last.pth")
    return parser.parse_args()


if __name__ == "__main__":
    main()
