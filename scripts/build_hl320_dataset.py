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

from autolabeler.hl320.dataset import build_hl320_dataset  # noqa: E402


def main() -> None:
    args = parse_args()
    result = build_hl320_dataset(
        csv_dir=args.csv_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        classes_yaml=args.classes_yaml,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "frame_count": result.frame_count,
                "train_frames": result.train_frames,
                "val_frames": result.val_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean HL320 point-wise dataset format.")
    parser.add_argument("--csv-dir", required=True, help="Directory with flat HL320 CSV frames.")
    parser.add_argument("--labels-dir", required=True, help="Directory with <frame>/semantic_mask.npy or <frame>.npy labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset/{train,val}/<frame>.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
