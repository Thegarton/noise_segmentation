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

from autolabeler.hl320.fusion import FusionPolicy, fuse_hl320_predictions  # noqa: E402


def main() -> None:
    args = parse_args()
    policy = FusionPolicy(
        min_sam3_confidence=args.min_sam3_confidence,
        litept_keep_threshold=args.litept_keep_threshold,
        noise_protect_threshold=args.noise_protect_threshold,
        sam3_override_margin=args.sam3_override_margin,
        allow_sam3_background=args.allow_sam3_background,
    )
    manifest = fuse_hl320_predictions(
        csv_dir=args.csv_dir,
        litept_dir=args.litept_dir,
        sam3_dir=args.sam3_dir,
        classes_yaml=args.classes_yaml,
        out_dir=args.out_dir,
        policy=policy,
        overwrite=args.overwrite,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "out_dir": manifest["out_dir"],
                "manifest": manifest["manifest"],
                "frame_count": len(manifest["frames"]),
                "summary": manifest["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse point-wise LitePT predictions with SAM3 image masks lifted through HL320 Cxd/Cyd.",
    )
    parser.add_argument("--csv-dir", required=True, help="Directory with flat HL320 CSV frames.")
    parser.add_argument("--litept-dir", required=True, help="LitePT inference output directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--sam3-dir", required=True, help="SAM3 image output directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--out-dir", required=True, help="Output directory for fused point predictions.")
    parser.add_argument("--min-sam3-confidence", type=float, default=0.7)
    parser.add_argument("--litept-keep-threshold", type=float, default=0.6)
    parser.add_argument("--noise-protect-threshold", type=float, default=0.4)
    parser.add_argument("--sam3-override-margin", type=float, default=0.2)
    parser.add_argument(
        "--allow-sam3-background",
        action="store_true",
        help="Allow camera background=0 to participate in fusion. Disabled by default.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
