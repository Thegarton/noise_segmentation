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

from autolabeler.hl320.teacher import TeacherPolicy, build_hl320_teacher_candidates  # noqa: E402


def main() -> None:
    args = parse_args()
    policy = TeacherPolicy(
        min_candidate_confidence=args.min_candidate_confidence,
        angular_bin_deg=args.angular_bin_deg,
        isolated_density_max=args.isolated_density_max,
        stable_distance_tolerance_m=args.stable_distance_tolerance,
        low_intensity_threshold=args.low_intensity_threshold,
        long_distance_min_m=args.long_distance_min,
        sam3_min_confidence=args.sam3_min_confidence,
    )
    result = build_hl320_teacher_candidates(
        csv_dir=args.csv_dir,
        classes_yaml=args.classes_yaml,
        out_dir=args.out_dir,
        sam3_dir=args.sam3_dir,
        litept_dir=args.litept_dir,
        image_dir=args.image_dir,
        overwrite=args.overwrite,
        policy=policy,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "frame_count": result.frame_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physics/rules teacher candidates for flat HL320 CSV frames.")
    parser.add_argument("--csv-dir", required=True, help="Directory with flat HL320 CSV frames.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--out-dir", required=True, help="Output directory for teacher candidates.")
    parser.add_argument("--sam3-dir", default=None, help="Optional SAM3 output directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--litept-dir", default=None, help="Optional LitePT output directory with <frame>/semantic_mask.npy.")
    parser.add_argument("--image-dir", default=None, help="Optional source image directory; used only for metadata.")
    parser.add_argument("--min-candidate-confidence", type=float, default=0.35)
    parser.add_argument("--angular-bin-deg", type=float, default=0.5)
    parser.add_argument("--isolated-density-max", type=int, default=3)
    parser.add_argument("--stable-distance-tolerance", type=float, default=0.75)
    parser.add_argument("--low-intensity-threshold", type=float, default=0.08)
    parser.add_argument("--long-distance-min", type=float, default=80.0)
    parser.add_argument("--sam3-min-confidence", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
