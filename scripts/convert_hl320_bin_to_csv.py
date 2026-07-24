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


def main() -> None:
    args = parse_args()
    from autolabeler.hl320.bin_to_csv import convert_hl320_bin_dir_to_csv  # noqa: WPS433

    result = convert_hl320_bin_dir_to_csv(
        bin_dir=args.bin_dir,
        output_dir=args.out_dir,
        calibration_map=args.calibration_map,
        name_mode=args.name_mode,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "bin_dir": str(result.bin_dir),
                "out_dir": str(result.output_dir),
                "calibration_map": str(result.calibration_map),
                "manifest": str(result.manifest_path),
                "echo_layout": "all_returns",
                "primary_return": "blockID == 0",
                "frames": len(result.frames),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw HL320 .bin frames to flat CSV files.")
    parser.add_argument("--bin-dir", required=True, help="Directory with raw HL320 .bin frames.")
    parser.add_argument("--out-dir", required=True, help="Output directory for converted CSV files.")
    parser.add_argument("--calibration-map", default=None, help="HL320 calibration_map.txt. Default: search from --bin-dir upwards.")
    parser.add_argument("--name-mode", choices=["sequential", "stem"], default="sequential")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
