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

    if args.echo_mode == "both":
        primary = convert_hl320_bin_dir_to_csv(
            bin_dir=args.bin_dir,
            output_dir=Path(args.out_dir).expanduser() / "primary",
            calibration_map=args.calibration_map,
            name_mode=args.name_mode,
            echo_mode="primary",
            overwrite=args.overwrite,
        )
        all_echo = convert_hl320_bin_dir_to_csv(
            bin_dir=args.bin_dir,
            output_dir=Path(args.out_dir).expanduser() / "all_echo",
            calibration_map=args.calibration_map,
            name_mode=args.name_mode,
            echo_mode="all",
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "bin_dir": str(primary.bin_dir),
                    "out_dir": str(Path(args.out_dir).expanduser().resolve()),
                    "calibration_map": str(primary.calibration_map),
                    "name_mode": args.name_mode,
                    "echo_mode": "both",
                    "primary_manifest": str(primary.manifest_path),
                    "all_echo_manifest": str(all_echo.manifest_path),
                    "frames": len(primary.frames),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = convert_hl320_bin_dir_to_csv(
        bin_dir=args.bin_dir,
        output_dir=args.out_dir,
        calibration_map=args.calibration_map,
        name_mode=args.name_mode,
        echo_mode=args.echo_mode,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "bin_dir": str(result.bin_dir),
                "out_dir": str(result.output_dir),
                "calibration_map": str(result.calibration_map),
                "manifest": str(result.manifest_path),
                "echo_mode": str(result.echo_mode),
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
    parser.add_argument("--echo-mode", choices=["primary", "all", "both"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
