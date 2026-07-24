#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_dir = Path(args.csv_dir).expanduser().resolve() if args.csv_dir else None
    echo_csv_dir = Path(args.echo_csv_dir).expanduser().resolve() if args.echo_csv_dir else None
    conversion_manifest = None
    conversion_primary_manifest = None
    conversion_all_echo_manifest = None

    from autolabeler.hl320.dataset import build_hl320_dataset  # noqa: WPS433

    if args.bin_dir:
        from autolabeler.hl320.bin_to_csv import convert_hl320_bin_dir_to_csv  # noqa: WPS433

        if args.bin_echo_mode == "both" and echo_csv_dir is not None:
            raise ValueError("--echo-csv-dir cannot be combined with --bin-echo-mode both")
        prepare_dataset_output_dir(output_dir, overwrite=args.overwrite)
        csv_out_dir = Path(args.csv_out_dir).expanduser().resolve() if args.csv_out_dir else output_dir / "converted_csv"
        if args.bin_echo_mode == "both":
            primary = convert_hl320_bin_dir_to_csv(
                bin_dir=args.bin_dir,
                output_dir=csv_out_dir / "primary",
                calibration_map=args.calibration_map,
                name_mode=args.bin_name_mode,
                echo_mode="primary",
                overwrite=args.overwrite,
            )
            all_echo = convert_hl320_bin_dir_to_csv(
                bin_dir=args.bin_dir,
                output_dir=csv_out_dir / "all_echo",
                calibration_map=args.calibration_map,
                name_mode=args.bin_name_mode,
                echo_mode="all",
                overwrite=args.overwrite,
            )
            csv_dir = primary.output_dir
            echo_csv_dir = all_echo.output_dir
            conversion_primary_manifest = primary.manifest_path
            conversion_all_echo_manifest = all_echo.manifest_path
            conversion_manifest = write_combined_conversion_manifest(
                csv_out_dir,
                primary_manifest=conversion_primary_manifest,
                all_echo_manifest=conversion_all_echo_manifest,
                mode="both",
            )
        else:
            conversion = convert_hl320_bin_dir_to_csv(
                bin_dir=args.bin_dir,
                output_dir=csv_out_dir,
                calibration_map=args.calibration_map,
                name_mode=args.bin_name_mode,
                echo_mode=args.bin_echo_mode,
                overwrite=args.overwrite,
            )
            csv_dir = conversion.output_dir
            conversion_manifest = conversion.manifest_path

    if csv_dir is None:
        raise ValueError("Internal error: csv_dir was not resolved")
    result = build_hl320_dataset(
        csv_dir=csv_dir,
        labels_dir=args.labels_dir,
        output_dir=output_dir,
        echo_csv_dir=echo_csv_dir,
        classes_yaml=args.classes_yaml,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        prepare_output=not bool(args.bin_dir),
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "csv_dir": str(csv_dir),
                "echo_csv_dir": str(echo_csv_dir) if echo_csv_dir is not None else None,
                "conversion_manifest": str(conversion_manifest) if conversion_manifest is not None else None,
                "conversion_primary_manifest": str(conversion_primary_manifest) if conversion_primary_manifest is not None else None,
                "conversion_all_echo_manifest": str(conversion_all_echo_manifest) if conversion_all_echo_manifest is not None else None,
                "frame_count": result.frame_count,
                "train_frames": result.train_frames,
                "val_frames": result.val_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def prepare_dataset_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_combined_conversion_manifest(
    output_dir: Path,
    *,
    primary_manifest: Path,
    all_echo_manifest: Path,
    mode: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hl320_bin_to_csv_both_manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "format": "hl320_bin_to_csv_both_v1",
                "echo_mode": mode,
                "primary_manifest": str(primary_manifest),
                "all_echo_manifest": str(all_echo_manifest),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean HL320 point-wise dataset format.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv-dir", default=None, help="Directory with flat HL320 CSV frames.")
    input_group.add_argument("--bin-dir", default=None, help="Directory with raw HL320 .bin frames; converted to CSV before building.")
    parser.add_argument("--echo-csv-dir", default=None, help="Optional all-echo CSV directory used only for multi-echo features.")
    parser.add_argument("--csv-out-dir", default=None, help="CSV output directory when --bin-dir is used. Default: <output-dir>/converted_csv.")
    parser.add_argument("--calibration-map", default=None, help="HL320 calibration_map.txt. Default: search from --bin-dir upwards.")
    parser.add_argument("--bin-name-mode", choices=["sequential", "stem"], default="sequential")
    parser.add_argument(
        "--bin-echo-mode",
        choices=["primary", "all", "both"],
        default="both",
        help="CSV echo rows when --bin-dir is used. both keeps primary labels and uses all_echo only for features.",
    )
    parser.add_argument("--labels-dir", required=True, help="Directory with <frame>/semantic_mask.npy or <frame>.npy labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset/{train,val}/<frame>.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
