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
    conversion_manifest = None

    from autolabeler.hl320.dataset import build_hl320_dataset  # noqa: WPS433

    if args.bin_dir:
        from autolabeler.hl320.bin_to_csv import convert_hl320_bin_dir_to_csv  # noqa: WPS433

        prepare_dataset_output_dir(output_dir, overwrite=args.overwrite)
        csv_out_dir = Path(args.csv_out_dir).expanduser().resolve() if args.csv_out_dir else output_dir / "converted_csv"
        conversion = convert_hl320_bin_dir_to_csv(
            bin_dir=args.bin_dir,
            output_dir=csv_out_dir,
            calibration_map=args.calibration_map,
            name_mode=args.bin_name_mode,
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
        classes_yaml=args.classes_yaml,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
        prepare_output=not bool(args.bin_dir),
        return_mode=args.return_mode,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest": str(result.manifest_path),
                "csv_dir": str(csv_dir),
                "conversion_manifest": str(conversion_manifest) if conversion_manifest is not None else None,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean HL320 point-wise dataset format.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv-dir", default=None, help="Directory with flat HL320 CSV frames.")
    input_group.add_argument("--bin-dir", default=None, help="Directory with raw HL320 .bin frames; converted to CSV before building.")
    parser.add_argument("--csv-out-dir", default=None, help="All-echo CSV output directory when --bin-dir is used. Default: <output-dir>/converted_csv.")
    parser.add_argument("--calibration-map", default=None, help="HL320 calibration_map.txt. Default: search from --bin-dir upwards.")
    parser.add_argument("--bin-name-mode", choices=["sequential", "stem"], default="sequential")
    parser.add_argument("--labels-dir", required=True, help="Directory with <frame>/semantic_mask.npy or <frame>.npy labels.")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset/{train,val}/<frame>.")
    parser.add_argument("--classes-yaml", default=str(REPO_ROOT / "configs" / "classes.yaml"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--return-mode",
        choices=("primary", "all"),
        default="primary",
        help="primary keeps only blockID == 0 rows; all trains on every echo row from the CSV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
