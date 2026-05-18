#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Run LitePT point-wise semantic segmentation over organized LiDAR frames")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/classes.yaml")
    args = p.parse_args()

    raise NotImplementedError(
        "LitePT inference integration is not implemented yet. "
        "Run this script from conda env LItePT after adding the LitePT model adapter. "
        f"Requested input_dir={args.input_dir} output_dir={args.output_dir} checkpoint={args.checkpoint}"
    )


if __name__ == "__main__":
    main()
