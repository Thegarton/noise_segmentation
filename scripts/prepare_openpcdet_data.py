#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autolabeler.teachers.openpcdet_adapter import prepare_openpcdet_points


def main() -> None:
    p = argparse.ArgumentParser(description="Convert organized CSV/bin LiDAR frames to OpenPCDet .npy point clouds")
    p.add_argument("--input-dir", default="./data")
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--output-dir", default="./out/openpcdet")
    args = p.parse_args()

    prepared = prepare_openpcdet_points(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_format=args.input_format,
    )

    index_path = Path(args.output_dir) / "frames.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([x.__dict__ for x in prepared], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prepared_frames={len(prepared)} index={index_path}")


if __name__ == "__main__":
    main()
