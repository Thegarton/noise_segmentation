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

from autolabeler.teachers.sam3_text_adapter import load_prompt_config, load_sam3_text_result, run_sam3_text_teacher  # noqa: E402


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    prompts = load_prompt_config(args.prompt_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for image_path in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))):
        out_npz = out_dir / f"{image_path.stem}.npz"
        if out_npz.exists() and not args.overwrite:
            status = "exists"
        else:
            run_sam3_text_teacher(
                conda_env=args.conda_env,
                sam3_script=args.sam3_script,
                image_path=image_path,
                prompt_config=args.prompt_config,
                output_npz=out_npz,
                extra_args=args.sam3_arg,
            )
            status = "created"
        if args.validate:
            load_sam3_text_result(out_npz)
        results.append({"frame_id": image_path.stem, "image": str(image_path), "sam3_npz": str(out_npz), "status": status})

    manifest = {"version": 1, "prompt_config": str(args.prompt_config), "prompts": prompts, "frames": results}
    manifest_path = out_dir / "sam3_text_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(results), "manifest": str(manifest_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a text-prompt SAM3 wrapper over synced camera frames.")
    p.add_argument("--image-dir", required=True, help="Directory with frame_id.jpg images.")
    p.add_argument("--prompt-config", default=str(REPO_ROOT / "configs" / "sam3_text_prompts_pointwise_v1.yaml"))
    p.add_argument("--conda-env", required=True)
    p.add_argument("--sam3-script", required=True, help="External SAM3 script accepting --image --prompts --output.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sam3-arg", action="append", default=[], help="Extra argument passed through to the SAM3 script.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()

