#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autolabeler.data.csv_loader import load_csv  # noqa: E402
from autolabeler.noise.candidates import build_noise_review_candidates  # noqa: E402


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    frame_id = args.frame_id or csv_path.stem
    frame = load_csv(str(csv_path), frame_id=frame_id)
    semantic = np.load(args.semantic_mask, allow_pickle=False)
    confidence = np.load(args.confidence, allow_pickle=False) if args.confidence else None
    candidates = build_noise_review_candidates(
        points_range=frame.points_range,
        semantic_mask=semantic,
        confidence_mask=confidence,
        low_confidence_threshold=args.low_confidence_threshold,
        streak_min_valid=args.streak_min_valid,
    )

    out_dir = Path(args.out_dir) / frame_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"frame_id": frame_id, "candidates": {}}
    for name, mask in candidates.items():
        path = out_dir / f"{name}.npy"
        np.save(path, mask.astype(bool, copy=False))
        summary["candidates"][name] = {"path": str(path), "points": int(np.count_nonzero(mask))}
    (out_dir / "noise_candidate_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build high-precision review candidate masks for coarse noise labeling.")
    p.add_argument("--csv", required=True)
    p.add_argument("--semantic-mask", required=True)
    p.add_argument("--confidence", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frame-id", default=None)
    p.add_argument("--low-confidence-threshold", type=float, default=0.25)
    p.add_argument("--streak-min-valid", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    main()
