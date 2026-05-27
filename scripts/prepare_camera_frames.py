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

from autolabeler.camera.sync import (  # noqa: E402
    build_camera_frame_matches,
    extract_synced_camera_frames,
    read_video_timestamps_csv,
    write_camera_frame_manifest,
)


def main() -> None:
    args = parse_args()
    video_timestamps = read_video_timestamps_csv(
        args.video_timestamps_csv,
        frame_index_col=args.video_frame_index_col,
        timestamp_col=args.video_timestamp_col,
        timestamp_unit=args.timestamp_unit,
    )
    matches = build_camera_frame_matches(
        lidar_dir=args.lidar_dir,
        out_dir=args.out_dir,
        video_timestamps=video_timestamps,
        max_delta_ms=args.max_delta_ms,
    )
    extract_synced_camera_frames(video_path=args.video, matches=matches)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "camera_frame_manifest.json"
    write_camera_frame_manifest(manifest_path, matches)

    unsynced = [m for m in matches if not m.synced]
    print(
        json.dumps(
            {
                "frames": len(matches),
                "synced": len(matches) - len(unsynced),
                "unsynced": len(unsynced),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    for match in unsynced:
        print(
            f"[WARN] unsynced frame_id={match.frame_id} lidar_ts={match.lidar_timestamp_us} "
            f"video_frame={match.video_frame_index} delta_ms={match.delta_ms}",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract video frames nearest to LiDAR CSV timestamps.")
    p.add_argument("--lidar-dir", required=True, help="Directory with frame_id.csv LiDAR frames.")
    p.add_argument("--video", required=True, help="Path to camera video.")
    p.add_argument("--video-timestamps-csv", required=True, help="CSV with video frame timestamps.")
    p.add_argument("--out-dir", required=True, help="Output directory for frame_id.jpg and manifest.")
    p.add_argument("--video-frame-index-col", default="frame_index")
    p.add_argument("--video-timestamp-col", default="timestamp")
    p.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="us")
    p.add_argument("--max-delta-ms", type=float, default=50.0)
    return p.parse_args()


if __name__ == "__main__":
    main()

