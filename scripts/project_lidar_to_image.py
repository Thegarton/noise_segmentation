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

from autolabeler.camera.calibration import load_camera_calibration  # noqa: E402
from autolabeler.camera.projection import project_points_to_image, save_projection_arrays, save_projection_overlay  # noqa: E402
from autolabeler.data.csv_loader import load_csv  # noqa: E402


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    frame_id = args.frame_id or csv_path.stem
    frame = load_csv(str(csv_path), frame_id=frame_id)
    image_shape = read_image_shape(args.image)
    calibration = load_camera_calibration(args.calibration_json, extrinsic_direction=args.extrinsic_direction)
    point_to_pixel, depth, visible = project_points_to_image(
        frame.points_flat[:, :3],
        calibration,
        image_shape=image_shape,
        use_z_buffer=not args.no_z_buffer,
    )

    out_dir = Path(args.out_dir) / frame_id
    paths = save_projection_arrays(out_dir, point_to_pixel=point_to_pixel, point_camera_depth=depth)
    overlay_path = out_dir / "projection_overlay.jpg"
    save_projection_overlay(args.image, overlay_path, point_to_pixel=point_to_pixel, depth=depth)
    paths["projection_overlay"] = str(overlay_path)

    summary = {
        "frame_id": frame_id,
        "csv": str(csv_path),
        "image": str(Path(args.image)),
        "calibration_json": str(Path(args.calibration_json)),
        "visible_points": int(np.count_nonzero(visible)),
        "point_count": int(frame.points_flat.shape[0]),
        **paths,
    }
    (out_dir / "projection_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def read_image_shape(image_path: str) -> tuple[int, int]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("project_lidar_to_image.py requires OpenCV. Install opencv-python or use an env with cv2.") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return int(image.shape[0]), int(image.shape[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project organized LiDAR points into a camera image and write QA artifacts.")
    p.add_argument("--csv", required=True, help="LiDAR frame CSV.")
    p.add_argument("--image", required=True, help="Synced camera image.")
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--extrinsic-direction", choices=["lidar_to_camera", "camera_to_lidar"], default="lidar_to_camera")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frame-id", default=None)
    p.add_argument("--no-z-buffer", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()

