from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autolabeler.camera.calibration import load_camera_calibration
from autolabeler.camera.projection import project_points_to_image
from autolabeler.camera.sync import VideoTimestamp, nearest_video_timestamp, read_video_timestamps_csv
from autolabeler.teachers.sam3_text_adapter import Sam3TextResult, load_prompt_config, load_sam3_text_result


def test_read_video_timestamps_and_nearest_match(tmp_path: Path):
    path = tmp_path / "timestamps.csv"
    path.write_text("idx,t\n0,1000\n1,1035\n2,1100\n", encoding="utf-8")

    timestamps = read_video_timestamps_csv(
        path,
        frame_index_col="idx",
        timestamp_col="t",
        timestamp_unit="ms",
    )
    match = nearest_video_timestamp(timestamps, 1_040_000)

    assert timestamps[0] == VideoTimestamp(frame_index=0, timestamp_us=1_000_000)
    assert match.frame_index == 1
    assert match.timestamp_us == 1_035_000


def test_camera_calibration_loads_and_projects_points(tmp_path: Path):
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps(
            {
                "K": [[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
                "T_lidar_to_camera": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )

    calibration = load_camera_calibration(calib_path, extrinsic_direction="lidar_to_camera")
    pixels, depth, visible = project_points_to_image(
        np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        calibration,
        image_shape=(10, 12),
        use_z_buffer=True,
    )

    assert pixels.tolist() == [[5, 4], [6, 4], [-1, -1]]
    assert depth.tolist() == [1.0, 1.0, -1.0]
    assert visible.tolist() == [True, True, False]


def test_camera_calibration_scales_intrinsics_to_resized_image(tmp_path: Path):
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        json.dumps(
            {
                "image_size": [20, 10],
                "K": [[10.0, 0.0, 10.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]],
                "T_lidar_to_camera": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )

    calibration = load_camera_calibration(calib_path, extrinsic_direction="lidar_to_camera")
    pixels, _, visible = project_points_to_image(
        np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]], dtype=np.float32),
        calibration,
        image_shape=(5, 10),
        use_z_buffer=False,
    )

    assert pixels.tolist() == [[5, 2], [6, 2]]
    assert visible.tolist() == [True, True]


def test_sam3_prompt_config_and_npz_validation(tmp_path: Path):
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text('traffic_sign:\n  - "traffic sign"\nCAR: ["car", "passenger car"]\n', encoding="utf-8")
    prompts = load_prompt_config(prompt_path)

    mask = np.zeros((2, 4, 5), dtype=bool)
    npz_path = tmp_path / "sam3.npz"
    np.savez(
        npz_path,
        masks=mask,
        scores=np.asarray([0.8, 0.9], dtype=np.float32),
        labels=np.asarray(["traffic_sign", "CAR"]),
        prompts=np.asarray(["traffic sign", "car"]),
        boxes_xyxy=np.zeros((2, 4), dtype=np.float32),
    )
    result = load_sam3_text_result(npz_path)

    assert prompts == {"traffic_sign": ["traffic sign"], "CAR": ["car", "passenger car"]}
    assert result.masks.shape == (2, 4, 5)
    assert result.labels.tolist() == ["traffic_sign", "CAR"]


def test_sam3_npz_rejects_wrong_mask_dtype(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, masks=np.zeros((1, 2, 2), dtype=np.uint8), scores=np.ones(1), labels=np.asarray(["x"]), prompts=np.asarray(["x"]))

    with pytest.raises(ValueError, match="SAM3 masks must be bool"):
        load_sam3_text_result(path)
