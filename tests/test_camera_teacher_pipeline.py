from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autolabeler.camera.calibration import load_camera_calibration
from autolabeler.camera.projection import project_points_to_image
from autolabeler.camera.sync import VideoTimestamp, nearest_video_timestamp, read_video_timestamps_csv
from autolabeler.data.bin_loader import H, W
from autolabeler.fusion.pointwise_teacher_fusion import (
    class_names_to_project_ids,
    fuse_litept_and_sam3,
    lift_sam3_candidates_to_points,
    load_name_mapping,
)
from autolabeler.noise.candidates import build_noise_review_candidates
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


def test_litept_sam3_fusion_accepts_candidates_and_ignores_weak_conflicts(tmp_path: Path):
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text("source_to_target:\n  Car: CAR\n  Road: road\n", encoding="utf-8")
    source_to_target = load_name_mapping(mapping_path)
    project_classes = {"background": 0, "CAR": 2, "road": 6, "traffic_sign": 10, "ignore": 255}
    litept_ids = class_names_to_project_ids(
        ["background", "Car", "Road"],
        source_to_target=source_to_target,
        project_classes=project_classes,
    )

    litept_mask = np.zeros((H, W), dtype=np.uint16)
    litept_conf = np.zeros((H, W), dtype=np.float32)
    litept_mask[0, 0] = 1
    litept_conf[0, 0] = 0.9
    litept_mask[0, 1] = 1
    litept_conf[0, 1] = 0.2

    sam3_mask = np.zeros((1, 8, 8), dtype=bool)
    sam3_mask[0, 1, 1] = True
    sam3_mask[0, 2, 2] = True
    sam3 = Sam3TextResult(
        masks=sam3_mask,
        scores=np.asarray([0.95], dtype=np.float32),
        labels=np.asarray(["traffic_sign"]),
        prompts=np.asarray(["traffic sign"]),
    )
    point_to_pixel = np.full((H * W, 2), -1, dtype=np.int32)
    point_to_pixel[1] = [1, 1]
    point_to_pixel[W + 1] = [2, 2]
    candidate, candidate_conf = lift_sam3_candidates_to_points(
        sam3,
        point_to_pixel=point_to_pixel,
        label_to_id={"traffic_sign": 10},
        min_points_per_mask=1,
        min_score=0.7,
    )
    fused = fuse_litept_and_sam3(
        litept_mask=litept_mask,
        litept_confidence=litept_conf,
        litept_id_to_project_id=litept_ids,
        sam3_candidate_mask=candidate,
        sam3_candidate_confidence=candidate_conf,
        litept_accept_threshold=0.65,
        sam3_accept_threshold=0.7,
    )

    assert fused.semantic_mask[0, 0] == 2
    assert fused.semantic_mask[0, 1] == 10
    assert fused.semantic_mask[1, 1] == 10
    assert fused.provenance_counts["sam3_text"] == 2


def test_noise_review_candidates_do_not_turn_all_residual_into_hard_noise():
    points = np.zeros((H, W, 4), dtype=np.float32)
    points[10, :30, 0] = 10.0
    points[10, :30, 1] = np.linspace(0.0, 1.0, 30, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint16)

    candidates = build_noise_review_candidates(points_range=points, semantic_mask=mask, streak_min_valid=12)

    assert candidates["residual_candidate"].dtype == np.bool_
    assert np.count_nonzero(candidates["residual_candidate"]) == 30
    assert np.count_nonzero(candidates["range_streak_candidate"]) >= 12
    assert "noise" not in candidates


def test_sam3_npz_rejects_wrong_mask_dtype(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, masks=np.zeros((1, 2, 2), dtype=np.uint8), scores=np.ones(1), labels=np.asarray(["x"]), prompts=np.asarray(["x"]))

    with pytest.raises(ValueError, match="SAM3 masks must be bool"):
        load_sam3_text_result(path)
