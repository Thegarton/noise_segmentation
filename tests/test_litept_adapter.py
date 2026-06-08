from pathlib import Path

import numpy as np
import pytest

from autolabeler.data.bin_loader import H, W
from autolabeler.data.schemas import OrganizedLiDARFrame
import autolabeler.teachers.litept_adapter as litept_adapter
from autolabeler.teachers.litept_adapter import (
    LitePTModelBundle,
    LitePTUnavailableError,
    build_litept_inference_plan,
    remap_training_predictions,
    run_litept_inference,
)


def test_litept_dry_run_plan_discovers_frames_without_checkpoint(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))

    plan = build_litept_inference_plan(
        litept_root=str(litept_root),
        checkpoint=str(tmp_path / "missing.ckpt"),
        input_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
        input_format="bin",
        validate_checkpoint=False,
    )

    assert plan.frame_count == 1
    assert plan.frame_ids == ["frame_000"]
    assert plan.litept_root == str(litept_root.resolve())


def test_litept_plan_can_limit_frame_count(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for idx in range(3):
        (data_dir / f"frame_{idx:03d}.bin").write_bytes(bytes(H * W * 4 * 4))

    plan = build_litept_inference_plan(
        litept_root=str(litept_root),
        checkpoint=str(tmp_path / "missing.ckpt"),
        input_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
        input_format="bin",
        validate_checkpoint=False,
        max_frames=2,
    )

    assert plan.frame_count == 2
    assert plan.frame_ids == ["frame_000", "frame_001"]


def test_litept_plan_rejects_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="LitePT root does not exist"):
        build_litept_inference_plan(
            litept_root=str(tmp_path / "missing_litept"),
            checkpoint=str(tmp_path / "missing.ckpt"),
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            validate_checkpoint=False,
        )


def test_custom_litept_requires_explicit_config_and_checkpoint(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))

    with pytest.raises(ValueError, match="--checkpoint is required"):
        build_litept_inference_plan(
            litept_root=str(litept_root),
            checkpoint=None,
            input_dir=str(data_dir),
            output_dir=str(tmp_path / "out"),
            litept_dataset="custom",
            validate_checkpoint=False,
        )


def test_custom_predictions_are_remapped_to_source_taxonomy_ids():
    labels = np.asarray([0, 1, 0, 2], dtype=np.int64)
    assert remap_training_predictions(labels, [2, 10, 42]).tolist() == [2, 10, 2, 42]
    with pytest.raises(ValueError, match="outside"):
        remap_training_predictions(np.asarray([3]), [2, 10, 42])


def test_litept_runtime_reports_unwired_external_repo(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))

    with pytest.raises(FileNotFoundError, match="LitePT config does not exist"):
        run_litept_inference(
            litept_root=str(litept_root),
            checkpoint=str(checkpoint),
            input_dir=str(data_dir),
            output_dir=str(tmp_path / "out"),
            input_format="bin",
        )


def test_litept_runtime_adds_nearest_ins_pose_metadata(tmp_path: Path, monkeypatch):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    config_path = litept_root / "configs" / "nuscenes"
    config_path.mkdir(parents=True)
    (config_path / "semseg-litept-small-v1m1.py").write_text("# config\n", encoding="utf-8")
    checkpoint_path = litept_root / "pth" / "nuscenes"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / "model_best.pth").write_bytes(b"checkpoint")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))
    ins_path = data_dir / "ins"
    ins_path.write_text(
        "secs\tnsecs\tattitude_X\tattitude_Y\tattitude_Z\tlatitude\tlongitude\televation\tutmPosition_X\tutmPosition_Y\tutmPosition_Z\n"
        "10\t100000000\t0\t0\t0\t55.0\t37.0\t150.0\t1\t2\t3\n"
        "10\t300000000\t0\t0\t0\t55.0\t37.0\t150.0\t4\t5\t6\n",
        encoding="utf-8",
    )

    def fake_load_litept_model(**kwargs):
        return LitePTModelBundle(
            model=object(),
            cfg=object(),
            device="cpu",
            device_name="cpu",
            device_capability=None,
            num_classes=1,
            class_names=["class_0"],
            pointrope_backend="torch",
            dataset_name="nuscenes",
            config_file=kwargs["litept_config"],
            checkpoint_file=kwargs["checkpoint"],
        )

    monkeypatch.setattr(litept_adapter, "_load_litept_model", fake_load_litept_model)
    monkeypatch.setattr(
        litept_adapter,
        "load_frame",
        lambda path, frame_id, input_format: OrganizedLiDARFrame(
            frame_id=frame_id,
            points_range=np.zeros((H, W, 4), dtype=np.float32),
            points_flat=np.zeros((H * W, 4), dtype=np.float32),
            timestamp_us=10_260_000,
        ),
    )
    monkeypatch.setattr(
        litept_adapter,
        "_predict_frame",
        lambda model, points_range: (np.zeros((H, W), dtype=np.uint16), np.ones((H, W), dtype=np.float32)),
    )

    results = run_litept_inference(
        litept_root=str(litept_root),
        checkpoint=None,
        input_dir=str(data_dir),
        output_dir=str(tmp_path / "out"),
        input_format="bin",
        ins_path=str(ins_path),
        force_torch_pointrope=True,
    )

    pose = results[0].metadata["ego_pose"]
    assert pose["source_timestamp_us"] == 10_300_000
    assert pose["delta_us"] == 40_000
    assert pose["translation"] == [4.0, 5.0, 6.0]
    assert len(pose["kitti_pose"]) == 12
