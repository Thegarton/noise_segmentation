from pathlib import Path

import pytest

from autolabeler.data.bin_loader import H, W
from autolabeler.teachers.litept_adapter import (
    LitePTUnavailableError,
    build_litept_inference_plan,
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


def test_litept_plan_rejects_missing_root(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="LitePT root does not exist"):
        build_litept_inference_plan(
            litept_root=str(tmp_path / "missing_litept"),
            checkpoint=str(tmp_path / "missing.ckpt"),
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            validate_checkpoint=False,
        )


def test_litept_runtime_reports_unwired_external_repo(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))

    with pytest.raises(LitePTUnavailableError, match="not wired to this external repository layout"):
        run_litept_inference(
            litept_root=str(litept_root),
            checkpoint=str(checkpoint),
            input_dir=str(data_dir),
            output_dir=str(tmp_path / "out"),
            input_format="bin",
        )
