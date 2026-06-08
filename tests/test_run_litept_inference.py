import json
import os
import subprocess
import sys
from pathlib import Path

from autolabeler.data.bin_loader import H, W


def test_run_litept_inference_dry_run_cli(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))
    (data_dir / "frame_001.bin").write_bytes(bytes(H * W * 4 * 4))

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_litept_inference.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--litept-root",
            str(litept_root),
            "--checkpoint",
            str(tmp_path / "missing.ckpt"),
            "--input-dir",
            str(data_dir),
            "--input-format",
            "bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--max-frames",
            "1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["frame_count"] == 1
    assert payload["frame_ids"] == ["frame_000"]
    assert payload["max_frames"] == 1


def test_run_custom_litept_dry_run_requires_and_accepts_custom_artifacts(tmp_path: Path):
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    config = tmp_path / "litept_custom_config.py"
    config.write_text("training_id_to_source_id = [2, 10]\n", encoding="utf-8")
    checkpoint = tmp_path / "model_best.pth"
    checkpoint.write_bytes(b"checkpoint")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "frame_000.bin").write_bytes(bytes(H * W * 4 * 4))

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_litept_inference.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--litept-root",
            str(litept_root),
            "--litept-dataset",
            "custom",
            "--litept-config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--input-dir",
            str(data_dir),
            "--input-format",
            "bin",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["litept_dataset"] == "custom"
    assert payload["litept_config"] == str(config.resolve())
    assert payload["checkpoint"] == str(checkpoint.resolve())
