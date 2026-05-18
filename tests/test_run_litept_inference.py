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
