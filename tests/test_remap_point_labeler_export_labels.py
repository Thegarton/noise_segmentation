from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_remap_export_replaces_ignore_with_background_without_touching_source(tmp_path: Path):
    export_dir = tmp_path / "export"
    frame_dir = export_dir / "000000"
    frame_dir.mkdir(parents=True)
    source_mask = np.asarray([255, 2, 255, 0], dtype=np.uint16)
    np.save(frame_dir / "semantic_mask.npy", source_mask)
    (frame_dir / "metadata.json").write_text(
        json.dumps({"semantic_classes": {"background": 0, "CAR": 2, "ignore": 255}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "export_bg"

    run_script(
        "--export-dir",
        str(export_dir),
        "--out-dir",
        str(out_dir),
        "--overwrite",
    )

    np.testing.assert_array_equal(np.load(frame_dir / "semantic_mask.npy"), source_mask)
    np.testing.assert_array_equal(
        np.load(out_dir / "000000" / "semantic_mask.npy"),
        np.asarray([0, 2, 0, 0], dtype=np.uint16),
    )
    metadata = json.loads((out_dir / "000000" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["label_remaps"][0]["from_id"] == 255
    assert metadata["label_remaps"][0]["to_id"] == 0
    assert metadata["label_remaps"][0]["replaced_points"] == 2
    manifest = json.loads((out_dir / "remap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_replaced"] == 2
    assert manifest["total_points"] == 4


def run_script(*args: str) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "remap_point_labeler_export_labels.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run([sys.executable, str(script), *args], check=True, env=env, capture_output=True, text=True)
