import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_visualize_semantic_mask_writes_png_and_legend(tmp_path: Path):
    mask = np.array([[0, 1, 255], [2, 2, 1]], dtype=np.uint16)
    confidence = np.ones(mask.shape, dtype=np.float32)
    mask_path = tmp_path / "semantic_mask.npy"
    confidence_path = tmp_path / "confidence.npy"
    metadata_path = tmp_path / "metadata.json"
    output_path = tmp_path / "mask.png"
    legend_path = tmp_path / "legend.json"
    np.save(mask_path, mask)
    np.save(confidence_path, confidence)
    metadata_path.write_text(json.dumps({"class_names": ["background", "car", "truck"]}), encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "visualize_semantic_mask.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--mask",
            str(mask_path),
            "--confidence",
            str(confidence_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
            "--legend",
            str(legend_path),
            "--scale",
            "1",
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        text=True,
    )

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    legend = json.loads(legend_path.read_text(encoding="utf-8"))
    assert {item["name"] for item in legend} == {"background", "car", "truck", "ignore"}
