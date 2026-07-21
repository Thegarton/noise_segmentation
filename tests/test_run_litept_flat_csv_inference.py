from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_load_flat_csv_supports_tab_separated_hl320_columns(tmp_path: Path):
    script = load_script()
    csv_path = tmp_path / "000000.csv"
    csv_path.write_text(
        "x\ty\tz\tazimuth\tvertical\tintensity\tslot\tpixel\thcell\tvcell\tCxd\tCyd\n"
        "0.1\t0.2\t0.3\t89\t38\t3\t2\t0\t49\t1\t294.1\t173.2\n"
        "0.4\t0.5\t0.6\t88\t37\t0\t2\t1\t52\t1\t305.2\t183.3\n",
        encoding="utf-8",
    )

    frame = script.load_flat_csv(csv_path)

    assert frame["frame_id"] == "000000"
    np.testing.assert_allclose(
        frame["points"],
        np.asarray([[0.1, 0.2, 0.3, 3.0], [0.4, 0.5, 0.6, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(frame["optional"]["cxd"], np.asarray([294.1, 305.2], dtype=np.float32))
    np.testing.assert_allclose(frame["optional"]["slot"], np.asarray([2, 2], dtype=np.float32))


def test_hl320_feature_mode_builds_multichannel_strength(tmp_path: Path):
    script = load_script()
    csv_path = tmp_path / "000000.csv"
    csv_path.write_text(
        "x y z azimuth vertical intensity reflectivity slot pixel blockID Cxd Cyd\n"
        "1 0 0 89 38 3 4 2 0 1 294.1 173.2\n"
        "2 0 0 88 37 6 7 2 0 0 294.1 173.2\n",
        encoding="utf-8",
    )

    frame = script.load_inference_frame(csv_path, feature_mode="hl320")

    assert frame["points"].shape == (2, 4)
    assert frame["strength"].ndim == 2
    assert frame["strength"].shape[0] == 2
    assert frame["strength"].shape[1] > 1


def test_dry_run_resolves_finetune_artifacts_without_importing_litept(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_litept_flat_csv_inference.py"
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "000000.csv").write_text("x y z intensity\n1 2 3 4\n", encoding="utf-8")
    litept_root = tmp_path / "LitePT"
    litept_root.mkdir()
    fine_tune_dir = tmp_path / "fine_tune"
    (fine_tune_dir / "experiment" / "model").mkdir(parents=True)
    (fine_tune_dir / "experiment" / "model" / "model_best.pth").write_bytes(b"checkpoint")
    (fine_tune_dir / "litept_custom_config.py").write_text("training_id_to_source_id = [0]\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--litept-root",
            str(litept_root),
            "--csv-dir",
            str(csv_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--fine-tune-dir",
            str(fine_tune_dir),
            "--dry-run",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["frames"] == ["000000"]
    assert payload["frame_count"] == 1
    assert payload["checkpoint"].endswith("experiment/model/model_best.pth")
    assert payload["litept_config"].endswith("litept_custom_config.py")
    assert payload["feature_mode"] == "auto"


def load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_litept_flat_csv_inference.py"
    spec = importlib.util.spec_from_file_location("run_litept_flat_csv_inference", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
