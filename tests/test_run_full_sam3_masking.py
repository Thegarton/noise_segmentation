from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def test_parse_gpu_ids() -> None:
    script = load_script()

    assert script.parse_gpu_ids("0,1, 3") == ["0", "1", "3"]
    with pytest.raises(ValueError, match="duplicate"):
        script.parse_gpu_ids("0,0")


def test_build_worker_command_keeps_original_args() -> None:
    script = load_script()

    command = script.build_worker_command(
        ["--image-dir", "/images", "--out-dir", "/out", "--gpu-ids", "0,1"],
        worker_index=1,
        num_workers=2,
    )

    assert command[0] == sys.executable
    assert command[-5:] == ["--skip-conversion", "--worker-index", "1", "--num-workers", "2"]
    assert "0,1" in command


def test_merge_worker_manifests_sorts_frames(tmp_path: Path) -> None:
    script = load_script()
    for worker_index, frame_id in [(0, "000002"), (1, "000001")]:
        payload = {
            "version": 1,
            "images": 1,
            "frames": [{"frame_id": frame_id, "image": f"/{frame_id}.jpg"}],
            "worker": {"index": worker_index, "count": 2},
            "sam3_runtime": {"precision": {"effective": "fp16"}},
        }
        path = tmp_path / f"sam3_single_image_folder_manifest.worker_{worker_index:03d}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    final_path = script.merge_worker_manifests(tmp_path, num_workers=2)
    combined = json.loads(final_path.read_text(encoding="utf-8"))

    assert combined["images"] == 2
    assert [item["frame_id"] for item in combined["frames"]] == ["000001", "000002"]
    assert combined["num_workers"] == 2
    assert "worker" not in combined


def load_script():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_full_sam3_masking.py"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("run_full_sam3_masking_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

