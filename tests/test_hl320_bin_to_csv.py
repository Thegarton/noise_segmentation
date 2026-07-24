from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from autolabeler.hl320.bin_to_csv import (
    BATCH_POINT_LEN,
    ECHO_COUNT,
    GZIP_HEADER_LEN,
    SLOT_DATA_SIZE,
    convert_hl320_bin_dir_to_csv,
    convert_hl320_bin_to_csv,
    resolve_calibration_map,
)
from autolabeler.hl320.csv_points import HL320_FEATURE_NAMES, load_hl320_csv, primary_returns_frame


def test_convert_hl320_bin_to_csv_preserves_row_order_and_columns(tmp_path: Path):
    calibration_path = write_calibration(tmp_path / "calibration_map.txt")
    bin_path = tmp_path / "frame.bin"
    write_synthetic_bin(bin_path, width=2, height=2)

    result = convert_hl320_bin_to_csv(
        bin_path=bin_path,
        output_csv=tmp_path / "000000.csv",
        calibration=json.loads(calibration_path.read_text(encoding="utf-8")),
        frame_id="000000",
    )

    assert result.rows == 12
    rows = read_csv_rows(result.output_csv)
    assert list(rows[0]) == ["x", "y", "z", "azimuth", "vertical", "intensity", "slot", "pixel", "blockID", "hcell", "vcell", "Cxd", "Cyd"]
    assert [float(row["x"]) for row in rows[:6]] == [1.0, 1.25, 1.5, 2.0, 2.25, 2.5]
    assert [int(row["slot"]) for row in rows[:6]] == [2, 2, 2, 2, 2, 2]
    assert [int(row["pixel"]) for row in rows[:6]] == [0, 0, 0, 1, 1, 1]
    assert [int(row["blockID"]) for row in rows[:6]] == [0, 1, 2, 0, 1, 2]
    assert int(rows[0]["hcell"]) == 49
    assert int(rows[0]["vcell"]) == 1
    assert float(rows[0]["Cxd"]) == 123.0
    assert float(rows[0]["Cyd"]) == 456.0

    frame = load_hl320_csv(result.output_csv)
    assert frame.point_count == 12
    np.testing.assert_allclose(frame.points[:6, 0], np.asarray([1.0, 1.25, 1.5, 2.0, 2.25, 2.5], dtype=np.float32))
    primary = primary_returns_frame(frame)
    assert primary.point_count == 4
    np.testing.assert_allclose(primary.points[:, 0], np.asarray([1.0, 2.0, 11.0, 12.0], dtype=np.float32))


def test_primary_returns_frame_filters_block_id_zero_without_writing_extra_csv(tmp_path: Path):
    calibration_path = write_calibration(tmp_path / "calibration_map.txt")
    bin_path = tmp_path / "frame.bin"
    write_synthetic_bin(bin_path, width=2, height=2)

    result = convert_hl320_bin_to_csv(
        bin_path=bin_path,
        output_csv=tmp_path / "000000.csv",
        calibration=json.loads(calibration_path.read_text(encoding="utf-8")),
        frame_id="000000",
    )

    primary = primary_returns_frame(load_hl320_csv(result.output_csv))
    assert result.rows == 12
    assert primary.point_count == 4
    np.testing.assert_allclose(primary.points[:, 0], np.asarray([1.0, 2.0, 11.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(primary.fields["block_id"], np.zeros((4,), dtype=np.float32))


def test_convert_hl320_bin_dir_supports_sequential_stem_and_calibration_discovery(tmp_path: Path):
    root = tmp_path / "capture"
    bin_dir = root / "nested" / "bin"
    bin_dir.mkdir(parents=True)
    calibration_path = write_calibration(root / "calibration_map.txt")
    write_synthetic_bin(bin_dir / "b_frame.bin", width=1, height=1, base_x=10.0)
    write_synthetic_bin(bin_dir / "a_frame.bin", width=1, height=1, base_x=20.0)

    assert resolve_calibration_map(bin_dir, calibration_map=None) == calibration_path.resolve()

    sequential = convert_hl320_bin_dir_to_csv(
        bin_dir=bin_dir,
        output_dir=tmp_path / "csv_seq",
        name_mode="sequential",
        overwrite=True,
    )
    assert [frame.output_csv.name for frame in sequential.frames] == ["000000.csv", "000001.csv"]
    assert sequential.manifest_path.is_file()

    stem = convert_hl320_bin_dir_to_csv(
        bin_dir=bin_dir,
        output_dir=tmp_path / "csv_stem",
        calibration_map=calibration_path,
        name_mode="stem",
        overwrite=True,
    )
    assert [frame.output_csv.name for frame in stem.frames] == ["a_frame.csv", "b_frame.csv"]


def test_build_hl320_dataset_cli_accepts_bin_dir(tmp_path: Path):
    root = tmp_path / "capture"
    bin_dir = root / "bin"
    labels_dir = tmp_path / "labels"
    bin_dir.mkdir(parents=True)
    labels_dir.mkdir()
    write_calibration(root / "calibration_map.txt")
    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  CAR: 2\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )

    for index in range(2):
        write_synthetic_bin(bin_dir / f"source_{index}.bin", width=2, height=2, base_x=10.0 + index)
        frame_label_dir = labels_dir / f"{index:06d}"
        frame_label_dir.mkdir()
        np.save(frame_label_dir / "semantic_mask.npy", np.asarray([2, 2, 255, 2], dtype=np.uint16))

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_hl320_dataset.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bin-dir",
            str(bin_dir),
            "--labels-dir",
            str(labels_dir),
            "--output-dir",
            str(tmp_path / "dataset_out"),
            "--classes-yaml",
            str(classes_yaml),
            "--val-ratio",
            "0.5",
            "--overwrite",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    out_dir = Path(payload["output_dir"])
    assert Path(payload["conversion_manifest"]).is_file()
    assert payload["csv_dir"].endswith("converted_csv")
    assert (out_dir / "converted_csv" / "000000.csv").is_file()
    assert (out_dir / "converted_csv" / "000001.csv").is_file()
    assert len(read_csv_rows(out_dir / "converted_csv" / "000000.csv")) == 12
    assert (out_dir / "hl320_dataset_manifest.json").is_file()
    dataset_frame_dirs = {path.name: path for path in (out_dir / "dataset").glob("*/*")}
    assert len(dataset_frame_dirs) == 2
    features = np.load(dataset_frame_dirs["000000"] / "features.npy")
    assert features.shape[0] == 4
    return_count_index = HL320_FEATURE_NAMES.index("return_count")
    np.testing.assert_allclose(features[:, return_count_index], np.full((4,), 3.0, dtype=np.float32))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_calibration(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    coefs = [[0, 0, 0, 0, 0, 0, 0, 123] for _ in range(24)]
    y_coefs = [[0, 0, 0, 0, 0, 0, 0, 456] for _ in range(24)]
    path.write_text(
        json.dumps({"calibration": {"HorizontalAngleCoef": coefs, "VerticalAngleCoef": y_coefs}}),
        encoding="utf-8",
    )
    return path


def write_synthetic_bin(path: Path, *, width: int, height: int, base_x: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bytearray(GZIP_HEADER_LEN + width * SLOT_DATA_SIZE)
    data[4:8] = int(width).to_bytes(4, "little")
    data[8:12] = int(height).to_bytes(4, "little")
    for slot_index in range(width):
        for pixel_index in range(height):
            ray_offset = GZIP_HEADER_LEN + pixel_index * BATCH_POINT_LEN * ECHO_COUNT + slot_index * SLOT_DATA_SIZE
            for echo_index in range(ECHO_COUNT):
                offset = ray_offset + echo_index * BATCH_POINT_LEN
                x = base_x + slot_index * 10.0 + pixel_index + echo_index * 0.25
                y = x + 0.1
                z = x + 0.2
                struct.pack_into("<f", data, offset, x)
                struct.pack_into("<f", data, offset + 4, y)
                struct.pack_into("<f", data, offset + 8, z)
                struct.pack_into("<f", data, offset + 16, 80.0 + slot_index)
                struct.pack_into("<f", data, offset + 20, 30.0 + pixel_index)
                data[offset + 24 : offset + 28] = int(2 + pixel_index + echo_index).to_bytes(4, "little")
                data[offset + 32 : offset + 36] = int(2 + slot_index).to_bytes(4, "little")
                data[offset + 36 : offset + 40] = int(pixel_index).to_bytes(4, "little")
    path.write_bytes(data)
