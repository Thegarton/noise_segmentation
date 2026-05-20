from pathlib import Path
import csv

import numpy as np

from autolabeler.data.csv_loader import load_csv
from autolabeler.data.bin_loader import H, W


def test_load_csv_organized(tmp_path: Path):
    csv_path = tmp_path / "frame_0001.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("x,y,z,intensity,timestamp_s,timestamp_u,background_light_intensity\n")
        for i in range(H * W):
            f.write(f"{i*0.1},{i*0.2},{i*0.3},1.0,10,20,123.4\n")

    frame = load_csv(str(csv_path), frame_id="frame_0001")
    assert frame.points_range.shape == (H, W, 4)
    assert frame.points_flat.shape == (H * W, 4)
    assert frame.timestamp_s == 10
    assert frame.timestamp_u == 20
    assert frame.timestamp_us == 10_000_020
    assert np.isclose(frame.background_light_intensity, 123.4)


def test_load_raw_packet_csv_preserves_beam_azimuth_layout(tmp_path: Path):
    csv_path = tmp_path / "raw_packet.csv"
    fieldnames = [
        "packSeqNum",
        "timeStampData_s",
        "timeStampData_u",
        *[f"dis{i}" for i in range(H)],
        *[f"int{i}" for i in range(H)],
        *[f"ref{i}" for i in range(H)],
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for azimuth_idx in range(W):
            row = {"packSeqNum": azimuth_idx, "timeStampData_s": 10, "timeStampData_u": 20}
            for beam_idx in range(H):
                row[f"dis{beam_idx}"] = 10.0 + beam_idx * 0.01 + azimuth_idx * 0.001
                row[f"int{beam_idx}"] = 50.0 + beam_idx
                row[f"ref{beam_idx}"] = 100.0 + beam_idx * 10.0 + azimuth_idx
            writer.writerow(row)

    frame = load_csv(str(csv_path), frame_id="raw_packet")

    assert frame.points_range.shape == (H, W, 4)
    assert frame.points_flat.shape == (H * W, 4)
    assert frame.timestamp_s == 10
    assert frame.timestamp_u == 20
    assert frame.timestamp_us == 10_000_020
    assert frame.meta["csv_layout"] == "raw_packet"
    assert frame.meta["strength_channel"] == "ref"
    assert np.isclose(frame.points_range[5, 0, 3], 150.0)
    assert np.isclose(frame.points_range[5, 7, 3], 157.0)
