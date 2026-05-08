from pathlib import Path

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
