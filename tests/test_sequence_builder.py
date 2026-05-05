from autolabeler.data.dataset_indexer import FrameRecord
from autolabeler.data.sequence_builder import build_temporal_windows


def test_temporal_window_sizes():
    recs = [FrameRecord(frame_id=str(i), lidar_path=f"{i}.bin", timestamp=float(i)) for i in range(5)]
    windows = build_temporal_windows(recs, k_past=2, k_future=2)
    assert len(windows) == 5
    assert len(windows[0]["past"]) == 0
    assert len(windows[0]["future"]) == 2
    assert len(windows[2]["past"]) == 2
