import numpy as np

from autolabeler.data.bin_loader import H, W, C
from autolabeler.data.schemas import OrganizedLiDARFrame, SequenceSample
from autolabeler.noise.noise_autolabeler import NoiseAutoLabeler


def test_noise_autolabeler_detects_horizontal_crosstalk_on_residual():
    arr = np.zeros((H, W, C), dtype=np.float32)
    row = 50
    cols = range(100, 112)
    for col in cols:
        arr[row, col] = [12.0 + 0.05 * (col - 100), 1.0, 0.2, 0.4]
    arr[row, 99] = [12.0, 1.0, 0.2, 0.4]

    residual = [False] * (H * W)
    for col in cols:
        residual[row * W + col] = True
    actor_removed_idx = row * W + 99

    labels = NoiseAutoLabeler().run(_sample(arr), residual)

    assert len(labels) == 1
    assert labels[0].semantic_class == "horizontal_crosstalk_noise"
    assert labels[0].branch_name == "noise"
    assert actor_removed_idx not in labels[0].point_indices
    assert labels[0].box_3d is not None
    assert labels[0].box_3d.box_type == "noise_adaptive_aabb"


def test_noise_autolabeler_detects_vertical_crosstalk_on_residual():
    arr = np.zeros((H, W, C), dtype=np.float32)
    col = 70
    rows = range(20, 32)
    for row in rows:
        arr[row, col] = [9.0, 0.5, -0.2 + 0.03 * (row - 20), 0.6]

    residual = [False] * (H * W)
    for row in rows:
        residual[row * W + col] = True

    labels = NoiseAutoLabeler().run(_sample(arr), residual)

    assert len(labels) == 1
    assert labels[0].semantic_class == "vertical_crosstalk_noise"
    assert labels[0].range_image_indices[0] == [20, col]


def _sample(arr: np.ndarray) -> SequenceSample:
    frame = OrganizedLiDARFrame(frame_id="frame_000", points_range=arr, points_flat=arr.reshape(-1, C))
    return SequenceSample(current=frame)
