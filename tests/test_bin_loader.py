import numpy as np
from autolabeler.data.bin_loader import H, W, C


def test_shape_constants():
    assert (H, W, C) == (192, 480, 4)
