from pathlib import Path

import pytest


FIXED_CSV_PATH = Path(__file__).parent / "data" / "sample.csv"
FIXED_DIR_PATH = Path(__file__).parent / "data_seq"
FIXED_MASK_PATH = Path(__file__).parent / "mask"
FIXED_FRAME_LIST_PATH = Path(__file__).parent / "data" / "frame_list.txt"


@pytest.fixture(scope="session")
def fixed_csv_path() -> Path:
    if not FIXED_CSV_PATH.exists():
        pytest.skip("The sample file doesn't exist")
    return FIXED_CSV_PATH


@pytest.fixture(scope="session")
def fixed_dir_path() -> Path:
    if not FIXED_DIR_PATH.exists():
        pytest.skip("The dir for sequence test builder doesn't exist")
    return FIXED_DIR_PATH


@pytest.fixture(scope="session")
def fixed_mask_path() -> Path:
    if not FIXED_MASK_PATH.exists():
        pytest.skip("The kitti mask folder doesn't exist")
    return FIXED_MASK_PATH


@pytest.fixture(scope="session")
def fixed_frame_list_path() -> Path:
    if not FIXED_FRAME_LIST_PATH.exists():
        pytest.skip("The frame list file doesn't exist")
    return FIXED_FRAME_LIST_PATH
