from autolabeler.data.dataset_indexer import build_dataset_index


def test_build_dataset_index_auto(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / "b.csv").write_text("x,y,z,intensity\n", encoding="utf-8")
    out = build_dataset_index(str(tmp_path), input_format="auto")
    assert len(out) == 2
