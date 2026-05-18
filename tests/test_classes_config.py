from autolabeler.data.class_config import load_noise_groups, load_semantic_classes


def test_semantic_class_ids_are_unique_and_complete():
    classes = load_semantic_classes("configs/classes.yaml")

    assert len(set(classes.values())) == len(classes)
    assert classes["ignore"] == 255
    for name in (
        "TRUCK_BUS",
        "CAR",
        "triangular_traffic_sign",
        "traffic_cone",
        "crosstalk_noise_1",
        "horizontal_crosstalk_noise",
        "dust_noise",
        "near_range_layered_noise",
        "unknown_noise",
        "unknown_object",
    ):
        assert name in classes


def test_noise_groups_reference_known_classes():
    classes = load_semantic_classes("configs/classes.yaml")
    groups = load_noise_groups("configs/classes.yaml")

    assert groups["crosstalk_noise"] == [
        "crosstalk_noise_1",
        "crosstalk_noise_2",
        "horizontal_crosstalk_noise",
        "vertical_crosstalk_noise",
        "lane_line_crosstalk_noise",
    ]
    for group_members in groups.values():
        for class_name in group_members:
            assert class_name in classes
