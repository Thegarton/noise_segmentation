from pathlib import Path

import numpy as np

from autolabeler.data.bin_loader import H, W, C
from autolabeler.data.kitti_mask_loader import build_manual_actor_labels, densify_actor_label_masks


def test_build_manual_actor_labels(tmp_path: Path):
    mask_dir = tmp_path / "dataset" / "mask"
    mask_dir.mkdir(parents=True)
    (mask_dir / "frame_list.txt").write_text("frame_0000\nframe_0001\n", encoding="utf-8")
    (mask_dir / "tracklet_labels.xml").write_text(
        """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<boost_serialization>
  <tracklets>
    <count>1</count>
    <item_version>1</item_version>
    <item>
      <objectType>car</objectType>
      <h>1.6</h><w>1.8</w><l>4.2</l>
      <first_frame>1</first_frame>
      <poses><count>1</count><item_version>0</item_version>
        <item><tx>10.0</tx><ty>2.0</ty><tz>0.3</tz><rz>0.1</rz></item>
      </poses>
    </item>
  </tracklets>
</boost_serialization>
""",
        encoding="utf-8",
    )

    by_frame = build_manual_actor_labels(str(mask_dir))
    assert len(by_frame["frame_0001"]) == 1
    label = by_frame["frame_0001"][0]
    assert label.semantic_class == "CAR"
    assert label.box_3d is not None
    assert label.box_3d.size == [4.2, 1.8, 1.6]


def test_manual_actor_labels_are_densified_from_box_masks(tmp_path: Path):
    mask_dir = tmp_path / "dataset" / "mask"
    mask_dir.mkdir(parents=True)
    (mask_dir / "frame_list.txt").write_text("frame_0000\n", encoding="utf-8")
    (mask_dir / "tracklet_labels.xml").write_text(
        """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<boost_serialization>
  <tracklets>
    <count>1</count>
    <item_version>1</item_version>
    <item>
      <objectType>car</objectType>
      <h>1.6</h><w>1.8</w><l>4.2</l>
      <first_frame>0</first_frame>
      <poses><count>1</count><item_version>0</item_version>
        <item><tx>10.0</tx><ty>2.0</ty><tz>0.3</tz><rz>0.0</rz></item>
      </poses>
    </item>
  </tracklets>
</boost_serialization>
""",
        encoding="utf-8",
    )
    arr = np.zeros((H, W, C), dtype=np.float32)
    arr[0, 10] = [10.0, 2.0, 0.3, 1.0]
    arr[0, 11] = [30.0, 2.0, 0.3, 1.0]

    labels = build_manual_actor_labels(str(mask_dir))["frame_0000"]
    densify_actor_label_masks(labels, arr.reshape(-1, C))

    assert labels[0].point_indices == [10]
    assert labels[0].range_image_indices == [[0, 10]]
