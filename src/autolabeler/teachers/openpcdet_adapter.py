from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

from ..actors.openpcdet_teacher import OPENPCDET_TEACHER_SOURCE, write_openpcdet_predictions_jsonl
from ..data.dataset_indexer import FrameRecord, build_dataset_index
from ..data.frame_loader import load_frame
from ..export.kitti_xml_exporter import export_openpcdet_records_as_kitti_xml


@dataclass(frozen=True)
class OpenPCDetPreparedFrame:
    frame_id: str
    source_path: str
    points_path: str


def prepare_openpcdet_points(
    input_dir: str,
    output_dir: str,
    *,
    input_format: str = "auto",
) -> list[OpenPCDetPreparedFrame]:
    records = build_dataset_index(input_dir, input_format=input_format)
    points_dir = (Path(output_dir) / "points").resolve()
    points_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[OpenPCDetPreparedFrame] = []
    for record in records:
        frame = load_frame(record.lidar_path, record.frame_id, input_format=input_format)
        points = np.asarray(frame.points_flat, dtype=np.float32)
        finite = np.isfinite(points[:, :3]).all(axis=1)
        nonzero = ~(
            (np.abs(points[:, 0]) < 1e-4)
            & (np.abs(points[:, 1]) < 1e-4)
            & (np.abs(points[:, 2]) < 1e-4)
        )
        points = points[finite & nonzero]
        out_path = points_dir / f"{record.frame_id}.npy"
        np.save(out_path, points.astype(np.float32))
        prepared.append(OpenPCDetPreparedFrame(record.frame_id, str(Path(record.lidar_path).resolve()), str(out_path)))

    return prepared


def run_openpcdet_inference(
    *,
    openpcdet_root: str,
    cfg_file: str,
    ckpt: str,
    prepared_frames: list[OpenPCDetPreparedFrame],
    output_jsonl: str,
    score_threshold: float = 0.15,
    kitti_output_dir: str | None = None,
    confidence_log_path: str | None = None,
) -> list[dict]:
    _add_openpcdet_to_path(openpcdet_root)

    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    class PreparedNpyDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )
            self.sample_file_list = [Path(x.points_path).resolve() for x in prepared_frames]

        def __len__(self):
            return len(self.sample_file_list)

        def __getitem__(self, index):
            points = np.load(self.sample_file_list[index]).astype(np.float32)
            points = _match_point_feature_dim(points, self.dataset_cfg)
            input_dict = {"points": points, "frame_id": prepared_frames[index].frame_id}
            return self.prepare_data(data_dict=input_dict)

    old_cwd = os.getcwd()
    try:
        os.chdir(Path(openpcdet_root) / "tools")
        cfg_from_yaml_file(_resolve_cfg_file(openpcdet_root, cfg_file), cfg)
        logger = common_utils.create_logger()
        dataset = PreparedNpyDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(openpcdet_root),
            logger=logger,
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        records: list[dict] = []
        with torch.no_grad():
            for idx in range(len(dataset)):
                data_dict = dataset.collate_batch([dataset[idx]])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                pred = pred_dicts[0]
                boxes = pred["pred_boxes"].detach().cpu().numpy()
                scores = pred["pred_scores"].detach().cpu().numpy()
                labels = pred["pred_labels"].detach().cpu().numpy()

                predictions = []
                for box, score, label_id in zip(boxes, scores, labels):
                    score = float(score)
                    if score < score_threshold:
                        continue
                    class_name = str(cfg.CLASS_NAMES[int(label_id) - 1])
                    predictions.append(
                        {
                            "class_name": class_name,
                            "score": score,
                            "box_3d": {
                                "center": [float(box[0]), float(box[1]), float(box[2])],
                                "size": [float(box[3]), float(box[4]), float(box[5])],
                                "yaw": float(box[6]),
                            },
                        }
                    )

                records.append(
                    {
                        "frame_id": prepared_frames[idx].frame_id,
                        "source_path": prepared_frames[idx].source_path,
                        "source": OPENPCDET_TEACHER_SOURCE,
                        "predictions": predictions,
                    }
                )
    finally:
        os.chdir(old_cwd)

    write_openpcdet_predictions_jsonl(output_jsonl, records)
    if kitti_output_dir:
        export_openpcdet_records_as_kitti_xml(kitti_output_dir, records, confidence_log_path=confidence_log_path)
    return records


def _add_openpcdet_to_path(openpcdet_root: str) -> None:
    root = Path(openpcdet_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"OpenPCDet root does not exist: {root}")
    sys.path.insert(0, str(root))


def _resolve_cfg_file(openpcdet_root: str, cfg_file: str) -> str:
    path = Path(cfg_file)
    if path.is_absolute():
        return str(path)
    root = Path(openpcdet_root)
    if (root / cfg_file).exists():
        return str(root / cfg_file)
    return str(root / "tools" / cfg_file)


def _match_point_feature_dim(points: np.ndarray, dataset_cfg) -> np.ndarray:
    expected_dim = _expected_point_feature_dim(dataset_cfg)
    if expected_dim is None or points.shape[-1] == expected_dim:
        return points.astype(np.float32, copy=False)
    if points.shape[-1] < expected_dim:
        pad = np.zeros((points.shape[0], expected_dim - points.shape[-1]), dtype=np.float32)
        return np.concatenate([points.astype(np.float32, copy=False), pad], axis=1)
    return points[:, :expected_dim].astype(np.float32, copy=False)


def _expected_point_feature_dim(dataset_cfg) -> int | None:
    point_encoding = _cfg_get(dataset_cfg, "POINT_FEATURE_ENCODING")
    if point_encoding is None:
        return None
    src_feature_list = _cfg_get(point_encoding, "src_feature_list")
    if src_feature_list is None:
        return None
    return len(src_feature_list)


def _cfg_get(cfg_obj, key: str):
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key)
    return getattr(cfg_obj, key, None)
