#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from autolabeler.data.schemas import SequenceSample, AutoLabelingResult
from autolabeler.data.dataset_indexer import build_dataset_index
from autolabeler.data.sequence_builder import build_temporal_windows
from autolabeler.data.frame_loader import load_frame
from autolabeler.actors.actor_autolabeler import ActorAutoLabeler
from autolabeler.irregular.irregular_autolabeler import IrregularAutoLabeler
from autolabeler.noise.noise_autolabeler import NoiseAutoLabeler
from autolabeler.residual.residual_builder import build_masks
from autolabeler.fusion.arbiter import arbitrate
from autolabeler.export.jsonl_exporter import export_jsonl
from autolabeler.export.kitti_xml_exporter import export_kitti_xml
from autolabeler.review.review_queue import build_review_queue
from autolabeler.database.label_db import write_versioned_snapshot
from autolabeler.data.kitti_mask_loader import build_manual_actor_labels, build_manual_labels, densify_actor_label_masks, densify_label_masks


def main() -> None:
    p = argparse.ArgumentParser(description="Offline v0.1 preprocessing and label preparation")
    p.add_argument("--input-dir", default="./data")
    p.add_argument("--output", required=True)
    p.add_argument("--snapshot", default="./out/pseudo_label_db_v0_1.json")
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--cache-bin-dir", default=None)
    p.add_argument("--mask-dir", default=None, help="Path to dataset/mask with frame_list.txt and tracklet_labels.xml")
    p.add_argument("--noise-mask-dir", default=None, help="Path to KITTI XML manual noise masks with frame_list.txt and tracklet_labels.xml")
    p.add_argument("--openpcdet-predictions", default=None, help="JSONL predictions produced by scripts/run_openpcdet_teacher.py")
    p.add_argument("--use-actor-heuristic", action="store_true", help="Also run the range-view heuristic actor fallback")
    p.add_argument("--kitti-output-dir", default=None, help="Optional output dir for KITTI XML export of final boxed labels")
    args = p.parse_args()

    index = build_dataset_index(args.input_dir, input_format=args.input_format)
    windows = build_temporal_windows(index, k_past=2, k_future=2)

    mask_dir = args.mask_dir
    default_mask_dir = Path(args.input_dir) / "mask"
    if mask_dir is None and default_mask_dir.exists():
        mask_dir = str(default_mask_dir)
    manual_by_frame = build_manual_actor_labels(mask_dir) if mask_dir else {}
    manual_noise_by_frame = build_manual_labels(args.noise_mask_dir, branch_name="noise") if args.noise_mask_dir else {}

    actor = ActorAutoLabeler(
        openpcdet_predictions_path=args.openpcdet_predictions,
        pseudo_label_version="v0.2_openpcdet_teacher" if args.openpcdet_predictions else "v0_actor_heuristic",
        use_heuristic_fallback=should_use_actor_heuristic(
            openpcdet_predictions=args.openpcdet_predictions,
            use_actor_heuristic=args.use_actor_heuristic,
        ),
    )
    irr = IrregularAutoLabeler()
    noise = NoiseAutoLabeler()

    results = []
    all_labels = []
    for w in windows:
        cur = load_frame(w["current"].lidar_path, w["current"].frame_id, input_format=args.input_format, cache_bin_dir=args.cache_bin_dir)
        past = [load_frame(r.lidar_path, r.frame_id, input_format=args.input_format, cache_bin_dir=args.cache_bin_dir) for r in w["past"]]
        future = [load_frame(r.lidar_path, r.frame_id, input_format=args.input_format, cache_bin_dir=args.cache_bin_dir) for r in w["future"]]
        sample = SequenceSample(current=cur, past=past, future=future)

        actor_labels = actor.run(sample)
        manual_labels = densify_actor_label_masks(manual_by_frame.get(cur.frame_id, []), cur.points_flat)
        actor_labels = manual_labels + actor_labels
        actor_indices = sorted({i for label in actor_labels for i in label.point_indices})
        actor_masks = build_masks(len(cur.points_flat), actor_indices, [])
        irr_labels = irr.run(sample, actor_masks["removed_by_actor"])
        irregular_indices = sorted({i for label in irr_labels for i in label.point_indices})
        masks = build_masks(len(cur.points_flat), actor_indices, irregular_indices)
        manual_noise_labels = densify_label_masks(
            manual_noise_by_frame.get(cur.frame_id, []),
            cur.points_flat,
            allowed_mask=masks["unexplained_residual"],
        )
        manual_noise_indices = {i for label in manual_noise_labels for i in label.point_indices}
        noise_residual = [is_residual and i not in manual_noise_indices for i, is_residual in enumerate(masks["unexplained_residual"])]
        noise_labels = manual_noise_labels + noise.run(sample, noise_residual)
        final_labels = arbitrate(actor_labels + irr_labels + noise_labels)

        results.append(AutoLabelingResult(frame_id=cur.frame_id, labels=final_labels))
        all_labels.extend(final_labels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_jsonl(args.output, results)
    if args.kitti_output_dir:
        export_kitti_xml(args.kitti_output_dir, results, actor_only=False)
    review_items = build_review_queue(all_labels)
    write_versioned_snapshot(args.snapshot, [x.to_jsonable() for x in results], version="v0.1.0")
    print(f"exported_frames={len(results)} review_items={len(review_items)}")


def should_use_actor_heuristic(*, openpcdet_predictions: str | None, use_actor_heuristic: bool) -> bool:
    return use_actor_heuristic or openpcdet_predictions is None


if __name__ == "__main__":
    main()
