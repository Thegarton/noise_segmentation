#!/usr/bin/env python3
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
from autolabeler.review.review_queue import build_review_queue
from autolabeler.database.label_db import write_versioned_snapshot


def main() -> None:
    p = argparse.ArgumentParser(description="Offline v0.1 preprocessing and label preparation")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--snapshot", default="./out/pseudo_label_db_v0_1.json")
    p.add_argument("--input-format", choices=["auto", "bin", "csv"], default="auto")
    p.add_argument("--cache-bin-dir", default=None)
    args = p.parse_args()

    index = build_dataset_index(args.input_dir, input_format=args.input_format)
    windows = build_temporal_windows(index, k_past=2, k_future=2)

    actor = ActorAutoLabeler()
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
        masks = build_masks(len(cur.points_flat), [], [])
        irr_labels = irr.run(sample, masks["removed_by_actor"])
        noise_labels = noise.run(sample, masks["unexplained_residual"])
        final_labels = arbitrate(actor_labels + irr_labels + noise_labels)

        results.append(AutoLabelingResult(frame_id=cur.frame_id, labels=final_labels))
        all_labels.extend(final_labels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_jsonl(args.output, results)
    review_items = build_review_queue(all_labels)
    write_versioned_snapshot(args.snapshot, [x.to_jsonable() for x in results], version="v0.1.0")
    print(f"exported_frames={len(results)} review_items={len(review_items)}")


if __name__ == "__main__":
    main()
