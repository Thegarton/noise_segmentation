# Offline LiDAR Auto-Labeling Factory (v0)

This repository implements an **offline, cascaded auto-labeling pipeline** for organized LiDAR frames (`192x480x4`) with actor, irregular-object, and noise branches.

## Status

- ✅ Offline v0 scaffold with runnable end-to-end pass
- ✅ Preserves range-view (`192x480`) and point-view mappings
- ✅ Uses masks instead of deleting points
- ✅ Actor v0 detects fixed-size classes with a range-view component proposal generator, fixed box decoder, temporal support, and simple tracking
- ✅ Actor v0.2 can consume OpenPCDet CenterPoint-PointPillar nuScenes predictions as a LiDAR teacher
- ✅ Exports JSONL with provenance, confidence, review status, and pseudo-label version

## Run

```bash
PYTHONPATH=src python scripts/run_full_autolabeling.py --input-dir ./data --output ./out/labels.jsonl
```

## OpenPCDet teacher

Recommended first teacher: OpenPCDet `CenterPoint-PointPillar` trained on nuScenes, because it covers car, truck, bus, motorcycle, bicycle, and pedestrian. KITTI PointPillars is front-view friendly but has fewer relevant classes.

Prepare the repository and checkpoint separately, then run:

```bash
PYTHONPATH=src python scripts/run_openpcdet_teacher.py \
  --input-dir ./data \
  --input-format csv \
  --openpcdet-root /path/to/OpenPCDet \
  --cfg-file /path/to/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml \
  --ckpt /path/to/cbgs_pp_centerpoint_nds6070.pth \
  --output ./out/openpcdet_predictions.jsonl
```

Then run the cascaded labeler with teacher predictions and optional manual KITTI tracklets in `./data/mask`:

```bash
PYTHONPATH=src python scripts/run_full_autolabeling.py \
  --input-dir ./data \
  --input-format csv \
  --openpcdet-predictions ./out/openpcdet_predictions.jsonl \
  --output ./out/labels.jsonl
```

## Point-wise segmentation path

The next pipeline version targets dense point-wise semantic segmentation instead of box-first labels.
The primary output is a `192x480` semantic mask aligned with the organized LiDAR grid, plus a flattened
`H*W` point-label view.

LitePT integration is intentionally isolated from the lightweight core package. Run it from the dedicated
conda environment:

```bash
conda activate LItePT
PYTHONPATH=src python scripts/run_litept_inference.py \
  --input-dir ./data \
  --input-format csv \
  --checkpoint /path/to/litept.ckpt \
  --output-dir ./out/litept \
  --config configs/classes.yaml
```

`scripts/run_litept_inference.py` is currently a stub for the next integration step. The semantic class ids
and noise groups live in `configs/classes.yaml`.
