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
  --litept-root ../LitePT \
  --input-dir ./data \
  --input-format csv \
  --litept-dataset nuscenes \
  --output-dir ./out/litept_nuscenes \
  --config configs/classes.yaml
```

`scripts/run_litept_inference.py` is currently a stub for the next integration step. The semantic class ids
and noise groups live in `configs/classes.yaml`.
If `./data/ins` exists, each frame is matched to the nearest INS row by timestamp and exported with
`pose.txt` in KITTI 3x4 pose format plus `ego_pose` metadata. Use `--ins-path /path/to/ins` to pass a
different INS file or `--skip-pose-export` to skip pose matching on repeated runs.
By default, NuScenes uses `../LitePT/configs/nuscenes/semseg-litept-small-v1m1.py` and
`../LitePT/pth/nuscenes/model_best.pth`. To run the Waymo preset, switch the dataset and output dir:

```bash
PYTHONPATH=src python scripts/run_litept_inference.py \
  --litept-root ../LitePT \
  --input-dir ./data \
  --input-format csv \
  --litept-dataset waymo \
  --output-dir ./out/litept_waymo \
  --config configs/classes.yaml
```

Use `--litept-config` or `--checkpoint` only when overriding the preset paths.

Use `--dry-run` first to validate the external LitePT repo path and frame discovery without importing LitePT:

```bash
PYTHONPATH=src python scripts/run_litept_inference.py \
  --litept-root ../LitePT \
  --input-dir ./data \
  --input-format csv \
  --litept-dataset waymo \
  --output-dir ./out/litept_waymo \
  --dry-run
```

## Camera/SAM3 teacher path

The camera teacher is an offline data-engine step. It is used to create better training labels; the
eventual student model still uses LiDAR only at inference time.

Install OpenCV support in the env used for video extraction and projection QA:

```bash
pip install -e ".[camera]"
```

Extract synced camera frames from a video. The script matches each LiDAR CSV timestamp to the nearest
row in the video timestamp CSV and writes `data/<frame_id>.jpg` plus `camera_frame_manifest.json`.
Frames farther than `--max-delta-ms` are marked unsynced and are not saved as hard matches.

```bash
PYTHONPATH=src python scripts/prepare_camera_frames.py \
  --lidar-dir ./data \
  --video ./camera.mp4 \
  --video-timestamps-csv ./video_timestamps.csv \
  --video-frame-index-col frame_index \
  --video-timestamp-col timestamp \
  --timestamp-unit us \
  --out-dir ./data
```

Check LiDAR-to-camera calibration before using SAM3 output. The projection script writes
`point_to_pixel.npy`, `point_camera_depth.npy`, and a `projection_overlay.jpg` for visual inspection.

```bash
PYTHONPATH=src python scripts/project_lidar_to_image.py \
  --csv ./data/000009.csv \
  --image ./data/000009.jpg \
  --calibration-json ./camera_calibration.json \
  --extrinsic-direction lidar_to_camera \
  --out-dir ./out/projection
```

Run SAM3 from its own Conda environment through a file-based text-prompt wrapper. The external SAM3
script must accept `--image`, `--prompts`, and `--output`, and write `.npz` files with
`masks: bool[N,H,W]`, `scores: float[N]`, `labels: str[N]`, and `prompts: str[N]`.

```bash
PYTHONPATH=src python scripts/run_sam3_text_teacher.py \
  --image-dir ./data \
  --prompt-config configs/sam3_text_prompts_pointwise_v1.yaml \
  --conda-env sam3 \
  --sam3-script /path/to/run_sam3_text.py \
  --out-dir ./out/sam3_text \
  --validate
```

Fuse Waymo LitePT masks and SAM3 camera candidates into project-v1 point-wise seed labels. Residual
points are not automatically converted to `noise`; ambiguous points stay background/ignore or go to review.

```bash
PYTHONPATH=src python scripts/fuse_pointwise_teachers.py \
  --litept-output-dir ./out/litept_waymo \
  --sam3-dir ./out/sam3_text \
  --projection-dir ./out/projection \
  --out-dir ./out/pointwise_teacher_v1
```

Build optional review masks for coarse `noise` labeling. These are candidate masks only, not hard labels:

```bash
PYTHONPATH=src python scripts/build_noise_review_candidates.py \
  --csv ./data/000009.csv \
  --semantic-mask ./out/pointwise_teacher_v1/000009/semantic_mask.npy \
  --confidence ./out/pointwise_teacher_v1/000009/confidence.npy \
  --out-dir ./out/noise_candidates
```

## Point-wise segmentation visualization

Install the optional PyVista viewer dependency:

```bash
pip install -e ".[visualization]"
```

Visualize LitePT semantic mask and confidence for an organized LiDAR CSV frame:

```bash
python3 scripts/visualize_point_segmentation.py \
  --points ./data/000009.csv \
  --semantic-mask ./out/litept/000009/semantic_mask.npy \
  --confidence ./out/litept/000009/confidence.npy \
  --allow-range-resize
```

For notebook-based inspection with fixed camera position, open:

```bash
jupyter notebook notebooks/visualize_point_segmentation.ipynb
```
