# HL320 Noise Segmentation Toolkit

This repository contains the current HL320 point-wise segmentation workflow. It is focused on flat CSV/LiDAR point clouds, camera masks from SAM3, point_labeler manual correction, and LitePT-based transition experiments.

The old box-first actor pipeline has been removed. There is no OpenPCDet teacher, fixed 3D boxes, KITTI XML label export, or `192x480` residual cascade in the active code path.

## Current Workflow

1. Prepare camera frames and synchronization metadata when the source is video:

```bash
PYTHONPATH=src python scripts/prepare_camera_frames.py \
  --video /path/to/video.mp4 \
  --lidar-dir /path/to/csv_or_lidar_frames \
  --out-dir ./output/camera_frames
```

2. Run SAM3 as a camera teacher. For image folders, use the single-image runner:

```bash
PYTHONPATH=src conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python scripts/run_sam3_single_image_folder.py \
  --image-dir /path/to/img2 \
  --projection-dir /path/to/projection_images \
  --out-dir ./output/sam3_single_image_folder \
  --prompt-config configs/sam3_text_prompts_pointwise_v1.yaml \
  --classes-yaml configs/classes_pointwise_v1.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --min-score 0.63
```

For video-context SAM3 runs, use `scripts/run_sam3_video_teacher.py`. The main environment does not import SAM3 directly; SAM3 should run through its own Conda environment.

3. Build HL320 physics teacher candidates.

```bash
PYTHONPATH=src python scripts/build_hl320_teacher_candidates.py \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --classes-yaml configs/classes.yaml \
  --sam3-dir ./output/sam3_single_image_folder \
  --out-dir ./output/HL320_teacher_candidates \
  --overwrite
```

This stage is a candidate teacher, not final labeling. It preserves CSV row order, groups multi-echo returns by `slot + pixel`, scores LiDAR physics rules, uses neighboring frames for temporal support, and uses SAM3 only as camera object context. Per frame it writes `candidate_mask.npy`, `candidate_confidence.npy`, `candidate_reasons.json`, `features_debug.npz`, and `metadata.json`; the top-level `teacher_candidates/<frame>.npy` mirrors the candidate mask for review tools.

4. Lift SAM3 masks to LiDAR point labels through the existing `Cxd/Cyd` columns:

```bash
PYTHONPATH=src python scripts/project_sam3_masks_to_point_labeler.py \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --sam3-dir ./output/sam3_single_image_folder \
  --image-dir /path/to/img2 \
  --out-labeler-dir ./output/HL320_point_labeler_seed \
  --classes-yaml configs/classes_pointwise_v1.yaml \
  --min-confidence 0.7 \
  --overwrite
```

This creates a point_labeler-compatible dataset with:

- `velodyne/<frame>.bin` as flat `N x 4` XYZI;
- `labels/<frame>.label` as `uint32` point labels;
- optional `point_rgb/<frame>.rgb`;
- optional `image_2/<frame>.jpg`;
- `labels.xml`, `settings.cfg`, and `bridge_manifest.json`.

5. Correct labels manually in point_labeler, then export them:

```bash
cd /home/a60116606/git_repo/point_labeler/scripts

python3 export_from_point_labeler.py \
  --labeler-dir /home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104 \
  --out-dir /home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104/export
```

Flat HL320 exports use `semantic_mask.npy` with shape `[N]`. The point index is the original CSV row index after invalid points are handled by downstream training.

6. Build the clean HL320 dataset format for the next model stage:

```bash
PYTHONPATH=src python scripts/build_hl320_dataset.py \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --labels-dir /home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104/export \
  --classes-yaml configs/classes.yaml \
  --output-dir ./output/HL320_dataset_v1 \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite
```

This writes `dataset/{train,val}/<frame>/coord.npy`, `features.npy`, `strength.npy`, `segment.npy`, and `metadata.json`.
`features.npy` is the explicit HL320 feature matrix. `strength.npy` contains the same matrix because LitePT `DefaultDataset` only loads known asset names.
`segment.npy` uses dense training ids `0..N-1`; original source ids are preserved in `hl320_dataset_manifest.json`.

7. Train LitePT from scratch on the clean HL320 dataset:

```bash
PYTHONPATH=src /home/a60116606/miniconda3/envs/litept/bin/python scripts/train_hl320_litept_from_scratch.py \
  --litept-root /home/a60116606/git_repo/LitePT \
  --dataset-root ./output/HL320_dataset_v1 \
  --output-dir ./output/HL320_litept_from_scratch \
  --epochs 100 \
  --batch-size 2 \
  --num-workers 4 \
  --num-gpus 1 \
  --grid-size 0.05 \
  --lr 0.002 \
  --class-weighting sqrt_inverse \
  --max-class-weight 10 \
  --noise-frame-repeat 4 \
  --force-torch-pointrope \
  --overwrite
```

Use `--dry-run` to validate the dataset and config plan without importing PyTorch/CUDA. Use `--prepare-only` to write the LitePT config and manifests without starting training.
This path does not load a Waymo/NuScenes checkpoint. The LitePT head and backbone are initialized from scratch, and the generated config sets `backbone.in_channels = 3 + len(HL320 features)`.

8. Fine-tune LitePT as the older transition baseline when you specifically want to reuse the Waymo backbone:

```bash
PYTHONPATH=src /home/a60116606/miniconda3/envs/litept/bin/python scripts/finetune_litept.py \
  --litept-root /home/a60116606/git_repo/LitePT \
  --export-dir /home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104/export \
  --labeler-dir /home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104 \
  --output-dir ./output/HL320_output_sam3_manual_104/fine_tune_100 \
  --epochs 100 \
  --val-ratio 0.20 \
  --batch-size 2 \
  --num-workers 4 \
  --class-weighting sqrt_inverse \
  --max-class-weight 10 \
  --noise-frame-repeat 4 \
  --overwrite \
  --seed 42 \
  --num-gpus 1 \
  --force-torch-pointrope
```

9. Run a trained LitePT model on flat HL320 CSV frames.

For the new from-scratch HL320 model:

```bash
PYTHONPATH=src /home/a60116606/miniconda3/envs/litept/bin/python scripts/run_litept_flat_csv_inference.py \
  --litept-root /home/a60116606/git_repo/LitePT \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --output-dir ./output/HL320_litept_from_scratch_inference \
  --checkpoint ./output/HL320_litept_from_scratch/experiment/model/model_best.pth \
  --litept-config ./output/HL320_litept_from_scratch/hl320_litept_config.py \
  --feature-mode hl320 \
  --device cuda:0 \
  --force-torch-pointrope
```

For the older Waymo fine-tune transition baseline:

```bash
PYTHONPATH=src /home/a60116606/miniconda3/envs/litept/bin/python scripts/run_litept_flat_csv_inference.py \
  --litept-root /home/a60116606/git_repo/LitePT \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --output-dir ./output/litept_flat_csv_inference \
  --fine-tune-dir ./output/HL320_output_sam3_manual_104/fine_tune_100 \
  --device cuda:0 \
  --force-torch-pointrope
```

10. Fuse LitePT point predictions with SAM3 image predictions.

```bash
PYTHONPATH=src python scripts/fuse_hl320_point_predictions.py \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --litept-dir ./output/HL320_litept_from_scratch_inference \
  --sam3-dir ./output/sam3_single_image_folder \
  --classes-yaml configs/classes.yaml \
  --out-dir ./output/HL320_fused_predictions \
  --min-sam3-confidence 0.7 \
  --litept-keep-threshold 0.6 \
  --noise-protect-threshold 0.4 \
  --sam3-override-margin 0.2 \
  --overwrite
```

Fusion is done at point level. SAM3 is lifted to each LiDAR row through `Cxd/Cyd`; no reordering by `slot` or `pixel` is used. A confident LitePT noise label is protected from camera overwrite. SAM3 fills ignored or weak LitePT points and can override non-noise labels only when its confidence is higher by the configured margin.

## Visualization

Create videos from SAM3 image outputs:

```bash
PYTHONPATH=src python scripts/make_sam3_overlay_video.py \
  --sam3-dir ./output/sam3_single_image_folder \
  --out-video ./output/sam3_overlay.mp4 \
  --fps 10
```

Create videos from point_labeler export projections:

```bash
PYTHONPATH=src python scripts/make_point_labeler_projection_video.py \
  --export-dir /path/to/point_labeler/export \
  --labeler-dir /path/to/point_labeler/dataset \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --image-dir /path/to/img2 \
  --out-video ./output/point_labeler_projection.mp4 \
  --fps 10
```

Create videos from LitePT flat CSV inference:

```bash
PYTHONPATH=src python scripts/make_litept_inference_projection_video.py \
  --inference-dir ./output/litept_flat_csv_inference \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --image-dir /path/to/img2 \
  --out-video ./output/litept_inference_projection.mp4 \
  --fps 10
```

The same visualization script can be pointed at `./output/HL320_fused_predictions`, because fused output uses the same per-frame `semantic_mask.npy` and `confidence.npy` layout.

## HL320 Data Contract

The active data contract is flat point-wise HL320 data. A CSV frame should contain at least:

```text
x y z azimuth vertical intensity slot pixel hcell vcell Cxd Cyd
```

The important invariants are:

- point order is preserved from CSV row order;
- `Cxd/Cyd` are camera image coordinates used for projection and mask lifting;
- `slot` and `pixel` are metadata, not image axes;
- background and ignore are distinct labels;
- `ignore=255` is excluded from loss in training.

## Current HL320 Model Direction

The HL320-specific path lives under `src/autolabeler/hl320/` and trains from scratch rather than relying on Waymo/NuScenes checkpoints. The dataset representation is:

- `coord.npy` for XYZ;
- `features.npy` for intensity, reflectivity, distance, echo metadata, and multi-echo relations;
- `segment.npy` for dense point labels;
- `metadata.json` for frame-level provenance and CSV/raw source paths.

SAM3 remains a camera teacher and candidate source. It must not overwrite LiDAR noise classes by itself; the physics teacher adds transparent noise candidates with reasons and temporal support, and the fusion stage protects confident LiDAR noise labels from camera overwrite.
