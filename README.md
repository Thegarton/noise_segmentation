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

For a circular fisheye camera, rectify frames before SAM3 with the OpenCV remapper. Output resolution and aspect ratio are independent from the source circle; the mapping changes the virtual camera FOV instead of geometrically stretching the circle into a rectangle:

```bash
PYTHONPATH=src python scripts/rectify_fisheye_opencv.py \
  --image /path/to/img2/000000.jpg \
  --output ./output/rectified/000000.jpg \
  --output-size 3840x2160 \
  --format circular \
  --dtype linear \
  --projection perspective \
  --fov 180 \
  --pfov 110 \
  --pfov-axis horizontal \
  --xcenter 960 \
  --ycenter 768 \
  --radius 768 \
  --interpolation lanczos \
  --save-map
```

Use `--auto-circle` instead of explicit center/radius only when the lens circle has a clean black border. `perspective` is the normal choice for SAM3 and other image models. `cylindrical` retains a wider horizontal view with less edge stretching, but its geometry is less similar to a conventional pinhole camera. Increasing `--output-size` improves sampling and downstream working resolution, but cannot restore detail absent from the source image.

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
  --min-score 0.63 \
  --label-min-scores configs/sam3_label_min_scores.yaml \
  --overwrite
```

`--min-score` is the fallback threshold. The optional per-label table changes the internal SAM3 detection, image-only and new-detection thresholds before each prompt. Labels absent from the table retain the global value:

```yaml
traffic_cone: 0.30
roadblock: 0.35
tire: 0.35
traffic_sign: 0.45
```

Label names must exactly match the top-level keys in the active prompt config. Per-label thresholds and the effective fallback-expanded table are saved in every frame's `metadata.json` and in the run manifest. Use `--overwrite` or a new output directory when changing thresholds, otherwise existing predictions are kept.

### Vehicle front/rear classifier

The standalone [`vehicle_orientation`](vehicle_orientation/README.md) project accepts one mixed camera-image folder, uses SAM3 to cut out every car and roughly sort crops into `review/front`, `review/rear`, and `review/side`, then rebuilds labels after manual file moves. It trains an ImageNet-pretrained EfficientNet-B0 on the reviewed `front/rear/side` dataset and optionally routes car masks to `front_of_vehicle`, `rear_of_vehicle`, or `side_of_vehicle` inside the folder runner. The output ids are read from the active classes YAML. Install it in the SAM3 environment and pass `--vehicle-orientation-checkpoint`; without that flag, the existing SAM3 behavior is unchanged.

The prompt config may contain a transient label which is intentionally absent from `classes.yaml`:

```yaml
vehicle:
  - "passenger car"
  - "sport utility vehicle"
  - "van"
  - "pickup truck"
  - "truck"
  - "bus"
```

To bypass EfficientNet and classify orientation directly with SAM3 text
prompts, pass `--sam3-only`. In this mode every top-level prompt label must
exist in the active classes YAML, for example:

```yaml
front_of_vehicle:
  - "front view of a four-wheeled car or truck"
rear_of_vehicle:
  - "rear view of a four-wheeled car or truck"
side_of_vehicle:
  - "side view of a four-wheeled car or truck"
```

`--sam3-only` overrides `--vehicle-orientation-checkpoint`, so EfficientNet is
not loaded even if a wrapper also supplies a checkpoint.

After all prompts have run, masks are deduplicated independently for every
label. For masks of the same label, the highest-confidence mask is kept and
lower-confidence masks whose Mask IoU exceeds
`--mask-nms-iou` (default `0.8`) are removed. The older
`--vehicle-orientation-nms-iou` spelling remains available as an alias. Masks belonging to
different labels are not suppressed by this step. This applies both with
EfficientNet and in `--sam3-only` mode.

Optional normalized-box geometry filters are configured in
`GEOMETRY_FILTER_RULES` inside `run_sam3_single_image_folder.py`. Each entry is
keyed by the exact semantic label and defines `min_y_bottom` and
`min_aspect_ratio`. Size filters are configured separately in
`SIZE_FILTER_RULES`; optional `min/max_box_width`, `min/max_box_height`, and
`min/max_box_area` values reject detections whose normalized SAM3 box
`[x, y, width, height]` is too small or too large. Labels absent from a table
bypass that filter.

Use `--log-json` to save `classes_log.json` for every frame. Its `instances`
array contains only objects that own pixels in the final overlay after geometry,
size, priority, confidence, and overlap filtering. Each entry includes
`visible_pixel_count` for the final non-overlapped mask fragment.

The positive whitelist intentionally avoids the broad `vehicle` prompt because it also covers bicycles and motorcycles. The runner deduplicates the car/truck/bus masks before one batched classifier call per image. Orientation probabilities, confidence, margin, and fallback reason are saved in `instances.npz`, `instances.json`, and frame metadata. Both `license_plate_and_taillights` and the older `license_plate&taillights` spelling remain priority labels and cannot be overwritten by the vehicle mask.

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

6. Import the new external HL320 manual JSON format when labels come from the new annotation tool.

In this format `indices` are original CSV row numbers, including all echo-channel rows. The importer writes both training labels and a point_labeler dataset:

```bash
PYTHONPATH=src python scripts/import_hl320_manual_json_labels.py \
  --csv-dir /path/to/csv_all_echo \
  --annotation-dir /path/to/RoadTest/label \
  --out-dir ./output/HL320_manual_json_import \
  --classes-yaml configs/classes.yaml \
  --image-dir /path/to/img2 \
  --overwrite
```

Important outputs:

- `manual_labels/<frame>/semantic_mask.npy`: one label per original CSV row, including echo rows;
- `manual_labels/<frame>/primary_semantic_mask.npy`: convenience mask for `blockID == 0`;
- `point_labeler/velodyne/<frame>.bin` and `point_labeler/labels/<frame>.label`: point_labeler-compatible flat semantic labels;
- `point_labeler/labels.xml`, `settings.cfg`, `bridge_manifest.json`.

For custom object names from the annotation tool, pass mappings like:

```bash
--class-map "串扰=crosstalk_noise_1" \
--class-map "路牌=traffic_sign"
```

7. Build the clean HL320 dataset format for the next model stage:

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

For the new all-echo manual JSON labels, use the imported labels and train on every echo row:

```bash
PYTHONPATH=src python scripts/build_hl320_dataset.py \
  --csv-dir /path/to/csv_all_echo \
  --labels-dir ./output/HL320_manual_json_import/manual_labels \
  --classes-yaml configs/classes.yaml \
  --output-dir ./output/HL320_dataset_all_echo_v1 \
  --return-mode all \
  --val-ratio 0.2 \
  --seed 42 \
  --overwrite
```

Without `--return-mode all`, the builder keeps the older primary-only behavior and selects only `blockID == 0` rows.

8. Train LitePT from scratch on the clean HL320 dataset:

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

9. Fine-tune LitePT as the older transition baseline when you specifically want to reuse the Waymo backbone:

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

10. Run a trained LitePT model on flat HL320 CSV frames.

For the new from-scratch HL320 model:

```bash
PYTHONPATH=src /home/a60116606/miniconda3/envs/litept/bin/python scripts/run_litept_flat_csv_inference.py \
  --litept-root /home/a60116606/git_repo/LitePT \
  --csv-dir /path/to/csv_shift_3_1090_1245 \
  --output-dir ./output/HL320_litept_from_scratch_inference \
  --checkpoint ./output/HL320_litept_from_scratch/experiment/model/model_best.pth \
  --litept-config ./output/HL320_litept_from_scratch/hl320_litept_config.py \
  --feature-mode hl320 \
  --return-mode all \
  --device cuda:0 \
  --force-torch-pointrope
```

Use `--return-mode all` for models trained from `build_hl320_dataset.py --return-mode all`; otherwise the inference output will be primary-only.

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

11. Fuse LitePT point predictions with SAM3 image predictions.

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
