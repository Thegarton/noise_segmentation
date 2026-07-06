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
conda activate litept
PYTHONPATH=src python scripts/run_litept_inference.py \
  --litept-root ../LitePT \
  --input-dir ./data \
  --input-format csv \
  --litept-dataset nuscenes \
  --output-dir ./out/litept_nuscenes \
  --config configs/classes.yaml
```

The semantic class ids and noise groups live in `configs/classes.yaml`.
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

## Fine-tune LitePT on point_labeler labels

First export the corrected masks from `point_labeler`:

```bash
python3 ../point_labeler/scripts/export_from_point_labeler.py \
  --labeler-dir /path/to/labeler_dataset \
  --out-dir /path/to/corrected_masks
```

The fine-tuning script reads the masks from that export and the matching XYZI point clouds and current
taxonomy from the original labeler dataset. Arbitrary class ids are converted to dense LitePT training ids;
`ignore`/id `255` and unknown ids are excluded from the loss. The mapping back to the original ids is written
to `taxonomy.json`.

Validate all frame pairs, taxonomy, shapes, and the deterministic random validation split without importing
PyTorch:

```bash
PYTHONPATH=src python scripts/finetune_litept.py \
  --litept-root ../LitePT \
  --export-dir /path/to/corrected_masks \
  --labeler-dir /path/to/labeler_dataset \
  --output-dir ./out/litept_custom \
  --dry-run
```

Run preparation and training inside the LitePT CUDA environment:

```bash
conda activate litept
PYTHONPATH=src python scripts/finetune_litept.py \
  --litept-root ../LitePT \
  --export-dir /path/to/corrected_masks \
  --labeler-dir /path/to/labeler_dataset \
  --output-dir ./out/litept_custom \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 4 \
  --num-gpus 1 \
  --class-weighting sqrt_inverse \
  --max-class-weight 10 \
  --noise-frame-repeat 4 \
  --force-torch-pointrope
```

By default the script loads `../LitePT/pth/waymo/model_best.pth`, removes only its old `seg_head`, initializes
a new head for the custom taxonomy, and fine-tunes the backbone at a lower learning rate. Use `--checkpoint`
to select another Waymo-compatible checkpoint, `--prepare-only` to stop before training, `--overwrite` to
replace a previous generated run, or `--resume` to continue from
`<output-dir>/experiment/model/model_last.pth`.

Frames are assigned using a reproducible class-aware random split according to `--val-ratio` (default `0.2`).
Every class with valid points is kept in at least one training frame; a class that occurs in only one frame
therefore keeps that frame in train. If the requested validation size would remove the last training example
of a class, the validation set is made smaller. `--seed 42` is the default, and changing `--seed` produces
another valid random split.

Classes with no valid training points are removed from the generated segmentation head and listed under
`excluded_classes` in `taxonomy.json` and `class_statistics.json`. Remaining classes use inverse-square-root
CrossEntropy weights computed from valid train points; `--max-class-weight` limits the weighting and
`--class-weighting none` disables it. Lovasz loss remains enabled without per-class weights.

Training frames containing an active class whose name includes `noise` are listed multiple times in
`dataset/train_oversampled.json`. `--noise-frame-repeat 4` means four total appearances per noise frame;
use `1` to disable this oversampling. Exact loss weights, repeated frames, and effective train sample count
are written to `class_statistics.json` and `run_manifest.json`.

These changes can alter both the split and the number of output classes. Start a fresh run with `--overwrite`;
do not use `--resume` with a checkpoint created from the previous taxonomy/head.

`--force-torch-pointrope` replaces LitePT's compiled PointROPE CUDA extension with its PyTorch implementation.
Use it when training fails on the first batch with `CUDA error: no kernel image is available for execution on
the device`. This fallback is slower, but does not require rebuilding PointROPE for the GPU's compute
capability.

Generated artifacts include:

- `dataset/{train,val}/<frame>/{coord.npy,strength.npy,segment.npy}`
- `dataset/train_oversampled.json`
- `litept_custom_config.py`
- `train_litept_custom.py`
- `pretrained_backbone.pth`
- `taxonomy.json`, `class_statistics.json`, and `run_manifest.json`
- `experiment/model/model_best.pth`

Run the fine-tuned model while preserving the original point_labeler ids:

```bash
PYTHONPATH=src python scripts/run_litept_inference.py \
  --litept-root ../LitePT \
  --input-dir /path/to/organized_frames \
  --input-format csv \
  --litept-dataset custom \
  --litept-config ./out/litept_custom/litept_custom_config.py \
  --checkpoint ./out/litept_custom/experiment/model/model_best.pth \
  --output-dir ./out/litept_custom_predictions
```

The custom config stores `training_id_to_source_id`; inference applies it before writing
`semantic_mask.npy` and includes exact `semantic_classes` in every `metadata.json`.

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

`project_lidar_to_image.py` expects a JSON camera calibration:

```json
{
  "image_size": [3840, 2160],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "T_lidar_to_camera": [[...], [...], [...], [0, 0, 0, 1]]
}
```

- `K` is the 3x3 intrinsic matrix for the image size in `image_size`.
- `image_size` is `[width, height]` for the frame used during calibration, for example 4K `3840x2160`.
- `T_lidar_to_camera` maps LiDAR xyz points into the camera coordinate frame. If your file contains
  `T_camera_to_lidar`, run with `--extrinsic-direction camera_to_lidar` and the script will invert it.
- The point_labeler RGB precompute uses a KITTI-style `rgb_calib*.txt` with `P2` and `Tr`; do not replace
  this JSON with the labeler dataset `calib.txt`, which is only for scan poses.

If the calibration was made for 4K but the synced frames are resized from the same camera, keep the original
`image_size` in the JSON or pass `--calibration-image-size 3840x2160`. The code scales `fx`, `cx` by
`actual_width / calibration_width` and `fy`, `cy` by `actual_height / calibration_height` before projection.
This handles compressed/resized images, including non-uniform resize. It does not compensate for crops,
letterboxing, or padding; those require adjusting the principal point before projection.

```bash
PYTHONPATH=src python scripts/project_lidar_to_image.py \
  --csv ./data/000009.csv \
  --image ./data/000009.jpg \
  --calibration-json ./camera_calibration.json \
  --calibration-image-size 3840x2160 \
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

For video-aware SAM3 pseudo-labels, run the video teacher from the main LitePT environment and let it
call SAM3 in its own Conda environment. `scripts/sam3_31_video_wrapper.py` follows the SAM3.1 notebook API:
`build_sam3_multiplex_video_predictor()`, `start_session`, text `add_prompt`, and `propagate_in_video`.

```bash
PYTHONPATH=src python scripts/run_sam3_video_teacher.py \
  --video ./camera.mp4 \
  --camera-frame-manifest ./data/camera_frame_manifest.json \
  --prompt-config configs/sam3_text_prompts_pointwise_v1.yaml \
  --classes-yaml configs/classes_pointwise_v1.yaml \
  --sam3-conda-env sam3 \
  --sam3-video-script scripts/sam3_31_video_wrapper.py \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --frame-batch-size 10 \
  --out-dir ./out/sam3_video_teacher \
  --validate
```

If the SAM3 environment must be selected by absolute path, replace `--sam3-conda-env sam3` with
`--sam3-conda-prefix /home/a60116606/miniconda3/envs/sam3`. `--sam3-root` points to the cloned SAM3 code,
while `--sam3-model-path` points to the local HuggingFace `facebook/sam3.1` directory containing
`config.json` and `sam3.1_multiplex.pt`; this avoids gated HuggingFace downloads at runtime. The wrapper accepts
an MP4 file or a directory of numbered JPEG frames. Outputs are written as `<out-dir>/<frame_id>/sam3_video.npz`,
`semantic_mask.npy`, `confidence.npy`, `metadata.json`, and, when the synced image exists, `image.jpg`/`overlay.jpg`.
Use `--max-frames N` for smoke tests when you intentionally want to process only the first `N` synced LiDAR
frames. For full runs on limited VRAM, keep `--max-frames` unset and use `--frame-batch-size N`
(`--sam3-frame-batch-size N` is an alias). The runner will still process every synced frame from
`camera_frame_manifest.json`, but it will call the SAM3 wrapper in separate sequential processes of at most `N`
frames each, for example 270 frames as 27 processes with `--frame-batch-size 10`. By default the SAM3.1 wrapper
builds a temporary numbered JPEG folder containing only those requested frames and passes that folder to
`start_session`, so the model does not decode/cache the full source video. If you need the old full-video behavior,
pass `--sam3-wrapper-arg=--use-original-video-resource`. FlashAttention 3 is disabled by default for GPU/dtype
compatibility; pass `--sam3-wrapper-arg=--use-fa3` only on a confirmed compatible SAM3 environment.

For a single-image, frame-by-frame SAM3 smoke test equivalent to `notebooks/sam3_single_image_prompts.ipynb`, run
the folder script inside the SAM3 environment:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 python scripts/run_sam3_single_image_folder.py \
  --image-dir ./output/2025_12_15_08_16_49_frames_270/img2 \
  --out-dir ./out/sam3_single_image_folder \
  --prompt-config configs/sam3_text_prompts_pointwise_v1.yaml \
  --classes-yaml configs/classes_pointwise_v1.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --projection-dir ./data/projection_shift_3_1090_1245 \
  --min-score 0.7 \
  --validate
```

Each image gets `<out-dir>/<image_stem>/semantic_mask.npy`, `confidence.npy`, `instances.npz`, `instances.json`,
`overlay.jpg`, `semantic_color.png`, `preview.jpg`, `image.jpg`, and `metadata.json`. If `--projection-dir` is set,
the script matches projections by file stem, for example `000000.jpg` with `<projection-dir>/000000.jpg`, and writes
`mask_projection.jpg` with semantic mask, overlay, and LiDAR projection side by side. Add `--require-projection` to
fail on missing projection files. The script builds the SAM3 predictor once, starts and closes a separate
single-image session per frame, and applies every text prompt from the prompt config.

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
