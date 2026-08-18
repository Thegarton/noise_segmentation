# HL320 Vehicle Orientation

This standalone project turns one mixed folder of camera images into a reviewed
vehicle orientation dataset. SAM3.1 detects every visible vehicle, makes a
full-mask crop, and roughly sorts the crops into `front`, `rear`, and `side`.
After manual folder correction, EfficientNet-B0 is trained on the corrected
labels and can help the main SAM3 pipeline distinguish vehicle fronts and rears.

## Install

Install the project inside the environment that already runs SAM3:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  pip install -e /home/a60116606/git_repo/noise_seg/pipeline_v0/vehicle_orientation
```

The local SAM3 checkout and local `sam3.1_multiplex.pt` are used. The builder
does not download model weights.

SAM3 visual-backbone features are computed once per image and reused for all
prompts. This cache is enabled by default; use `--no-cache-visual-features`
only for debugging or compatibility checks.

## Input

`--source-root` is one directory of mixed images. A frame may contain many cars
at different orientations. It must not be pre-sorted into class folders:

```text
vehicle_images/
  000000.jpg
  000001.jpg
  000002.jpg
```

Subdirectories are scanned recursively by default. Use `--non-recursive` to
read only files directly inside `--source-root`.

## Build And Auto-Sort

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python vehicle_orientation/scripts/build_dataset.py \
  --source-root /data/new_HL320/camera_images \
  --out-dir ./output/vehicle_orientation_dataset \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --min-score 0.45 \
  --min-mask-size 900 \
  --overwrite
```

The model is loaded once and reused for every image. A positive whitelist of
`passenger car`, `SUV`, `van`, `pickup truck`, `truck`, and `bus` prompts
detects the full vehicle mask. The broad `vehicle` prompt is deliberately not
used because it also selects bicycles and motorcycles. Orientation prompts are
matched to a confirmed whitelist mask and only choose the initial
`front/rear/side` folder; an orientation-only detection is rejected. Duplicate
generic detections are removed with mask-IoU NMS. Vehicles without a reliable
orientation match go to `side`.

Prompts are configured in
`configs/vehicle_dataset_prompts.yaml`. Default fisheye processing matches the
current HL320 setup: `3840x3060`, equal-area source model, cylindrical output,
`fov=190`, `pfov=140`, center `(960,750)`, radius `1068`, and Lanczos
interpolation. All camera parameters have CLI overrides.

The source-image mask can use different radii above and below its center. For
example, keep the upper half at radius `860` and crop the lower half more:

```bash
--mask-upper-radius 860 \
--mask-lower-radius 720
```

`--mask-radius` remains an alias for the upper radius. When
`--mask-lower-radius` is omitted, both halves use the same radius as before.

Important outputs:

```text
<out-dir>/
  prepared/*.jpg
  samples/<sample-id>/rgb.png
  samples/<sample-id>/mask.png
  samples/<sample-id>/masked_rgb.png
  samples/<sample-id>/preview.jpg
  review/front/*.png
  review/rear/*.png
  review/side/*.png
  manifest.jsonl
  dataset_manifest.json
```

## Manual Review

Open the three `review` folders and move incorrectly sorted PNG files to the
correct folder. A bad detection can be deleted. Do not rename retained files.
During reindexing, a missing review PNG is removed from `manifest.jsonl` and
therefore is not used for training. Its stable assets remain under
`samples/<sample-id>` unless `--prune-removed` is explicitly passed.

Check the result without writing changes:

```bash
python vehicle_orientation/scripts/reindex_dataset.py \
  --dataset-dir ./output/vehicle_orientation_dataset \
  --dry-run
```

Then apply the reviewed folders to `manifest.jsonl` and regenerate the split:

```bash
python vehicle_orientation/scripts/reindex_dataset.py \
  --dataset-dir ./output/vehicle_orientation_dataset \
  --seed 42
```

To also remove the stable assets of deleted review images:

```bash
python vehicle_orientation/scripts/reindex_dataset.py \
  --dataset-dir ./output/vehicle_orientation_dataset \
  --prune-removed
```

The split is group-aware `70/15/15`: all vehicle crops from one original frame
remain in the same train, validation, or test split even when that frame
contains both front and rear views.

## Merge Datasets

Pass `--dataset-dir` multiple times to combine reviewed datasets. A namespace
is added to sample and source ids, so identical image names from different
folders do not collide:

```bash
python vehicle_orientation/scripts/reindex_dataset.py \
  --dataset-dir ./output/vehicle_dataset_parking_a \
  --dataset-dir ./output/vehicle_dataset_parking_b \
  --dataset-dir ./output/vehicle_dataset_road \
  --out-dir ./output/vehicle_orientation_dataset_merged \
  --seed 42 \
  --overwrite
```

The input datasets are not modified. The merged directory receives copied
`samples`, reviewed images, a combined `manifest.jsonl`, and a new group-aware
split. The merged result can itself be reviewed and reindexed later using a
single `--dataset-dir`.

## Train

Run training only after `reindex_dataset.py`:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python vehicle_orientation/scripts/train.py \
  --manifest ./output/vehicle_orientation_dataset/manifest.jsonl \
  --out-dir ./output/vehicle_orientation_efficientnet_b0 \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --overwrite
```

The classifier head is trained for three epochs, then the full ImageNet-
pretrained EfficientNet-B0 is fine-tuned. `model_best.pth` is selected by
validation macro-F1. Metrics include per-class precision, recall, F1, and a
confusion matrix.

## Generate More Training Data With EfficientNet

After the first classifier is trained, use it instead of SAM3 orientation
prompts to bootstrap a larger dataset. SAM3 receives only the four-or-more-wheel
positive whitelist and produces full vehicle masks. EfficientNet independently
assigns each masked crop to `front`, `rear`, or `side`:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python vehicle_orientation/scripts/generate_dataset_with_efficientnet.py \
  --image-dir /data/new_HL320/more_camera_images \
  --out-dir ./output/vehicle_orientation_round_2 \
  --checkpoint ./output/vehicle_orientation_efficientnet_b0/model_best.pth \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --min-score 0.45 \
  --min-mask-size 900 \
  --device cuda \
  --overwrite
```

The script performs the same fisheye preprocessing, loads SAM3 and EfficientNet
once, batches all vehicle crops from a frame through EfficientNet, and writes:

- `review/front`, `review/rear`, and `review/side` for manual correction;
- `annotated/*.jpg` with class names and classifier confidence;
- all three probabilities, top-1 margin, SAM3 score, and timing in metadata;
- `classifier_needs_review=true` when confidence or top-1 margin is low.

Low-confidence samples still go to their EfficientNet top-1 class so the output
always has exactly three review folders. Move or delete wrong samples, run
`reindex_dataset.py`, and merge the corrected round with previous datasets.

### Recover Or Resume After A Crash

The generator now atomically updates `manifest.jsonl` and
`generation_state.json` after every completed source frame. If CUDA OOM or
another error stops the process, restart the same command with `--resume` and
without `--overwrite`:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python vehicle_orientation/scripts/generate_dataset_with_efficientnet.py \
  --image-dir /data/new_HL320/more_camera_images \
  --out-dir ./output/vehicle_orientation_round_2 \
  --checkpoint ./output/vehicle_orientation_efficientnet_b0/model_best.pth \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --device cuda \
  --resume
```

For an output created by the older version, where reviewed crops exist but
`manifest.jsonl` was never written, recover it without loading SAM3 or CUDA:

```bash
python vehicle_orientation/scripts/recover_generated_dataset.py \
  --image-dir /data/new_HL320/more_camera_images \
  --dataset-dir ./output/vehicle_orientation_round_2
```

Recovery uses the current locations of files in `review/front`, `review/rear`,
and `review/side`; manually deleted files stay excluded. It can reconstruct all
fields required for training and merging. Old per-instance SAM3 confidence,
EfficientNet probabilities, bounding boxes, and crop coordinates cannot be
recovered because the interrupted version never wrote them to disk. After
recovery, the same generation command can also be continued with `--resume`.

## Use With SAM3

The production prompt config may contain a transient label absent from the
semantic class YAML:

```yaml
vehicle:
  - "passenger car"
  - "sport utility vehicle"
  - "van"
  - "pickup truck"
  - "truck"
  - "bus"
```

The active semantic taxonomy must contain `front_of_vehicle`,
`rear_of_vehicle`, and `side_of_vehicle`. Their numeric ids are read from the
active classes YAML, so both a compact camera-only taxonomy such as `1/2/3`
and a larger project taxonomy such as `9/10/33` are supported.

```bash
PYTHONPATH=src conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python scripts/run_sam3_single_image_folder.py \
  --image-dir /path/to/defisheye_images \
  --out-dir ./output/sam3_with_vehicle_orientation \
  --prompt-config vehicle_orientation/configs/vehicle_prompts.yaml \
  --classes-yaml vehicle_orientation/configs/classes_pointwise_vehicle_orientation.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --vehicle-orientation-checkpoint ./output/vehicle_orientation_efficientnet_b0/model_best.pth \
  --vehicle-orientation-device cuda \
  --vehicle-orientation-min-confidence 0.70 \
  --vehicle-orientation-min-margin 0.10 \
  --overwrite \
  --validate
```

`front`, `rear`, and `side` are mapped to the ids assigned to the corresponding
classes in the active classes YAML.
Low-confidence and low-margin decisions keep their top-1 semantic class and are
marked as uncertain in NPZ/JSON metadata; they are no longer relabeled as
`front_of_vehicle`. Without the checkpoint flag, the existing SAM3 behavior is
unchanged.

For direct SAM3 orientation prompts without EfficientNet, use `--sam3-only`
and name the prompt-config sections `front_of_vehicle`, `rear_of_vehicle`, and
`side_of_vehicle`. Those names must match the active classes YAML. A single
transient `vehicle` section cannot produce three orientation classes without
the classifier.
