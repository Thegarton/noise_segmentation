# HL320 Vehicle Orientation

This standalone project builds a SAM3 mask-based vehicle dataset, trains an
EfficientNet-B0 classifier with `front`, `rear`, and `other` classes, and can
route generic SAM3 `vehicle` instances to the stable point-wise taxonomy.

## Install

Install it inside the environment that already runs SAM3:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  pip install -e /home/a60116606/git_repo/noise_seg/pipeline_v0/vehicle_orientation
```

The project uses the local SAM3 checkout and local `sam3.1_multiplex.pt`; it
does not download SAM3 weights.

## Source Layout

The source root normally contains three folders:

```text
vehicle_orientation_sources/
  front/
  rear/
  other/
```

Every vehicle detected in an image inherits the label of its folder. Use
`--folder-map` when an existing folder has another name, for example
`rear=backlights_and_licence_plate`.

## Build Dataset

The builder loads SAM3.1 Multiplex once, then reuses it for every image and all
three prompts. Every input image is color-corrected, circularly masked, and
defished before detection. Source names are recorded in metadata and are not
drawn into classifier images.

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python vehicle_orientation/scripts/build_dataset.py \
  --source-root /data/new_HL320/classes \
  --folder-map rear=backlights_and_licence_plate \
  --out-dir ./output/vehicle_orientation_dataset \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --min-score 0.45 \
  --min-mask-size 900 \
  --overwrite
```

Default fisheye parameters match the current HL320 server preprocessing:
`3840x3060`, equal-area source model, cylindrical output, `fov=190`,
`pfov=140`, center `(960,750)`, radius `1068`, and Lanczos interpolation.
All parameters have CLI overrides.

Important outputs:

```text
<out-dir>/
  prepared/<class>/*.jpg
  samples/<class>/<sample-id>/rgb.png
  samples/<class>/<sample-id>/mask.png
  samples/<class>/<sample-id>/masked_rgb.png
  samples/<class>/<sample-id>/preview.jpg
  manifest.jsonl
  dataset_manifest.json
```

Duplicate detections from `vehicle`, `car`, and `passenger vehicle` are merged
with mask-IoU NMS. Splits are stratified `70/15/15`, and all crops from one
source image remain in the same split.

## Train

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

The classifier head is trained alone for three epochs, then the complete
ImageNet-pretrained EfficientNet-B0 is fine-tuned. The best checkpoint is
selected by validation macro-F1. Metrics include per-class precision, recall,
F1, and the confusion matrix.

## Use With SAM3

Add a transient entry to the active prompt config. It does not need an id in
`classes.yaml`:

```yaml
vehicle:
  - "vehicle"
  - "car"
  - "passenger vehicle"
```

The active semantic taxonomy must contain `front_of_vehicle: 9` and
`rear_of_vehicle: 10`.

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

`rear` becomes class `10`; `front` becomes class `9`. By the selected project
policy, `other`, low-confidence, and low-margin predictions also fall back to
`front_of_vehicle` while retaining the fallback reason and all three
probabilities in NPZ/JSON metadata. Without the checkpoint flag, the original
SAM3 runner behavior is unchanged.

For the complete camera taxonomy, add the same top-level `vehicle` entry to
your full prompt YAML and keep using the production classes YAML. The included
configs are sufficient for a vehicle-only smoke run.
