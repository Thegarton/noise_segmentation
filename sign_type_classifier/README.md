# HL320 Sign-Type Dataset

This standalone project uses one SAM3.1 Multiplex instance to create a manually
reviewable dataset for six visually similar sign and clearance-bar classes. The
input directory must already contain prepared or defisheye images; this project
does not alter camera geometry.

The project builds the reviewed dataset and trains a two-branch classifier.
EfficientNet-B0 reads the context crop, while a small MLP reads position,
geometry, mask, colour, edge, and SAM3-score features. Their probabilities are
combined with a weight selected on the validation split.

## Classes

The classifier class order is fixed:

1. `height_restriction_sign_at_underground`
2. `height_restriction_barrel`
3. `underground_parking_sign`
4. `overhead_traffic_sign`
5. `side_plate`
6. `induction_sign`
7. `not_a_sign` (false-positive SAM3 detections)

Prompts live in `configs/sign_type_prompts.yaml`. The first six prompt groups
must be present and non-empty. `not_a_sign` has no prompt; it is populated by
manual review.

## Install

Install the project in the Conda environment that already runs SAM3:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  pip install -e /home/a60116606/git_repo/noise_seg/pipeline_v0/sign_type_classifier
```

## Build And Auto-Sort

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python sign_type_classifier/scripts/build_dataset.py \
  --source-root /path/to/prepared_images \
  --out-dir ./output/sign_type_dataset_round_1 \
  --prompt-config sign_type_classifier/configs/sign_type_prompts.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --min-score 0.45 \
  --min-mask-size 900 \
  --overwrite
```

SAM3 is loaded once. All class-specific prompts are run for each image. Masks
are treated as the same physical object when their Mask IoU is at least `0.55`
or their containment overlap is at least `0.80`. The highest per-class SAM3
score selects the initial review folder. There is no `unknown` folder.

SAM3 visual-backbone features are computed once per image and reused for all
sign prompts. This cache is enabled by default; use
`--no-cache-visual-features` to disable it for debugging.

For source images that have not had colour correction, add
`--colour-correction`. The builder applies the same OpenCV `SimpleWB(P=0.5)`
and gray-world correction as the main HL320 pipeline, then passes the corrected
RGB frame directly to SAM3 in memory. It does not save a corrected full-frame
image; the copied source image remains unchanged.

Label-specific normalized XYWH geometry filters run before cross-prompt mask
clustering. They are enabled by default so rejected detections cannot affect
the class score vector. Use `--no-geometry-filters` for comparison runs. Per
frame rejection details are stored in `dataset_manifest.json` under `sources`.

The builder atomically checkpoints `manifest.jsonl` and
`generation_state.json` after every completed source frame. Resume an
interrupted run with the same parameters:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python sign_type_classifier/scripts/build_dataset.py \
  --source-root /path/to/prepared_images \
  --out-dir ./output/sign_type_dataset_round_1 \
  --prompt-config sign_type_classifier/configs/sign_type_prompts.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --resume
```

Changing prompts, SAM3 model path, thresholds, clustering, or crop parameters
causes resume validation to fail instead of mixing incompatible samples.

## Bootstrap New Rounds With The Trained Classifier

After training the numeric + EfficientNet ensemble, pass its checkpoint back
to the dataset builder:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python sign_type_classifier/scripts/build_dataset.py \
  --source-root /path/to/new_prepared_images \
  --out-dir ./output/sign_type_dataset_round_9 \
  --prompt-config sign_type_classifier/configs/sign_type_prompts.yaml \
  --sam3-root /home/a60116606/git_repo/sam3 \
  --sam3-model-path /home/a60116606/git_repo/sam3/sam3.1 \
  --classifier-checkpoint ./output/sign_type_classifier_ensemble/model_best.pth \
  --classifier-device cpu \
  --classifier-min-confidence 0.50 \
  --classifier-min-margin 0.05 \
  --min-score 0.45 \
  --min-mask-size 900 \
  --overwrite
```

SAM3 still detects and segments all candidate masks. The trained classifier is
loaded once and classifies all candidates from one frame as a batch. A
prediction is used as the review-folder label only when both confidence and
top-1 margin pass the configured thresholds. Otherwise the sample falls back
to the original SAM3 top-1 folder. Set both thresholds to `0` to always accept
the classifier top-1 prediction.

The classifier defaults to `--classifier-device cpu` so SAM3 keeps the GPU
memory. The
EfficientNet-B0 and numeric branch are small, so CPU classification is normally
minor compared with SAM3 prompt inference. Use `cuda` or `cuda:0` when enough
VRAM is available.

Every sample keeps the following audit trail in `metadata.json` and
`manifest.jsonl`:

- `initial_label`: original SAM3 prompt class;
- `label`: final folder selected by the classifier or fallback;
- `classifier_prediction`: ensemble, image, and numeric probabilities;
- `classifier_accepted` and `classifier_fallback_reason`.

High-confidence `not_a_sign` predictions are preserved in
`review/not_a_sign`, allowing them to remain useful hard negatives after manual
review. Resume state includes the classifier checkpoint SHA256 and thresholds,
so a run cannot accidentally continue with different model weights.

## Output

```text
<out-dir>/
  images/<source-id>.jpg
  samples/<sample-id>/rgb.png
  samples/<sample-id>/mask.png
  samples/<sample-id>/masked_rgb.png
  samples/<sample-id>/context_rgb.png
  samples/<sample-id>/context_mask.png
  samples/<sample-id>/preview.jpg
  samples/<sample-id>/metadata.json
  review/<class>/<sample-id>.jpg
  manifest.jsonl
  dataset_manifest.json
  generation_state.json
```

`review/*.jpg` is a presentation image containing the full-frame location,
context crop, mask overlay, initial class, score, and margin. It is never used
as a training input. Clean model inputs are stored separately under `samples`.

The metadata contains normalized geometry, mask shape features, all six SAM3
scores, color histograms, brightness statistics, edge density, and an ordered
`numeric_feature_vector`. Position is measured in the full prepared image, so
cropping does not discard it.

## Manual Review

Move incorrect review JPGs into the correct class folder. Move false-positive
detections into `review/not_a_sign`; they are useful negative examples for the
classifier. Delete a file only when it must be excluded from training entirely.
Do not rename retained files.

Preview changes without writing:

```bash
python sign_type_classifier/scripts/reindex_dataset.py \
  --dataset-dir ./output/sign_type_dataset_round_1 \
  --dry-run
```

Apply labels and regenerate the source-frame-aware `70/15/15` split:

```bash
python sign_type_classifier/scripts/reindex_dataset.py \
  --dataset-dir ./output/sign_type_dataset_round_1 \
  --seed 42
```

Use `--prune-removed` to also delete `samples/<id>` assets for deleted review
JPGs. All samples produced from one source frame remain in the same split.

## Merge Reviewed Rounds

```bash
python sign_type_classifier/scripts/reindex_dataset.py \
  --dataset-dir ./output/sign_type_dataset_round_1 \
  --dataset-dir ./output/sign_type_dataset_round_2 \
  --out-dir ./output/sign_type_dataset_merged \
  --seed 42 \
  --overwrite
```

Input datasets are not modified. Sample and source ids are namespaced before
copying, so identical image names from different rounds do not collide.

## Train Numeric + EfficientNet Ensemble

Install the training dependencies in the SAM3 environment:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  pip install -e './sign_type_classifier[training]'
```

After manually reviewing and merging the datasets, train the ensemble:

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python sign_type_classifier/scripts/train.py \
  --manifest ./output/sign_type_dataset_merged/manifest.jsonl \
  --out-dir ./output/sign_type_classifier_ensemble \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda \
  --overwrite
```

Run `reindex_dataset.py` before training. The training split must contain all
six sign classes and `not_a_sign`. All crops from one source frame stay in the
same split, preventing nearly identical signs from leaking between train and
validation.

The image branch receives `context_rgb.png`, not the tight masked crop, so it
can distinguish overhead, underground, and roadside signs from their scene
context. The numeric MLP receives the stored `numeric_feature_vector`, which
contains normalized bbox position and size, mask geometry, object/context
colour histograms, brightness, edge density, and all original SAM3 class
scores.

For the first three epochs only the EfficientNet classifier head and numeric
MLP are trained. The EfficientNet backbone is then unfrozen. Both branches use
class-weighted cross entropy. After each epoch the script searches for the
image/numeric probability weight that gives the best validation macro-F1.

Outputs:

```text
<out-dir>/
  model_best.pth
  model_last.pth
  model_config.json
  metrics.json
  history.json
  split_manifest.jsonl
```

`metrics.json` reports precision, recall, F1, and confusion matrices separately
for `image`, `numeric`, and `ensemble`. This makes it possible to see whether
position/geometry really improves a particular sign class.

## Evaluate A Checkpoint

```bash
conda run -p /home/a60116606/miniconda3/envs/sam3 \
  python sign_type_classifier/scripts/predict.py \
  --manifest ./output/sign_type_dataset_merged/manifest.jsonl \
  --checkpoint ./output/sign_type_classifier_ensemble/model_best.pth \
  --output-jsonl ./output/sign_type_classifier_ensemble/test_predictions.jsonl \
  --split test \
  --device cuda
```

The prediction JSONL includes ensemble, EfficientNet, and numeric probabilities
for every sample. A neighboring `.metrics.json` contains aggregate test
metrics.

## Use In The SAM3 Production Runner

Pass the trained checkpoint to the main wrapper:

```bash
python scripts/run_full_sam3_masking.py \
  ... \
  --sign-classifier-checkpoint ./output/sign_type_classifier_ensemble/model_best.pth \
  --sign-classifier-device cpu \
  --sign-classifier-min-confidence 0.50 \
  --sign-classifier-min-margin 0.05
```

SAM3 still produces the sign masks and six prompt-score inputs. The ensemble
only selects the final sign class. Accepted `not_a_sign` predictions are
removed; uncertain predictions keep SAM3 top-1. CPU is the default device so
the sign classifier does not compete with SAM3 and vehicle EfficientNet for GPU
memory. Use `--sign-classifier-device cuda` when enough VRAM is available.

## Feature Analysis Notebook

Open `notebooks/sign_type_feature_analysis.ipynb` after reindexing or merging
the reviewed datasets. Set `DATASET_DIR` in its configuration cell. The
notebook visualizes class/split balance, source-frame counts, bbox geometry,
absolute image position, mask shape, object/context colour, edge density, SAM3
scores, per-feature Fisher separation, PCA, class-centroid distances, and the
initial-SAM3-versus-manual confusion matrix. It also exports compact CSV reports
to `<dataset-dir>/feature_analysis`.

Optional notebook dependencies can be installed with:

```bash
pip install -e './sign_type_classifier[analysis]'
```

After training, point `PREDICTIONS_JSONL` to the output of `predict.py` to
compare normalized confusion matrices for the EfficientNet branch, numeric
branch, and final ensemble.
