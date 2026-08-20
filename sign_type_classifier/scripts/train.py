#!/usr/bin/env python3
"""Train numeric and EfficientNet-B0 sign-type classifiers with validation fusion."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sign_type_classifier.dataset import CLASSIFIER_CLASS_NAMES, load_jsonl  # noqa: E402
from sign_type_classifier.model import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    blend_probabilities,
    build_efficientnet_b0,
    build_numeric_mlp,
    compute_numeric_normalization,
    resolve_device,
    set_image_backbone_trainable,
    standardize_numeric_features,
)


@dataclass(frozen=True)
class EpochOutput:
    summary: dict[str, Any]
    targets: np.ndarray
    image_probabilities: np.ndarray
    numeric_probabilities: np.ndarray


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch, nn, DataLoader, Dataset, transforms = _import_training_dependencies()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, overwrite=args.overwrite)
    _seed_everything(args.seed, torch=torch)

    records = load_jsonl(manifest_path)
    dataset_root = manifest_path.parent
    feature_names = _load_feature_names(dataset_root)
    _validate_records(records, dataset_root=dataset_root, feature_names=feature_names)
    by_split = {
        split: [item for item in records if item.get("split") == split]
        for split in ("train", "val", "test")
    }
    split_sizes = {name: len(items) for name, items in by_split.items()}
    if any(size == 0 for size in split_sizes.values()):
        raise ValueError(f"Dataset must contain non-empty train/val/test splits, got {split_sizes}")

    train_counts = Counter(str(item["label"]) for item in by_split["train"])
    missing_classes = [name for name in CLASSIFIER_CLASS_NAMES if train_counts[name] == 0]
    if missing_classes:
        raise ValueError(
            "Training split has no samples for classes: "
            f"{missing_classes}. Move examples into those review folders and run reindex_dataset.py again."
        )

    train_numeric = _numeric_values(by_split["train"], feature_names=feature_names)
    numeric_mean, numeric_std = compute_numeric_normalization(train_numeric)
    train_dataset = SignTypeDataset(
        records=by_split["train"],
        dataset_root=dataset_root,
        feature_names=feature_names,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        transform=_build_transform(transforms, input_size=args.input_size, training=True),
        dataset_base=Dataset,
    )
    val_dataset = SignTypeDataset(
        records=by_split["val"],
        dataset_root=dataset_root,
        feature_names=feature_names,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        transform=_build_transform(transforms, input_size=args.input_size, training=False),
        dataset_base=Dataset,
    )
    test_dataset = SignTypeDataset(
        records=by_split["test"],
        dataset_root=dataset_root,
        feature_names=feature_names,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        transform=_build_transform(transforms, input_size=args.input_size, training=False),
        dataset_base=Dataset,
    )

    device = resolve_device(args.device)
    pin_memory = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    image_model = build_efficientnet_b0(
        pretrained=not args.no_pretrained,
        num_classes=len(CLASSIFIER_CLASS_NAMES),
    )
    numeric_model = build_numeric_mlp(
        num_features=len(feature_names),
        num_classes=len(CLASSIFIER_CLASS_NAMES),
        hidden_size=args.numeric_hidden_size,
        dropout=args.numeric_dropout,
    )
    set_image_backbone_trainable(image_model, False)
    image_model.to(device)
    numeric_model.to(device)

    class_weights = _class_weights(train_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": image_model.classifier.parameters(), "lr": args.head_lr},
            {"params": numeric_model.parameters(), "lr": args.numeric_lr},
        ],
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        if epoch == args.head_warmup_epochs:
            set_image_backbone_trainable(image_model, True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": image_model.features.parameters(), "lr": args.backbone_lr},
                    {"params": image_model.classifier.parameters(), "lr": args.finetune_head_lr},
                    {"params": numeric_model.parameters(), "lr": args.numeric_lr},
                ],
                weight_decay=args.weight_decay,
            )

        train_output = _run_epoch(
            image_model=image_model,
            numeric_model=numeric_model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            torch=torch,
            optimizer=optimizer,
            numeric_loss_weight=args.numeric_loss_weight,
        )
        val_output = _run_epoch(
            image_model=image_model,
            numeric_model=numeric_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            torch=torch,
            optimizer=None,
            numeric_loss_weight=args.numeric_loss_weight,
        )
        image_weight, val_ensemble = choose_fusion_weight(
            targets=val_output.targets,
            image_probabilities=val_output.image_probabilities,
            numeric_probabilities=val_output.numeric_probabilities,
            step=args.fusion_weight_step,
        )
        train_ensemble = _metrics_from_probabilities(
            train_output.targets,
            blend_probabilities(
                train_output.image_probabilities,
                train_output.numeric_probabilities,
                image_weight=image_weight,
            ),
        )
        epoch_record = {
            "epoch": epoch + 1,
            "fusion_image_weight": image_weight,
            "train": {**train_output.summary, "ensemble": train_ensemble},
            "val": {**val_output.summary, "ensemble": val_ensemble},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        checkpoint = _checkpoint_payload(
            image_model=image_model,
            numeric_model=numeric_model,
            optimizer=optimizer,
            epoch=epoch + 1,
            args=args,
            feature_names=feature_names,
            numeric_mean=numeric_mean,
            numeric_std=numeric_std,
            class_weights=class_weights,
            fusion_image_weight=image_weight,
            metrics=epoch_record,
        )
        torch.save(checkpoint, out_dir / "model_last.pth")
        current_f1 = float(val_ensemble["macro_f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_without_improvement = 0
            torch.save(checkpoint, out_dir / "model_best.pth")
        else:
            epochs_without_improvement += 1
        _write_json(out_dir / "history.json", history)
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break

    best = torch.load(out_dir / "model_best.pth", map_location="cpu", weights_only=False)
    image_model.load_state_dict(best["image_model_state_dict"], strict=True)
    numeric_model.load_state_dict(best["numeric_model_state_dict"], strict=True)
    image_model.to(device)
    numeric_model.to(device)
    test_output = _run_epoch(
        image_model=image_model,
        numeric_model=numeric_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        torch=torch,
        optimizer=None,
        numeric_loss_weight=args.numeric_loss_weight,
    )
    best_image_weight = float(best["fusion_image_weight"])
    test_ensemble = _metrics_from_probabilities(
        test_output.targets,
        blend_probabilities(
            test_output.image_probabilities,
            test_output.numeric_probabilities,
            image_weight=best_image_weight,
        ),
    )
    final_metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best_f1),
        "fusion_image_weight": best_image_weight,
        "fusion_numeric_weight": 1.0 - best_image_weight,
        "test": {**test_output.summary, "ensemble": test_ensemble},
        "class_names": list(CLASSIFIER_CLASS_NAMES),
        "train_class_counts": {name: int(train_counts[name]) for name in CLASSIFIER_CLASS_NAMES},
        "split_sizes": split_sizes,
        "class_weights": class_weights,
    }
    _write_json(out_dir / "metrics.json", final_metrics)
    _write_json(
        out_dir / "model_config.json",
        {
            "version": 1,
            "image_model": "efficientnet_b0",
            "numeric_model": "mlp",
            "class_names": list(CLASSIFIER_CLASS_NAMES),
            "numeric_feature_names": list(feature_names),
            "input_size": int(args.input_size),
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "numeric_hidden_size": int(args.numeric_hidden_size),
            "numeric_dropout": float(args.numeric_dropout),
            "fusion_image_weight": best_image_weight,
            "fusion_numeric_weight": 1.0 - best_image_weight,
            "image_input": "context_rgb",
            "numeric_input": "numeric_feature_vector",
        },
    )
    shutil.copy2(manifest_path, out_dir / "split_manifest.jsonl")
    print(json.dumps({"model_best": str(out_dir / "model_best.pth"), **final_metrics}, ensure_ascii=False, indent=2))


class SignTypeDataset:
    """Torch Dataset wrapper kept importable without torch at module import time."""

    def __new__(
        cls,
        *,
        records: list[dict[str, Any]],
        dataset_root: Path,
        feature_names: tuple[str, ...],
        numeric_mean: np.ndarray,
        numeric_std: np.ndarray,
        transform: Any,
        dataset_base: Any,
    ) -> Any:
        class _Dataset(dataset_base):
            def __init__(self) -> None:
                self.records = records
                self.dataset_root = dataset_root
                self.transform = transform
                self.class_to_id = {
                    name: index for index, name in enumerate(CLASSIFIER_CLASS_NAMES)
                }
                raw = _numeric_values(records, feature_names=feature_names)
                self.numeric = standardize_numeric_features(
                    raw,
                    mean=numeric_mean,
                    std=numeric_std,
                )

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, index: int) -> tuple[Any, np.ndarray, int]:
                from PIL import Image  # noqa: WPS433

                record = self.records[index]
                image_path = self.dataset_root / str(record["context_rgb"])
                image = Image.open(image_path).convert("RGB")
                target = self.class_to_id[str(record["label"])]
                return self.transform(image), self.numeric[index], target

        return _Dataset()


def choose_fusion_weight(
    *,
    targets: np.ndarray,
    image_probabilities: np.ndarray,
    numeric_probabilities: np.ndarray,
    step: float,
) -> tuple[float, dict[str, Any]]:
    count = int(round(1.0 / float(step)))
    candidates = np.linspace(0.0, 1.0, count + 1)
    scored = []
    for weight in candidates:
        probabilities = blend_probabilities(
            image_probabilities,
            numeric_probabilities,
            image_weight=float(weight),
        )
        metrics = _metrics_from_probabilities(targets, probabilities)
        scored.append((float(metrics["macro_f1"]), -abs(float(weight) - 0.5), float(weight), metrics))
    _, _, weight, metrics = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return weight, metrics


def _run_epoch(
    *,
    image_model: Any,
    numeric_model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    torch: Any,
    optimizer: Any | None,
    numeric_loss_weight: float,
) -> EpochOutput:
    training = optimizer is not None
    image_model.train(training)
    numeric_model.train(training)
    total_loss = 0.0
    total_image_loss = 0.0
    total_numeric_loss = 0.0
    total_items = 0
    all_targets = []
    all_image_probabilities = []
    all_numeric_probabilities = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, numeric, targets in loader:
            images = images.to(device, non_blocking=True)
            numeric = numeric.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            image_logits = image_model(images)
            numeric_logits = numeric_model(numeric)
            image_loss = criterion(image_logits, targets)
            numeric_loss = criterion(numeric_logits, targets)
            loss = image_loss + float(numeric_loss_weight) * numeric_loss
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(targets.shape[0])
            total_items += batch_size
            total_loss += float(loss.detach().cpu()) * batch_size
            total_image_loss += float(image_loss.detach().cpu()) * batch_size
            total_numeric_loss += float(numeric_loss.detach().cpu()) * batch_size
            all_targets.append(targets.detach().cpu().numpy())
            all_image_probabilities.append(torch.softmax(image_logits, dim=1).detach().cpu().numpy())
            all_numeric_probabilities.append(torch.softmax(numeric_logits, dim=1).detach().cpu().numpy())
    targets_array = np.concatenate(all_targets, axis=0)
    image_probabilities = np.concatenate(all_image_probabilities, axis=0)
    numeric_probabilities = np.concatenate(all_numeric_probabilities, axis=0)
    return EpochOutput(
        summary={
            "loss": total_loss / max(total_items, 1),
            "image_loss": total_image_loss / max(total_items, 1),
            "numeric_loss": total_numeric_loss / max(total_items, 1),
            "samples": total_items,
            "image": _metrics_from_probabilities(targets_array, image_probabilities),
            "numeric": _metrics_from_probabilities(targets_array, numeric_probabilities),
        },
        targets=targets_array,
        image_probabilities=image_probabilities,
        numeric_probabilities=numeric_probabilities,
    )


def _metrics_from_probabilities(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    target_values = np.asarray(targets, dtype=np.int64).reshape(-1)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.shape != (target_values.size, len(CLASSIFIER_CLASS_NAMES)):
        raise ValueError(
            f"Expected probabilities {(target_values.size, len(CLASSIFIER_CLASS_NAMES))}, "
            f"got {probability_values.shape}"
        )
    predictions = probability_values.argmax(axis=1)
    confusion = np.zeros(
        (len(CLASSIFIER_CLASS_NAMES), len(CLASSIFIER_CLASS_NAMES)),
        dtype=np.int64,
    )
    np.add.at(confusion, (target_values, predictions), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(CLASSIFIER_CLASS_NAMES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }
    return {
        "accuracy": float(np.trace(confusion)) / max(int(confusion.sum()), 1),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _numeric_values(records: list[dict[str, Any]], *, feature_names: tuple[str, ...]) -> np.ndarray:
    rows = []
    for record in records:
        vector = np.asarray(record.get("numeric_feature_vector", ()), dtype=np.float32).reshape(-1)
        if vector.shape != (len(feature_names),):
            raise ValueError(
                f"Sample {record.get('sample_id')!r} has numeric vector {vector.shape}; "
                f"expected {(len(feature_names),)}"
            )
        rows.append(vector)
    return np.stack(rows, axis=0)


def _load_feature_names(dataset_root: Path) -> tuple[str, ...]:
    path = dataset_root / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    class_names = tuple(value.get("class_names", ()))
    if class_names != CLASSIFIER_CLASS_NAMES:
        raise ValueError(
            f"Dataset classes are {class_names}; expected {CLASSIFIER_CLASS_NAMES}. "
            "Run reindex_dataset.py after reviewing not_a_sign."
        )
    names = tuple(str(item) for item in value.get("numeric_feature_names", ()))
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Dataset has invalid numeric_feature_names: {names}")
    return names


def _validate_records(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
    feature_names: tuple[str, ...],
) -> None:
    if not records:
        raise ValueError("Dataset manifest is empty")
    unknown = sorted({str(item.get("label")) for item in records} - set(CLASSIFIER_CLASS_NAMES))
    if unknown:
        raise ValueError(f"Manifest contains unknown classes: {unknown}")
    _numeric_values(records, feature_names=feature_names)
    missing_images = [
        str(item.get("context_rgb"))
        for item in records
        if not (dataset_root / str(item.get("context_rgb", ""))).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing context images, first entries: {missing_images[:10]}")


def _build_transform(transforms: Any, *, input_size: int, training: bool) -> Any:
    from PIL import Image, ImageOps  # noqa: WPS433

    def letterbox(image: Any) -> Any:
        contained = ImageOps.contain(image, (input_size, input_size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (input_size, input_size), color=(127, 127, 127))
        canvas.paste(contained, ((input_size - contained.width) // 2, (input_size - contained.height) // 2))
        return canvas

    operations: list[Any] = [transforms.Lambda(letterbox)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=3.0,
                    translate=(0.03, 0.03),
                    scale=(0.90, 1.10),
                    fill=127,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.15,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def _class_weights(counts: Counter[str]) -> list[float]:
    total = float(sum(counts.values()))
    weights = [
        total / (len(CLASSIFIER_CLASS_NAMES) * float(counts[name]))
        for name in CLASSIFIER_CLASS_NAMES
    ]
    mean = float(np.mean(weights))
    return [float(value / mean) for value in weights]


def _checkpoint_payload(
    *,
    image_model: Any,
    numeric_model: Any,
    optimizer: Any,
    epoch: int,
    args: argparse.Namespace,
    feature_names: tuple[str, ...],
    numeric_mean: np.ndarray,
    numeric_std: np.ndarray,
    class_weights: list[float],
    fusion_image_weight: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "image_model_name": "efficientnet_b0",
        "numeric_model_name": "mlp",
        "image_model_state_dict": image_model.state_dict(),
        "numeric_model_state_dict": numeric_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "class_names": CLASSIFIER_CLASS_NAMES,
        "numeric_feature_names": feature_names,
        "numeric_mean": numeric_mean.tolist(),
        "numeric_std": numeric_std.tolist(),
        "numeric_hidden_size": int(args.numeric_hidden_size),
        "numeric_dropout": float(args.numeric_dropout),
        "input_size": int(args.input_size),
        "class_weights": class_weights,
        "fusion_image_weight": float(fusion_image_weight),
        "metrics": metrics,
    }


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _import_training_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
        from torch.utils.data import DataLoader, Dataset  # noqa: WPS433
        from torchvision import transforms  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "Training requires torch, torchvision, and Pillow; install sign_type_classifier with its training extras"
        ) from exc
    return torch, nn, DataLoader, Dataset, transforms


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.input_size <= 0:
        raise ValueError("--epochs, --batch-size, and --input-size must be positive")
    if args.head_warmup_epochs < 0 or args.head_warmup_epochs >= args.epochs:
        raise ValueError("--head-warmup-epochs must be >=0 and smaller than --epochs")
    if args.numeric_hidden_size <= 0:
        raise ValueError("--numeric-hidden-size must be positive")
    if not 0.0 <= args.numeric_dropout < 1.0:
        raise ValueError("--numeric-dropout must be in [0,1)")
    if args.numeric_loss_weight <= 0.0:
        raise ValueError("--numeric-loss-weight must be positive")
    if not 0.0 < args.fusion_weight_step <= 1.0:
        raise ValueError("--fusion-weight-step must be in (0,1]")
    inverse = 1.0 / args.fusion_weight_step
    if not np.isclose(inverse, round(inverse)):
        raise ValueError("--fusion-weight-step must divide 1.0 exactly, e.g. 0.05 or 0.1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a seven-class sign ensemble: EfficientNet-B0 context branch plus numeric-feature MLP."
    )
    parser.add_argument("--manifest", required=True, help="Reviewed/merged dataset manifest.jsonl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--head-warmup-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-head-lr", type=float, default=2e-4)
    parser.add_argument("--numeric-lr", type=float, default=5e-4)
    parser.add_argument("--numeric-hidden-size", type=int, default=128)
    parser.add_argument("--numeric-dropout", type=float, default=0.20)
    parser.add_argument("--numeric-loss-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight-step", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet EfficientNet weights.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
