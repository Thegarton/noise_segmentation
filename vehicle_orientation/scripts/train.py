#!/usr/bin/env python3
"""Train an EfficientNet-B0 front/rear/side classifier."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vehicle_orientation.dataset import load_jsonl  # noqa: E402
from vehicle_orientation.model import (  # noqa: E402
    CLASS_NAMES,
    build_efficientnet_b0,
    resolve_device,
    set_backbone_trainable,
)
from vehicle_orientation.preprocessing import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, Dataset, transforms = _import_training_dependencies()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    _prepare_output_dir(out_dir, overwrite=args.overwrite)
    _seed_everything(args.seed, torch=torch)

    records = load_jsonl(manifest_path)
    by_split = {split: [item for item in records if item.get("split") == split] for split in ("train", "val", "test")}
    if not by_split["train"] or not by_split["val"] or not by_split["test"]:
        sizes = {name: len(items) for name, items in by_split.items()}
        raise ValueError(f"Dataset must contain non-empty train/val/test splits, got {sizes}")
    train_counts = Counter(str(item["label"]) for item in by_split["train"])
    missing_classes = [name for name in CLASS_NAMES if train_counts[name] == 0]
    if missing_classes:
        raise ValueError(f"Training split has no samples for classes: {missing_classes}")

    dataset_root = manifest_path.parent
    train_dataset = OrientationDataset(
        records=by_split["train"],
        dataset_root=dataset_root,
        transform=_build_transform(transforms, input_size=args.input_size, training=True),
        dataset_base=Dataset,
    )
    val_dataset = OrientationDataset(
        records=by_split["val"],
        dataset_root=dataset_root,
        transform=_build_transform(transforms, input_size=args.input_size, training=False),
        dataset_base=Dataset,
    )
    test_dataset = OrientationDataset(
        records=by_split["test"],
        dataset_root=dataset_root,
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

    model = build_efficientnet_b0(pretrained=not args.no_pretrained)
    set_backbone_trainable(model, False)
    model.to(device)
    class_weights = _class_weights(train_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    dataset_manifest = _load_dataset_manifest(dataset_root)
    crop_padding = float(dataset_manifest.get("crop_padding", 0.12))
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        if epoch == args.head_warmup_epochs:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": args.backbone_lr},
                    {"params": model.classifier.parameters(), "lr": args.finetune_head_lr},
                ],
                weight_decay=args.weight_decay,
            )

        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            torch=torch,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            torch=torch,
            optimizer=None,
        )
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            input_size=args.input_size,
            crop_padding=crop_padding,
            class_weights=class_weights,
            metrics=epoch_record,
        )
        torch.save(checkpoint, out_dir / "model_last.pth")
        if float(val_metrics["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            torch.save(checkpoint, out_dir / "model_best.pth")
        else:
            epochs_without_improvement += 1
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break

    best = torch.load(out_dir / "model_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.to(device)
    test_metrics = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        torch=torch,
        optimizer=None,
    )
    final_metrics = {
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best_f1),
        "test": test_metrics,
        "class_names": list(CLASS_NAMES),
        "train_class_counts": dict(train_counts),
        "class_weights": class_weights,
    }
    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": "efficientnet_b0",
                "class_names": list(CLASS_NAMES),
                "input_size": int(args.input_size),
                "crop_padding": crop_padding,
                "imagenet_mean": list(IMAGENET_MEAN),
                "imagenet_std": list(IMAGENET_STD),
                "uncertain_prediction_policy": "keep_top1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(manifest_path, out_dir / "split_manifest.jsonl")
    print(json.dumps({"model_best": str(out_dir / "model_best.pth"), **final_metrics}, ensure_ascii=False, indent=2))


class OrientationDataset:
    """Small wrapper intentionally independent of torch at import time."""

    def __new__(
        cls,
        *,
        records: list[dict[str, Any]],
        dataset_root: Path,
        transform: Any,
        dataset_base: Any,
    ) -> Any:
        class _Dataset(dataset_base):
            def __init__(self) -> None:
                self.records = records
                self.dataset_root = dataset_root
                self.transform = transform
                self.class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, index: int) -> tuple[Any, int]:
                from PIL import Image  # noqa: WPS433

                record = self.records[index]
                image_path = record.get("classifier_image", record["masked_rgb"])
                image = Image.open(self.dataset_root / image_path).convert("RGB")
                return self.transform(image), self.class_to_id[str(record["label"])]

        return _Dataset()


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
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(operations)


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    torch: Any,
    optimizer: Any | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            target_values = targets.detach().cpu().numpy()
            for target, prediction in zip(target_values, predictions):
                confusion[int(target), int(prediction)] += 1
    metrics = _classification_metrics(confusion)
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["samples"] = total_items
    return metrics


def _classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
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


def _class_weights(counts: Counter[str]) -> list[float]:
    total = float(sum(counts.values()))
    weights = [total / (len(CLASS_NAMES) * float(counts[name])) for name in CLASS_NAMES]
    mean = float(np.mean(weights))
    return [float(value / mean) for value in weights]


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    input_size: int,
    crop_padding: float,
    class_weights: list[float],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "model_name": "efficientnet_b0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "class_names": CLASS_NAMES,
        "input_size": int(input_size),
        "crop_padding": float(crop_padding),
        "class_weights": class_weights,
        "metrics": metrics,
    }


def _load_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "dataset_manifest.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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


def _import_training_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
        from torch.utils.data import DataLoader, Dataset  # noqa: WPS433
        from torchvision import transforms  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("Training requires torch, torchvision, and Pillow; install the vehicle_orientation project") from exc
    return torch, nn, DataLoader, Dataset, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vehicle front/rear/side classifier.")
    parser.add_argument("--manifest", required=True, help="Path to dataset manifest.jsonl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--head-warmup-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
