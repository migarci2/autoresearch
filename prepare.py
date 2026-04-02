"""
MNIST data preparation and runtime utilities for local autoresearch.

Usage:
    uv run prepare.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch-mnist"
DATA_DIR = CACHE_DIR / "data"
SPLIT_FILE = CACHE_DIR / "split_indices.pt"
SPLIT_MANIFEST = CACHE_DIR / "split_manifest.json"

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

SEARCH_TRAIN_SIZE = 50_000
SEARCH_VAL_SIZE = 10_000
SPLIT_SEED = 1337

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = max(0, min(4, (os.cpu_count() or 1) - 1))


@dataclass(frozen=True)
class SplitSpec:
    train_size: int = SEARCH_TRAIN_SIZE
    val_size: int = SEARCH_VAL_SIZE
    seed: int = SPLIT_SEED


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_mnist(download: bool = True) -> None:
    """Ensure MNIST data and the cached split manifest exist."""
    ensure_directories()
    datasets.MNIST(root=DATA_DIR, train=True, download=download)
    datasets.MNIST(root=DATA_DIR, train=False, download=download)
    get_split_indices()


def _build_split_indices(spec: SplitSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(spec.seed)
    permutation = torch.randperm(60_000, generator=generator)
    train_indices = permutation[: spec.train_size].clone()
    val_indices = permutation[spec.train_size : spec.train_size + spec.val_size].clone()
    return {
        "train": train_indices,
        "val": val_indices,
        "seed": torch.tensor(spec.seed, dtype=torch.int64),
    }


def get_split_indices(spec: SplitSpec = SplitSpec()) -> dict[str, torch.Tensor]:
    """
    Return the fixed 50k/10k train/validation split.

    The split is cached on disk so all agents reuse the exact same partition.
    """
    ensure_directories()
    if SPLIT_FILE.exists():
        cached = torch.load(SPLIT_FILE, map_location="cpu")
        if (
            cached["train"].numel() == spec.train_size
            and cached["val"].numel() == spec.val_size
            and int(cached["seed"]) == spec.seed
        ):
            return cached

    split = _build_split_indices(spec)
    torch.save(split, SPLIT_FILE)
    manifest = {
        "search_train_size": spec.train_size,
        "search_val_size": spec.val_size,
        "seed": spec.seed,
        "train_min_index": int(split["train"].min()),
        "train_max_index": int(split["train"].max()),
        "val_min_index": int(split["val"].min()),
        "val_max_index": int(split["val"].max()),
    }
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return split


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _tuple_range(value: float | list[float] | tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return float(value[0]), float(value[1])
        raise ValueError(f"Expected a pair, got: {value}")
    value_f = float(value)
    return max(0.0, 1.0 - value_f), 1.0 + value_f


def build_transforms(augment_cfg: dict | None, split: str) -> transforms.Compose:
    augment_cfg = dict(augment_cfg or {})
    normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    if split != "train":
        return transforms.Compose([transforms.ToTensor(), normalize])

    pre_tensor: list[object] = []
    post_tensor: list[object] = []

    crop_padding = int(augment_cfg.get("random_crop_padding", 0))
    if crop_padding > 0:
        pre_tensor.append(transforms.RandomCrop(IMAGE_SIZE, padding=crop_padding, fill=0))

    rotation = float(augment_cfg.get("random_rotation_degrees", 0.0))
    if rotation > 0:
        pre_tensor.append(
            transforms.RandomRotation(
                degrees=rotation,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    affine_degrees = float(augment_cfg.get("random_affine_degrees", 0.0))
    translate = float(augment_cfg.get("random_affine_translate", 0.0))
    scale = _tuple_range(augment_cfg.get("random_affine_scale"), (1.0, 1.0))
    shear = float(augment_cfg.get("random_affine_shear", 0.0))
    if affine_degrees > 0 or translate > 0 or scale != (1.0, 1.0) or shear > 0:
        pre_tensor.append(
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=(translate, translate) if translate > 0 else None,
                scale=scale,
                shear=(-shear, shear) if shear > 0 else None,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    randaugment_ops = int(augment_cfg.get("randaugment_ops", 0))
    if randaugment_ops > 0:
        pre_tensor.append(
            transforms.RandAugment(
                num_ops=randaugment_ops,
                magnitude=int(augment_cfg.get("randaugment_magnitude", 5)),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    post_tensor.append(transforms.ToTensor())

    elastic_alpha = float(augment_cfg.get("elastic_alpha", 0.0))
    elastic_sigma = float(augment_cfg.get("elastic_sigma", 0.0))
    if elastic_alpha > 0 and elastic_sigma > 0:
        post_tensor.append(
            transforms.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )

    post_tensor.append(normalize)

    erasing_prob = float(augment_cfg.get("random_erasing_prob", 0.0))
    if erasing_prob > 0:
        scale_range = tuple(augment_cfg.get("random_erasing_scale", (0.02, 0.15)))
        ratio_range = tuple(augment_cfg.get("random_erasing_ratio", (0.3, 3.3)))
        post_tensor.append(
            transforms.RandomErasing(
                p=erasing_prob,
                scale=scale_range,
                ratio=ratio_range,
                value=0.0,
            )
        )

    return transforms.Compose([*pre_tensor, *post_tensor])


# ---------------------------------------------------------------------------
# Datasets and loaders
# ---------------------------------------------------------------------------

def get_datasets(augment_cfg: dict | None = None, spec: SplitSpec = SplitSpec()):
    prepare_mnist(download=True)
    split = get_split_indices(spec)
    train_transform = build_transforms(augment_cfg, "train")
    eval_transform = build_transforms({}, "eval")

    train_base = datasets.MNIST(root=DATA_DIR, train=True, transform=train_transform, download=False)
    val_base = datasets.MNIST(root=DATA_DIR, train=True, transform=eval_transform, download=False)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, transform=eval_transform, download=False)

    train_set = Subset(train_base, split["train"].tolist())
    val_set = Subset(val_base, split["val"].tolist())
    return train_set, val_set, test_set


def make_dataloaders(
    augment_cfg: dict | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    spec: SplitSpec = SplitSpec(),
) -> dict[str, DataLoader]:
    train_set, val_set, test_set = get_datasets(augment_cfg=augment_cfg, spec=spec)
    eval_batch_size = eval_batch_size or batch_size
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        ),
        "val": DataLoader(
            val_set,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            test_set,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        ),
    }
    return loaders


def cycle(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, loss_sum: float) -> dict[str, float]:
    num_examples = int(targets.numel())
    predictions = logits.argmax(dim=1)
    correct = int((predictions == targets).sum().item())
    errors = num_examples - correct
    return {
        "loss": float(loss_sum / max(1, num_examples)),
        "accuracy": float(correct / max(1, num_examples)),
        "errors": errors,
        "num_examples": num_examples,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_logits: bool = False,
) -> dict[str, object]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    targets_parts: list[torch.Tensor] = []
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=torch.cuda.is_available())
        targets = targets.to(device, non_blocking=torch.cuda.is_available())
        logits = model(inputs)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(targets.numel())

        if collect_logits:
            logits_parts.append(logits.detach().cpu())
        targets_parts.append(targets.detach().cpu())

    targets_cpu = torch.cat(targets_parts, dim=0)
    errors = total_examples - total_correct
    metrics = {
        "loss": float(total_loss / max(1, total_examples)),
        "accuracy": float(total_correct / max(1, total_examples)),
        "errors": int(errors),
        "num_examples": int(total_examples),
    }
    if not collect_logits:
        metrics["logits"] = None
        metrics["targets"] = targets_cpu
        return metrics

    logits_cpu = torch.cat(logits_parts, dim=0)
    metrics["logits"] = logits_cpu
    metrics["targets"] = targets_cpu
    return metrics


def save_logits_artifact(path: os.PathLike[str] | str, logits: torch.Tensor, targets: torch.Tensor, metadata: dict | None = None) -> None:
    payload = {
        "logits": logits,
        "targets": targets,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def split_summary(spec: SplitSpec = SplitSpec()) -> dict[str, int]:
    split = get_split_indices(spec)
    train_ids = set(split["train"].tolist())
    val_ids = set(split["val"].tolist())
    overlap = len(train_ids.intersection(val_ids))
    return {
        "search_train_size": len(train_ids),
        "search_val_size": len(val_ids),
        "test_size": 10_000,
        "overlap": overlap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST for local autoresearch")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for smoke inspection")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Workers for smoke inspection")
    args = parser.parse_args()

    prepare_mnist(download=True)
    loaders = make_dataloaders(batch_size=args.batch_size, eval_batch_size=args.batch_size, num_workers=args.num_workers)
    train_inputs, train_targets = next(iter(loaders["train"]))
    val_inputs, val_targets = next(iter(loaders["val"]))
    summary = split_summary()

    print(f"Cache directory: {CACHE_DIR}")
    print("Split summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("Smoke batches:")
    print(f"  train_batch_shape: {tuple(train_inputs.shape)} labels={tuple(train_targets.shape)}")
    print(f"  val_batch_shape:   {tuple(val_inputs.shape)} labels={tuple(val_targets.shape)}")
    print(f"  image_shape:       {(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)}")
    print(f"  normalization:     mean={MNIST_MEAN:.4f} std={MNIST_STD:.4f}")
    print(f"  split_spec:        {asdict(SplitSpec())}")


if __name__ == "__main__":
    main()
