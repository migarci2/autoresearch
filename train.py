"""
MNIST autoresearch driver.

The agent-facing surface lives in the editable config sections below:
  - RUN_MODE
  - MODEL_CFG
  - AUGMENT_CFG
  - OPTIM_CFG
  - ENSEMBLE_CFG
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from coordinator import Coordinator
from prepare import NUM_CLASSES, cycle, evaluate_model, make_dataloaders, save_logits_artifact

# ---------------------------------------------------------------------------
# Editable config surface
# ---------------------------------------------------------------------------

SEARCH_TIME_BUDGET = 45
FINAL_TIME_BUDGET = 300

RUN_MODE = {
    "kind": "single_model",  # "single_model" | "ensemble"
    "time_budget_seconds": SEARCH_TIME_BUDGET,
    "final_eval": False,
    "seed": 1337,
    "artifact_root": "runs",
}

MODEL_CFG = {
    "family": "cnn",  # "cnn" | "vit" | "hybrid"
    "channels": [64, 128, 256],
    "blocks_per_stage": 2,
    "kernel_size": 3,
    "dropout": 0.10,
    "classifier_hidden": 256,
    "use_residual": True,
    "norm": "batchnorm",  # "batchnorm" | "groupnorm" | "layernorm"
    "activation": "gelu",  # "relu" | "gelu" | "silu"
    "patch_size": 7,
    "embed_dim": 192,
    "depth": 6,
    "heads": 6,
    "mlp_ratio": 4.0,
    "attention_dropout": 0.0,
    "stochastic_depth": 0.0,
    "hybrid_channels": [32, 64],
}

AUGMENT_CFG = {
    "random_crop_padding": 2,
    "random_rotation_degrees": 0.0,
    "random_affine_degrees": 10.0,
    "random_affine_translate": 0.10,
    "random_affine_scale": [0.95, 1.05],
    "random_affine_shear": 0.0,
    "elastic_alpha": 0.0,
    "elastic_sigma": 0.0,
    "randaugment_ops": 0,
    "randaugment_magnitude": 5,
    "random_erasing_prob": 0.0,
    "random_erasing_scale": [0.02, 0.12],
    "random_erasing_ratio": [0.3, 3.3],
    "mixup_alpha": 0.0,
    "cutmix_alpha": 0.0,
}

OPTIM_CFG = {
    "batch_size": 512,
    "eval_batch_size": 1024,
    "num_workers": 4,
    "grad_accum_steps": 1,
    "optimizer": "adamw",  # "adamw" | "sgd" | "rmsprop"
    "lr": 2.0e-3,
    "weight_decay": 1.0e-4,
    "momentum": 0.9,
    "betas": [0.9, 0.999],
    "scheduler": "cosine",  # "cosine" | "linear" | "none"
    "warmup_ratio": 0.10,
    "min_lr_ratio": 0.05,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "label_smoothing": 0.0,
}

ENSEMBLE_CFG = {
    "candidate_limit": 8,
    "families": ["cnn", "vit", "hybrid"],
    "roles": ["A", "B", "C", "D"],
    "max_members": 4,
    "weight_grid": [0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
    "temperature_grid": [0.85, 0.95, 1.00, 1.05, 1.15],
}

# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = 28


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def runtime_settings() -> dict[str, Any]:
    kind = os.environ.get("AUTORESEARCH_RUN_MODE_KIND", RUN_MODE["kind"])
    final_eval = env_flag("AUTORESEARCH_FINAL_EVAL", bool(RUN_MODE["final_eval"]))
    default_budget = FINAL_TIME_BUDGET if final_eval else int(RUN_MODE["time_budget_seconds"])
    time_budget = float(os.environ.get("AUTORESEARCH_TIME_BUDGET", default_budget))
    seed = int(os.environ.get("AUTORESEARCH_SEED", RUN_MODE["seed"]))
    artifact_root = os.environ.get("AUTORESEARCH_ARTIFACT_ROOT", RUN_MODE["artifact_root"])
    return {
        "kind": kind,
        "final_eval": final_eval,
        "time_budget_seconds": time_budget,
        "seed": seed,
        "artifact_root": artifact_root,
    }


def activation_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.GELU()


def norm_2d(name: str, channels: int) -> nn.Module:
    name = name.lower()
    if name == "groupnorm":
        groups = 8 if channels % 8 == 0 else 4
        return nn.GroupNorm(groups, channels)
    if name == "layernorm":
        return nn.GroupNorm(1, channels)
    return nn.BatchNorm2d(channels)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, norm: str, activation: str):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            norm_2d(norm, out_channels),
            activation_fn(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, kernel_size: int, norm: str, activation: str, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm1 = norm_2d(norm, channels)
        self.act = activation_fn(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm2 = norm_2d(norm, channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ConvNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        channels = list(cfg["channels"])
        blocks_per_stage = int(cfg["blocks_per_stage"])
        kernel_size = int(cfg["kernel_size"])
        dropout = float(cfg["dropout"])
        norm = str(cfg["norm"])
        activation = str(cfg["activation"])
        use_residual = bool(cfg["use_residual"])
        classifier_hidden = int(cfg["classifier_hidden"])

        stages = []
        in_channels = 1
        for stage_idx, out_channels in enumerate(channels):
            layers = [ConvBlock(in_channels, out_channels, kernel_size, norm, activation)]
            if use_residual:
                layers.extend(
                    ResidualUnit(out_channels, kernel_size, norm, activation, dropout)
                    for _ in range(max(0, blocks_per_stage - 1))
                )
            else:
                layers.extend(
                    ConvBlock(out_channels, out_channels, kernel_size, norm, activation)
                    for _ in range(max(0, blocks_per_stage - 1))
                )
            if stage_idx < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels

        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], classifier_hidden),
            activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.grid_h = math.ceil(IMAGE_SIZE / self.patch_size)
        self.grid_w = math.ceil(IMAGE_SIZE / self.patch_size)

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = self.grid_h * self.patch_size - x.size(-2)
        pad_w = self.grid_w * self.patch_size - x.size(-1)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        embed_dim = int(cfg["embed_dim"])
        patch_size = int(cfg["patch_size"])
        depth = int(cfg["depth"])
        heads = int(cfg["heads"])
        mlp_ratio = float(cfg["mlp_ratio"])
        dropout = float(cfg["dropout"])

        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens[:, 0])
        return self.head(tokens)


class HybridTransformer(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        channels = list(cfg["hybrid_channels"])
        embed_dim = int(cfg["embed_dim"])
        depth = int(cfg["depth"])
        heads = int(cfg["heads"])
        mlp_ratio = float(cfg["mlp_ratio"])
        dropout = float(cfg["dropout"])
        norm = str(cfg["norm"])
        activation = str(cfg["activation"])

        stem_layers = []
        in_channels = 1
        for idx, out_channels in enumerate(channels):
            stem_layers.append(ConvBlock(in_channels, out_channels, 3, norm, activation))
            if idx < len(channels) - 1:
                stem_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.stem = nn.Sequential(*stem_layers)
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
            feature_map = self.proj(self.stem(dummy))
            token_count = int(feature_map.shape[-2] * feature_map.shape[-1])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.stem(x))
        tokens = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens[:, 0])
        return self.head(tokens)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    family = str(cfg["family"]).lower()
    if family == "vit":
        return VisionTransformer(cfg)
    if family == "hybrid":
        return HybridTransformer(cfg)
    return ConvNet(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            current = model.state_dict()
            for name, tensor in current.items():
                self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def one_hot_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    targets_oh = F.one_hot(targets, NUM_CLASSES).float()
    if label_smoothing <= 0:
        return targets_oh
    off_value = label_smoothing / NUM_CLASSES
    on_value = 1.0 - label_smoothing + off_value
    return targets_oh * (on_value - off_value) + off_value


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    _, _, height, width = size
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))
    cx = torch.randint(0, width, (1,)).item()
    cy = torch.randint(0, height, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_batch_mix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    augment_cfg: dict[str, Any],
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mixup_alpha = float(augment_cfg.get("mixup_alpha", 0.0))
    cutmix_alpha = float(augment_cfg.get("cutmix_alpha", 0.0))
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return inputs, one_hot_targets(targets, label_smoothing)

    perm = torch.randperm(inputs.size(0), device=inputs.device)
    targets_a = one_hot_targets(targets, label_smoothing)
    targets_b = one_hot_targets(targets[perm], label_smoothing)

    if cutmix_alpha > 0:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))
        return mixed, lam * targets_a + (1.0 - lam) * targets_b

    lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
    mixed = lam * inputs + (1.0 - lam) * inputs[perm]
    return mixed, lam * targets_a + (1.0 - lam) * targets_b


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(cfg["optimizer"]).lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["momentum"]),
            weight_decay=weight_decay,
            nesterov=True,
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=float(cfg["momentum"]),
            weight_decay=weight_decay,
        )
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def lr_multiplier(progress: float, cfg: dict[str, Any]) -> float:
    progress = min(max(progress, 0.0), 1.0)
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.0))
    scheduler = str(cfg.get("scheduler", "cosine")).lower()

    if warmup_ratio > 0 and progress < warmup_ratio:
        return max(1e-6, progress / warmup_ratio)

    post_warmup = 0.0 if warmup_ratio >= 1.0 else (progress - warmup_ratio) / max(1e-8, 1.0 - warmup_ratio)
    if scheduler == "none":
        return 1.0
    if scheduler == "linear":
        return min_lr_ratio + (1.0 - post_warmup) * (1.0 - min_lr_ratio)
    cosine = 0.5 * (1.0 + math.cos(math.pi * post_warmup))
    return min_lr_ratio + cosine * (1.0 - min_lr_ratio)


@contextlib.contextmanager
def autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def json_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def make_artifact_dir(runtime: dict[str, Any], model_cfg: dict[str, Any]) -> Path:
    workspace_root = Path(os.environ.get("AUTORESEARCH_WORKSPACE_ROOT", str(Path.cwd())))
    artifact_root = Path(runtime["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = workspace_root / artifact_root
    run_id = os.environ.get("AUTORESEARCH_EXPERIMENT_KEY")
    if not run_id:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        family = str(model_cfg["family"]).lower()
        run_id = f"{runtime['kind']}-{family}-{timestamp}"
    artifact_dir = artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def evaluate_logits(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> dict[str, Any]:
    scaled = logits / temperature
    loss = F.cross_entropy(scaled, targets).item()
    predictions = scaled.argmax(dim=1)
    errors = int((predictions != targets).sum().item())
    accuracy = float((predictions == targets).float().mean().item())
    return {"loss": loss, "errors": errors, "accuracy": accuracy, "temperature": temperature}


def load_single_checkpoint(checkpoint_path: str | os.PathLike[str], device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(payload["model_cfg"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def evaluate_single_model(
    model: nn.Module,
    loaders: dict[str, Any],
    device: torch.device,
    *,
    final_eval: bool,
) -> dict[str, Any]:
    val_metrics = evaluate_model(model, loaders["val"], device, collect_logits=True)
    metrics = {
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_errors": int(val_metrics["errors"]),
        "val_logits": val_metrics["logits"],
        "val_targets": val_metrics["targets"],
    }
    if final_eval:
        test_metrics = evaluate_model(model, loaders["test"], device, collect_logits=False)
        metrics.update(
            {
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_errors": int(test_metrics["errors"]),
            }
        )
    return metrics


def run_single_model(
    runtime: dict[str, Any],
    device: torch.device,
    coord: Coordinator,
) -> dict[str, Any]:
    torch.manual_seed(runtime["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(runtime["seed"])
        torch.backends.cudnn.benchmark = True

    loaders = make_dataloaders(
        augment_cfg=AUGMENT_CFG,
        batch_size=int(OPTIM_CFG["batch_size"]),
        eval_batch_size=int(OPTIM_CFG["eval_batch_size"]),
        num_workers=int(OPTIM_CFG["num_workers"]),
    )
    train_iterator = cycle(loaders["train"])

    model = build_model(MODEL_CFG).to(device)
    optimizer = build_optimizer(model, OPTIM_CFG)
    ema_decay = float(OPTIM_CFG.get("ema_decay", 0.0))
    ema = ModelEMA(model, ema_decay) if ema_decay > 0 else None
    grad_accum_steps = max(1, int(OPTIM_CFG.get("grad_accum_steps", 1)))
    grad_clip = float(OPTIM_CFG.get("grad_clip", 0.0))
    label_smoothing = float(OPTIM_CFG.get("label_smoothing", 0.0))
    num_params = count_parameters(model)

    artifact_dir = make_artifact_dir(runtime, MODEL_CFG)
    config_blob = {
        "runtime": runtime,
        "model_cfg": MODEL_CFG,
        "augment_cfg": AUGMENT_CFG,
        "optim_cfg": OPTIM_CFG,
        "ensemble_cfg": ENSEMBLE_CFG,
    }
    config_sha = json_sha(config_blob)
    config_path = artifact_dir / "config.json"
    save_json(config_path, config_blob)

    start_time = time.time()
    train_loss_ema = 0.0
    steps = 0
    samples_seen = 0
    progress_log_interval = 10

    while time.time() - start_time < runtime["time_budget_seconds"]:
        optimizer.zero_grad(set_to_none=True)
        batch_loss_value = 0.0
        for _ in range(grad_accum_steps):
            inputs, targets = next(train_iterator)
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            inputs, soft_targets = apply_batch_mix(inputs, targets, AUGMENT_CFG, label_smoothing)

            with autocast_context(device):
                logits = model(inputs)
                loss = soft_cross_entropy(logits, soft_targets) / grad_accum_steps
            loss.backward()
            batch_loss_value += loss.item() * grad_accum_steps
            samples_seen += int(inputs.size(0))

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        progress = (time.time() - start_time) / runtime["time_budget_seconds"]
        lr_mult = lr_multiplier(progress, OPTIM_CFG)
        for group in optimizer.param_groups:
            group["lr"] = float(OPTIM_CFG["lr"]) * lr_mult

        optimizer.step()
        if ema is not None:
            ema.update(model)

        train_loss_ema = 0.9 * train_loss_ema + 0.1 * batch_loss_value if steps > 0 else batch_loss_value
        steps += 1

        if steps % progress_log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"step={steps:04d} elapsed={elapsed:6.1f}s "
                f"train_loss={train_loss_ema:.5f} lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    eval_model_instance = model
    if ema is not None:
        ema.apply_to(model)
        eval_model_instance = model

    evaluation = evaluate_single_model(eval_model_instance, loaders, device, final_eval=runtime["final_eval"])
    state_dict_to_save = {
        name: tensor.detach().cpu()
        for name, tensor in eval_model_instance.state_dict().items()
    }

    if ema is not None:
        ema.restore(model)

    checkpoint_path = artifact_dir / "checkpoint.pt"
    torch.save(
        {
            "model_cfg": MODEL_CFG,
            "state_dict": state_dict_to_save,
            "config_sha": config_sha,
            "model_family": MODEL_CFG["family"],
        },
        checkpoint_path,
    )

    val_logits_path = artifact_dir / "val_logits.pt"
    save_logits_artifact(
        val_logits_path,
        evaluation["val_logits"],
        evaluation["val_targets"],
        metadata={"config_sha": config_sha, "model_family": MODEL_CFG["family"]},
    )

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    training_seconds = time.time() - start_time
    metadata = {
        "config_sha": config_sha,
        "model_family": MODEL_CFG["family"],
        "run_mode": runtime["kind"],
        "num_params": num_params,
        "train_loss": float(train_loss_ema),
        "training_seconds": training_seconds,
        "steps": steps,
        "samples_seen": samples_seen,
        "peak_vram_mb": peak_vram_mb,
        "val_errors": evaluation["val_errors"],
        "val_accuracy": evaluation["val_accuracy"],
        "val_loss": evaluation["val_loss"],
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "val_logits_path": str(val_logits_path),
        "final_eval": runtime["final_eval"],
    }
    if runtime["final_eval"]:
        metadata.update(
            {
                "test_errors": evaluation["test_errors"],
                "test_accuracy": evaluation["test_accuracy"],
                "test_loss": evaluation["test_loss"],
            }
        )

    metadata_path = artifact_dir / "metadata.json"
    save_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def ensemble_candidates(coord: Coordinator) -> list[dict[str, Any]]:
    return coord.get_ranked_results(
        limit=int(ENSEMBLE_CFG["candidate_limit"]),
        status="keep",
        run_mode="single_model",
        families=list(ENSEMBLE_CFG.get("families", [])),
        roles=list(ENSEMBLE_CFG.get("roles", [])),
        final_eval=False,
    )


def greedy_ensemble_search(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    prepared = []
    for row in candidates:
        metrics = row["metrics"]
        payload = torch.load(metrics["val_logits_path"], map_location="cpu")
        prepared.append(
            {
                "record": row,
                "logits": payload["logits"].float(),
                "targets": payload["targets"].long(),
                "weight": 1.0,
            }
        )

    if not prepared:
        raise RuntimeError("No eligible checkpoints found in shared/best_checkpoints for ensemble mode")

    reference_targets = prepared[0]["targets"]
    for candidate in prepared[1:]:
        if not torch.equal(reference_targets, candidate["targets"]):
            raise RuntimeError("Validation target mismatch across candidate checkpoints")

    temp_grid = [float(value) for value in ENSEMBLE_CFG["temperature_grid"]]
    weight_grid = [float(value) for value in ENSEMBLE_CFG["weight_grid"]]
    max_members = int(ENSEMBLE_CFG["max_members"])

    def score(combo: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
        total_weight = sum(member["weight"] for member in combo)
        logits = sum(member["weight"] * member["logits"] for member in combo) / total_weight
        metrics = evaluate_logits(logits, reference_targets, temperature=temperature)
        metrics["logits"] = logits
        return metrics

    best_single = min(prepared, key=lambda item: item["record"]["metrics"]["val_errors"])
    selected = [best_single]
    best_metrics = min((score(selected, temp) for temp in temp_grid), key=lambda item: (item["errors"], item["loss"]))
    best_temperature = best_metrics["temperature"]

    remaining = [candidate for candidate in prepared if candidate is not best_single]
    while remaining and len(selected) < max_members:
        improvement = None
        for candidate in remaining:
            for weight in weight_grid:
                trial_member = dict(candidate)
                trial_member["weight"] = weight
                trial_combo = selected + [trial_member]
                trial_metrics = min(
                    (score(trial_combo, temp) for temp in temp_grid),
                    key=lambda item: (item["errors"], item["loss"]),
                )
                trial_signature = (trial_metrics["errors"], trial_metrics["loss"])
                best_signature = (best_metrics["errors"], best_metrics["loss"])
                if trial_signature < best_signature:
                    improvement = (candidate, weight, trial_metrics)
                    best_signature = trial_signature

        if improvement is None:
            break

        candidate, weight, trial_metrics = improvement
        accepted = dict(candidate)
        accepted["weight"] = weight
        selected.append(accepted)
        remaining = [item for item in remaining if item is not candidate]
        best_metrics = trial_metrics
        best_temperature = trial_metrics["temperature"]

    return {
        "selected": selected,
        "temperature": best_temperature,
        "val_metrics": best_metrics,
        "targets": reference_targets,
    }


@torch.no_grad()
def evaluate_ensemble_on_loader(selected: list[dict[str, Any]], loader, device: torch.device, temperature: float) -> dict[str, Any]:
    models = []
    for member in selected:
        checkpoint_path = member["record"]["metrics"]["checkpoint_path"]
        models.append((load_single_checkpoint(checkpoint_path, device), float(member["weight"])))

    total_loss = 0.0
    total_examples = 0
    total_errors = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        ensemble_logits = None
        total_weight = 0.0
        for model, weight in models:
            logits = model(inputs)
            ensemble_logits = logits * weight if ensemble_logits is None else ensemble_logits + logits * weight
            total_weight += weight
        ensemble_logits = ensemble_logits / total_weight
        scaled_logits = ensemble_logits / temperature
        total_loss += F.cross_entropy(scaled_logits, targets, reduction="sum").item()
        total_errors += int((scaled_logits.argmax(dim=1) != targets).sum().item())
        total_examples += int(targets.numel())

    return {
        "loss": total_loss / max(1, total_examples),
        "errors": total_errors,
        "accuracy": 1.0 - (total_errors / max(1, total_examples)),
    }


def run_ensemble(
    runtime: dict[str, Any],
    device: torch.device,
    coord: Coordinator,
) -> dict[str, Any]:
    loaders = make_dataloaders(
        augment_cfg={},
        batch_size=int(OPTIM_CFG["batch_size"]),
        eval_batch_size=int(OPTIM_CFG["eval_batch_size"]),
        num_workers=int(OPTIM_CFG["num_workers"]),
    )
    candidates = ensemble_candidates(coord)
    search = greedy_ensemble_search(candidates)
    artifact_dir = make_artifact_dir(runtime, {"family": "ensemble"})

    ensemble_logits_path = artifact_dir / "val_logits.pt"
    save_logits_artifact(
        ensemble_logits_path,
        search["val_metrics"]["logits"],
        search["targets"],
        metadata={"temperature": search["temperature"], "members": len(search["selected"])},
    )

    members = []
    for member in search["selected"]:
        members.append(
            {
                "experiment_key": member["record"]["experiment_key"],
                "role": member["record"]["role"],
                "weight": member["weight"],
                "checkpoint_path": member["record"]["metrics"]["checkpoint_path"],
                "config_path": member["record"]["metrics"].get("config_path"),
                "model_family": member["record"]["metrics"]["model_family"],
                "val_errors": member["record"]["metrics"]["val_errors"],
            }
        )

    manifest = {
        "run_mode": runtime["kind"],
        "model_family": "ensemble",
        "temperature": search["temperature"],
        "members": members,
        "val_errors": int(search["val_metrics"]["errors"]),
        "val_accuracy": float(search["val_metrics"]["accuracy"]),
        "val_loss": float(search["val_metrics"]["loss"]),
    }

    if runtime["final_eval"]:
        test_metrics = evaluate_ensemble_on_loader(search["selected"], loaders["test"], device, search["temperature"])
        manifest.update(
            {
                "test_errors": int(test_metrics["errors"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
            }
        )

    manifest_path = artifact_dir / "ensemble_manifest.json"
    save_json(manifest_path, manifest)
    metadata_path = artifact_dir / "metadata.json"
    save_json(metadata_path, manifest)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    manifest.update(
        {
            "train_loss": 0.0,
            "training_seconds": 0.0,
            "peak_vram_mb": peak_vram_mb,
            "checkpoint_path": None,
            "config_path": None,
            "val_logits_path": str(ensemble_logits_path),
            "manifest_path": str(manifest_path),
            "metadata_path": str(metadata_path),
            "config_sha": json_sha(manifest),
        }
    )
    return manifest


def print_summary(metrics: dict[str, Any]) -> None:
    print("---")
    keys = [
        ("val_errors", metrics.get("val_errors")),
        ("val_accuracy", metrics.get("val_accuracy")),
        ("val_loss", metrics.get("val_loss")),
        ("train_loss", metrics.get("train_loss")),
        ("training_seconds", metrics.get("training_seconds")),
        ("peak_vram_mb", metrics.get("peak_vram_mb")),
        ("checkpoint_path", metrics.get("checkpoint_path")),
        ("run_mode", metrics.get("run_mode")),
        ("model_family", metrics.get("model_family")),
        ("config_sha", metrics.get("config_sha")),
    ]
    optional = [("test_errors", metrics.get("test_errors")), ("test_accuracy", metrics.get("test_accuracy"))]
    for key, value in keys + optional:
        if value is None:
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    runtime = runtime_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coord = Coordinator()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with coord.gpu_lease():
        if runtime["kind"] == "ensemble":
            metrics = run_ensemble(runtime, device, coord)
        else:
            metrics = run_single_model(runtime, device, coord)

    print_summary(metrics)


if __name__ == "__main__":
    main()
