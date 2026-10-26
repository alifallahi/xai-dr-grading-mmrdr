"""Fine-tuning loop with validation-based checkpoint selection and early stopping.

Only the training split is used for gradient updates and only the validation split for
model selection. The official test split is never loaded here.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import save_config
from .data.dataset import FundusDataset, build_transforms, load_metadata, make_loader
from .metrics import core_metrics
from .models import build_model, save_checkpoint
from .utils import get_device, run_dir, save_json, seed_everything


def class_weights(labels: np.ndarray, scheme: str, n_classes: int = 5) -> torch.Tensor | None:
    """Per-class loss weights, normalised so the average weight per training image is 1."""
    if scheme in (None, "none"):
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    if scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown class_weighting {scheme!r}")
    w = w * len(labels) / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and normalisation parameters."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device, amp: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def train(cfg: dict) -> dict:
    tcfg = cfg["train"]
    seed_everything(cfg["seed"], tcfg.get("deterministic", False))
    device = get_device()
    out_dir = run_dir(cfg)
    save_config(cfg, out_dir / "config.yaml")

    meta = load_metadata(cfg["paths"]["prepared_dir"])
    train_df = meta[meta["split"] == "train"].reset_index(drop=True)
    val_df = meta[meta["split"] == "val"].reset_index(drop=True)
    print(f"[train] {len(train_df)} training / {len(val_df)} validation images on {device}")

    nw = cfg["data"].get("num_workers", 2)
    prepared = cfg["paths"]["prepared_dir"]
    train_loader = make_loader(FundusDataset(train_df, prepared, build_transforms(cfg, True)),
                               tcfg["batch_size"], True, nw, cfg["seed"])
    val_loader = make_loader(FundusDataset(val_df, prepared, build_transforms(cfg, False)),
                             cfg["evaluation"].get("batch_size", 32), False, nw)

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    weights = class_weights(train_df["grade"].to_numpy(), tcfg.get("class_weighting", "none"))
    print(f"[train] class weights: {None if weights is None else np.round(weights.numpy(), 3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device),
                                    label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=tcfg["lr"])
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, int(tcfg.get("warmup_epochs", 1) * steps_per_epoch),
                              tcfg["epochs"] * steps_per_epoch)
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    selection = tcfg.get("selection_metric", "qwk")
    best_score, best_epoch, patience_left = -np.inf, -1, tcfg.get("early_stopping_patience", 1000)
    history = []
    y_val = val_df["grade"].to_numpy()

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        logits = predict_logits(model, val_loader, device, use_amp)
        val_loss = float(nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y_val)).item())
        m = core_metrics(y_val, logits.argmax(1))
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss,
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_qwk": m["qwk"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"[train] epoch {epoch:3d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | QWK {m['qwk']:.4f} | {row['seconds']:.0f}s")

        score = m[selection]
        if score > best_score:
            best_score, best_epoch = score, epoch
            patience_left = tcfg.get("early_stopping_patience", 1000)
            save_checkpoint(out_dir / "best.pt", model, cfg, epoch, {**m, "val_loss": val_loss})
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    summary = {"run": cfg["experiment_name"], "seed": cfg["seed"], "replicate_of": cfg.get("replicate_of"),
               "lr": tcfg["lr"], "class_weighting": tcfg.get("class_weighting", "none"),
               "best_epoch": best_epoch, "selection_metric": selection,
               "best_val_score": best_score, "epochs_run": len(history),
               "minutes": round(sum(h["seconds"] for h in history) / 60, 1),
               "best_val_metrics": next(h for h in history if h["epoch"] == best_epoch)}
    save_json(summary, out_dir / "train_summary.json")
    print(f"[train] best validation {selection} = {best_score:.4f} at epoch {best_epoch}; saved {out_dir / 'best.pt'}")
    return summary
