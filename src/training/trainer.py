"""
Trainer
=======
Training loop with:
  - Mixed precision  (fp16 via torch.amp)
  - Differential LRs (backbone 10x lower than new layers)
  - Linear warmup + cosine LR decay
  - Gradient clipping
  - Early stopping  (monitors weighted-F1)
  - TensorBoard logging
  - Best-model checkpointing
  - Supports both raw and cached batch formats
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer:          AdamW,
    num_warmup_steps:   int,
    num_training_steps: int,
    min_lr_ratio:       float = 0.05,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Model Configuration per Mode
# ─────────────────────────────────────────────────────────────────────────────

def configure_model_for_mode(model: nn.Module, mode: str) -> None:
    """
    Set requires_grad based on which modules are actually used.
    Works for both MultimodalEmotionModel and CachedMultimodalEmotionModel.
    """
    # Start fully trainable
    for p in model.parameters():
        p.requires_grad = True

    # Re-apply backbone partial freezes (only for end-to-end model)
    if hasattr(model, "audio_branch"):
        for p in model.audio_branch.hubert.feature_extractor.parameters():
            p.requires_grad = False
        for p in model.audio_branch.hubert.feature_projection.parameters():
            p.requires_grad = False

    if hasattr(model, "visual_branch"):
        freeze_prefixes = ["conv_stem", "bn1", "blocks.0", "blocks.1"]
        for name, p in model.visual_branch.backbone.named_parameters():
            if any(name.startswith(pf) for pf in freeze_prefixes):
                p.requires_grad = False

    # Mode-specific freezing
    if mode == "audio_only":
        modules_to_freeze = []
        if hasattr(model, "visual_branch"):
            modules_to_freeze.append(model.visual_branch)
        if hasattr(model, "frame_proj"):
            modules_to_freeze.append(model.frame_proj)
        if hasattr(model, "temporal_transformer"):
            modules_to_freeze.append(model.temporal_transformer)
        if hasattr(model, "visual_out_proj"):
            modules_to_freeze.append(model.visual_out_proj)
        modules_to_freeze.append(model.fusion)
        modules_to_freeze.append(model.visual_only_head)
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

    elif mode == "visual_only":
        modules_to_freeze = []
        if hasattr(model, "audio_branch"):
            modules_to_freeze.append(model.audio_branch)
        if hasattr(model, "audio_proj"):
            modules_to_freeze.append(model.audio_proj)
        modules_to_freeze.append(model.fusion)
        modules_to_freeze.append(model.audio_only_head)
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

    else:  # multimodal
        for p in model.audio_only_head.parameters():
            p.requires_grad = False
        for p in model.visual_only_head.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"configure_model_for_mode [{mode}]  →  {trainable:,} trainable params")


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer: differential learning rates
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: Dict, mode: str) -> AdamW:
    """
    For end-to-end model: backbone gets lr×0.1, new layers get lr×1.0
    For cached model: all params get same lr (no heavy backbone)
    """
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]

    # Check if this is a cached model (no heavy backbone)
    is_cached = not hasattr(model, "audio_branch")

    if is_cached:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=wd)
        logger.info(
            f"Optimizer (cached)  |  "
            f"all params: {sum(p.numel() for p in trainable):,} (lr={lr:.1e})"
        )
        return optimizer

    # End-to-end model: differential LRs
    backbone_ids: set = set()

    if mode in ("multimodal", "audio_only"):
        for p in model.audio_branch.hubert.parameters():
            if p.requires_grad:
                backbone_ids.add(id(p))

    if mode in ("multimodal", "visual_only"):
        for p in model.visual_branch.backbone.parameters():
            if p.requires_grad:
                backbone_ids.add(id(p))

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in backbone_ids
    ]
    new_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * 0.1, "name": "backbone"})
    if new_params:
        param_groups.append({"params": new_params, "lr": lr, "name": "new_layers"})

    optimizer = AdamW(param_groups, weight_decay=wd)
    logger.info(
        f"Optimizer  |  "
        f"backbone: {sum(p.numel() for p in backbone_params):,} (lr={lr*0.1:.1e})  |  "
        f"new_layers: {sum(p.numel() for p in new_params):,} (lr={lr:.1e})"
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Batch unpacking helper
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_batch(batch: Dict, device: torch.device):
    """
    Supports both raw and cached batch formats.
    Returns (audio_or_feat, frames_or_feat, labels).
    """
    labels = batch["label"].to(device, non_blocking=True)
    if "audio_feat" in batch:
        # Cached format
        audio  = batch["audio_feat"].to(device, non_blocking=True)
        frames = batch["frame_feat"].to(device, non_blocking=True)
    else:
        # Raw format
        audio  = batch["audio"].to(device,  non_blocking=True)
        frames = batch["frames"].to(device, non_blocking=True)
    return audio, frames, labels


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = -float("inf")

    def __call__(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Args:
        model     : MultimodalEmotionModel or CachedMultimodalEmotionModel
        config    : loaded config.yaml dict
        device    : torch.device
        run_name  : unique experiment identifier
        criterion : loss function
    """

    def __init__(
        self,
        model:     nn.Module,
        config:    Dict,
        device:    torch.device,
        run_name:  str,
        criterion: nn.Module,
    ):
        self.model     = model
        self.config    = config
        self.device    = device
        self.run_name  = run_name
        self.criterion = criterion.to(device)

        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.log_dir  = Path(config["paths"]["logs"]) / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer     = SummaryWriter(str(self.log_dir))
        self.use_amp    = config["training"]["mixed_precision"]
        self.scaler     = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.max_epochs = config["training"]["num_epochs"]
        self.grad_clip  = config["training"]["grad_clip"]
        self.patience   = config["training"]["early_stopping_patience"]

        logger.info(
            f"Trainer ready  |  run={run_name}  |  "
            f"amp={self.use_amp}  |  device={device}"
        )

    # ─────────────────────────────────────────────────────────────

    def train_one_epoch(
        self,
        loader:    DataLoader,
        optimizer: AdamW,
        scheduler: LambdaLR,
        mode:      str,
        epoch:     int,
    ) -> Dict:
        self.model.train()
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0

        pbar = tqdm(loader, desc=f"Train E{epoch:03d}", leave=False, ncols=90)

        for batch in pbar:
            audio, frames, labels = _unpack_batch(batch, self.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.model(audio, frames, mode=mode)
                loss      = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            with torch.no_grad():
                n_correct  += (logits.argmax(-1) == labels).sum().item()
                n_total    += labels.size(0)
                total_loss += loss.item() * labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc =f"{100*n_correct/max(n_total,1):.1f}%",
            )

        return {
            "loss": total_loss / max(n_total, 1),
            "acc" : n_correct  / max(n_total, 1),
        }

    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader:             DataLoader,
        mode:               str,
        epoch:              int,
        collect_embeddings: bool = False,
    ) -> Dict:
        self.model.eval()
        total_loss  = 0.0
        all_preds   = []
        all_labels  = []
        audio_embs  = []
        visual_embs = []

        for batch in tqdm(loader, desc=f"Val   E{epoch:03d}", leave=False, ncols=90):
            audio, frames, labels = _unpack_batch(batch, self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, extra = self.model(audio, frames, mode=mode)
                loss          = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(labels.cpu())

            if collect_embeddings:
                if extra.get("audio_emb") is not None:
                    audio_embs.append(extra["audio_emb"].cpu().float())
                if extra.get("visual_emb") is not None:
                    visual_embs.append(extra["visual_emb"].cpu().float())

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        metrics        = compute_metrics(all_labels, all_preds)
        metrics["loss"]   = total_loss / max(len(all_labels), 1)
        metrics["preds"]  = all_preds
        metrics["labels"] = all_labels

        if audio_embs:
            metrics["audio_embs"]  = torch.cat(audio_embs)
        if visual_embs:
            metrics["visual_embs"] = torch.cat(visual_embs)

        return metrics

    # ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("preds", "labels", "audio_embs", "visual_embs",
                         "f1_per_class", "confusion_matrix", "per_class_f1_dict")
        }
        state = {
            "epoch"      : epoch,
            "run_name"   : self.run_name,
            "model_state": self.model.state_dict(),
            "metrics"    : scalar_metrics,
        }
        torch.save(state, self.ckpt_dir / f"{self.run_name}_last.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / f"{self.run_name}_best.pt")
            logger.info(f"  ★  Best checkpoint saved  (epoch {epoch})")

    def load_best_checkpoint(self) -> Dict:
        path  = self.ckpt_dir / f"{self.run_name}_best.pt"
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        logger.info(f"Loaded best checkpoint  epoch={state['epoch']}  ←  {path}")
        return state["metrics"]

    # ─────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode:         str = "multimodal",
    ) -> Dict:
        configure_model_for_mode(self.model, mode)
        optimizer = build_optimizer(self.model, self.config, mode)

        steps_per_epoch = len(train_loader)
        total_steps     = steps_per_epoch * self.max_epochs
        warmup_steps    = steps_per_epoch * self.config["training"]["warmup_epochs"]
        scheduler       = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        early_stop = EarlyStopping(patience=self.patience)

        best_f1      = -1.0
        best_metrics = None

        logger.info(f"\n{'='*62}")
        logger.info(f"  TRAINING  [{self.run_name}]  mode={mode}")
        logger.info(f"  Max epochs     : {self.max_epochs}")
        logger.info(f"  Steps / epoch  : {steps_per_epoch}")
        logger.info(f"  Warmup steps   : {warmup_steps}")
        logger.info(f"  Early stopping : patience={self.patience}")
        logger.info(f"{'='*62}\n")

        for epoch in range(1, self.max_epochs + 1):

            tr = self.train_one_epoch(train_loader, optimizer, scheduler, mode, epoch)
            vl = self.evaluate(val_loader, mode, epoch)

            # TensorBoard
            self.writer.add_scalars("Loss", {"train": tr["loss"], "val": vl["loss"]}, epoch)
            self.writer.add_scalars("Acc",  {"train": tr["acc"],  "val": vl["acc"]},  epoch)
            self.writer.add_scalar("Metrics/F1_weighted", vl["f1_weighted"], epoch)
            self.writer.add_scalar("Metrics/UAR",         vl["uar"],         epoch)
            self.writer.add_scalar("Metrics/F1_macro",    vl["f1_macro"],    epoch)
            self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"],    epoch)

            # Checkpoint
            is_best = vl["f1_weighted"] > best_f1
            if is_best:
                best_f1      = vl["f1_weighted"]
                best_metrics = vl
            self.save_checkpoint(epoch, vl, is_best=is_best)

            # Console summary
            star = "  ★" if is_best else ""
            logger.info(
                f"E{epoch:03d}/{self.max_epochs}  "
                f"tr_loss={tr['loss']:.4f} tr_acc={100*tr['acc']:.1f}%  |  "
                f"vl_loss={vl['loss']:.4f} vl_acc={100*vl['acc']:.1f}%  "
                f"F1={vl['f1_weighted']:.4f}  UAR={vl['uar']:.4f}"
                f"{star}"
            )

            # Per-class F1 every 5 epochs
            if epoch % 5 == 0:
                pcd = vl["per_class_f1_dict"]
                logger.info("  Per-class F1: " + "  ".join(
                    f"{k}={v:.3f}" for k, v in pcd.items()
                ))

            # Early stopping
            if early_stop(vl["f1_weighted"]):
                logger.info(f"\n⚠  Early stopping triggered at epoch {epoch}")
                break

        self.writer.close()

        logger.info(f"\n{'='*62}")
        logger.info(f"  Training complete  [{self.run_name}]")
        logger.info(f"  Best val  F1       : {best_f1:.4f}")
        logger.info(f"  Best val  UAR      : {best_metrics['uar']:.4f}")
        logger.info(f"  Best val  Accuracy : {100*best_metrics['acc']:.2f}%")
        logger.info(f"{'='*62}\n")

        return best_metrics