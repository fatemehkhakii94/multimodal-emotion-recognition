#!/usr/bin/env python
"""
scripts/02_train.py  (updated — uses cached features)
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset            import (
    build_cached_dataloaders, CachedEmotionDataset
)
from src.models.multimodal_model import CachedMultimodalEmotionModel
from src.training.losses         import build_criterion
from src.training.trainer        import Trainer, configure_model_for_mode

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENTS = {
    "cross_modal": {"fusion": "cross_modal", "mode": "multimodal"},
    "early"      : {"fusion": "early",       "mode": "multimodal"},
    "late"       : {"fusion": "late",        "mode": "multimodal"},
    "audio_only" : {"fusion": "cross_modal", "mode": "audio_only"},
    "visual_only": {"fusion": "cross_modal", "mode": "visual_only"},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def run_experiment(name: str, cfg: dict, device: torch.device) -> dict:
    exp_cfg = EXPERIMENTS[name]
    fusion  = exp_cfg["fusion"]
    mode    = exp_cfg["mode"]

    logger.info(f"\n{'#'*62}")
    logger.info(f"  EXPERIMENT : {name}  (fusion={fusion}  mode={mode})")
    logger.info(f"{'#'*62}")

    loaders   = build_cached_dataloaders(cfg, cfg["data"]["splits_dir"])
    train_csv = Path(cfg["data"]["splits_dir"]) / "train.csv"
    train_ds  = CachedEmotionDataset(str(train_csv), "train", cfg)
    class_w   = train_ds.get_class_weights().to(device)

    model     = CachedMultimodalEmotionModel(cfg, fusion_type=fusion).to(device)
    criterion = build_criterion(cfg, class_weights=class_w)

    trainer = Trainer(
        model     = model,
        config    = cfg,
        device    = device,
        run_name  = name,
        criterion = criterion,
    )

    # ── Patch trainer to use cached batch keys ─────────────────────
    trainer._is_cached = True

    best = trainer.fit(
        train_loader = loaders["train"],
        val_loader   = loaders["val"],
        mode         = mode,
    )

    logger.info(
        f"✅  [{name}]  "
        f"F1={best['f1_weighted']:.4f}  "
        f"UAR={best['uar']:.4f}  "
        f"Acc={100*best['acc']:.2f}%"
    )
    return best


def print_summary(results: dict):
    logger.info(f"\n{'='*66}")
    logger.info(f"{'EXPERIMENT':<15} {'Acc':>9} {'UAR':>9} {'F1-W':>9} {'F1-M':>9}")
    logger.info("-" * 66)
    for name, m in results.items():
        logger.info(
            f"{name:<15} "
            f"{100*m['acc']:>8.2f}%  "
            f"{m['uar']:>9.4f}  "
            f"{m['f1_weighted']:>9.4f}  "
            f"{m['f1_macro']:>9.4f}"
        )
    logger.info("=" * 66)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument(
        "--experiment", default="cross_modal",
        choices=list(EXPERIMENTS) + ["all"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["num_epochs"] = args.epochs

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0)}")

    to_run  = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    results = {}

    for exp_name in to_run:
        results[exp_name] = run_experiment(exp_name, cfg, device)
        torch.cuda.empty_cache()

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()