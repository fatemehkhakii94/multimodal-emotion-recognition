#!/usr/bin/env python
"""
scripts/03_evaluate.py
======================
Load best checkpoints, run test-set evaluation,
generate all plots and CSV reports.

Usage
-----
  python scripts/03_evaluate.py                          # all experiments
  python scripts/03_evaluate.py --experiment cross_modal
  python scripts/03_evaluate.py --no-tsne               # skip t-SNE (slow)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (
    CachedEmotionDataset,
    cached_collate_fn,
)
from src.evaluation.metrics   import compute_metrics, EMOTION_NAMES
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_modality_comparison,
    plot_training_curves,
    plot_tsne,
    plot_roc_curves,
    save_metrics_csv,
)
from src.models.multimodal_model import CachedMultimodalEmotionModel

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


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_test_inference(
    model:        torch.nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    mode:         str,
    collect_embs: bool = True,
) -> dict:
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []
    audio_embs  = []
    visual_embs = []

    for batch in tqdm(loader, desc=f"  Inference [{mode}]", ncols=80):
        audio_feat = batch["audio_feat"].to(device, non_blocking=True)
        frame_feat = batch["frame_feat"].to(device, non_blocking=True)
        labels     = batch["label"]

        logits, extra = model(audio_feat, frame_feat, mode=mode)
        probs         = F.softmax(logits.float(), dim=-1)

        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

        if collect_embs:
            if extra.get("audio_emb") is not None:
                audio_embs.append(extra["audio_emb"].cpu().float())
            if extra.get("visual_emb") is not None:
                visual_embs.append(extra["visual_emb"].cpu().float())

    return {
        "preds"      : torch.cat(all_preds).numpy(),
        "labels"     : torch.cat(all_labels).numpy(),
        "probs"      : torch.cat(all_probs).numpy(),
        "audio_embs" : torch.cat(audio_embs).numpy()  if audio_embs  else None,
        "visual_embs": torch.cat(visual_embs).numpy() if visual_embs else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one experiment
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(
    name:        str,
    cfg:         dict,
    device:      torch.device,
    test_loader: DataLoader,
    run_tsne:    bool,
    plots_dir:   Path,
    metrics_dir: Path,
    log_dir:     str,
) -> dict:
    exp_cfg   = EXPERIMENTS[name]
    fusion    = exp_cfg["fusion"]
    mode      = exp_cfg["mode"]
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / f"{name}_best.pt"

    if not ckpt_path.exists():
        logger.warning(f"  ⚠  Checkpoint not found: {ckpt_path}  — skipping")
        return {}

    logger.info(f"\n{'─'*55}")
    logger.info(f"  Evaluating: {name}  (fusion={fusion}, mode={mode})")
    logger.info(f"{'─'*55}")

    # ── Load model ────────────────────────────────────────────────
    model = CachedMultimodalEmotionModel(cfg, fusion_type=fusion).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    logger.info(f"  Loaded checkpoint  epoch={state['epoch']}")

    # ── Inference ─────────────────────────────────────────────────
    out = run_test_inference(
        model, test_loader, device, mode, collect_embs=run_tsne
    )

    # ── Metrics ───────────────────────────────────────────────────
    metrics          = compute_metrics(out["labels"], out["preds"])
    metrics["probs"] = out["probs"]

    logger.info(
        f"  Test  →  "
        f"Acc={100*metrics['acc']:.2f}%  "
        f"UAR={metrics['uar']:.4f}  "
        f"F1-W={metrics['f1_weighted']:.4f}  "
        f"F1-M={metrics['f1_macro']:.4f}"
    )
    logger.info("  Per-class F1:")
    for cls, f1 in metrics["per_class_f1_dict"].items():
        bar = "█" * int(f1 * 20)
        logger.info(f"    {cls}  {f1:.4f}  {bar}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_confusion_matrix(
        metrics["confusion_matrix"], name,
        str(plots_dir / "confusion_matrices"),
    )
    plot_roc_curves(
        out["labels"], out["probs"], name,
        str(plots_dir / "roc"),
    )
    plot_training_curves(
        log_dir, name,
        str(plots_dir / "training_curves"),
    )

    # ── t-SNE ─────────────────────────────────────────────────────
    if run_tsne:
        tsne_dir = str(plots_dir / "tsne")
        if out["audio_embs"] is not None:
            plot_tsne(
                out["audio_embs"], out["labels"],
                f"Audio Embeddings — {name}", tsne_dir,
            )
        if out["visual_embs"] is not None:
            plot_tsne(
                out["visual_embs"], out["labels"],
                f"Visual Embeddings — {name}", tsne_dir,
            )
        if out["audio_embs"] is not None and out["visual_embs"] is not None:
            fused = np.concatenate([out["audio_embs"], out["visual_embs"]], axis=-1)
            plot_tsne(
                fused, out["labels"],
                f"Fused Embeddings — {name}", tsne_dir,
            )

    del model
    torch.cuda.empty_cache()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument(
        "--experiment", default="all",
        choices=list(EXPERIMENTS) + ["all"],
    )
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE (faster evaluation)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = Path(cfg["paths"]["results"]) / "plots"
    met_dir   = Path(cfg["paths"]["results"]) / "metrics"
    (plots_dir / "roc").mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    # ── Build test loader (shared across all experiments) ─────────
    test_csv = Path(cfg["data"]["splits_dir"]) / "test.csv"
    test_ds  = CachedEmotionDataset(str(test_csv), "test", cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["training"]["num_workers"],
        collate_fn  = cached_collate_fn,
        pin_memory  = True,
    )
    logger.info(f"Test set: {len(test_ds)} samples  →  {len(test_loader)} batches")

    to_run       = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    results_dict = {}

    for name in to_run:
        m = evaluate_experiment(
            name        = name,
            cfg         = cfg,
            device      = device,
            test_loader = test_loader,
            run_tsne    = not args.no_tsne,
            plots_dir   = plots_dir,
            metrics_dir = met_dir,
            log_dir     = cfg["paths"]["logs"],
        )
        if m:
            results_dict[name] = m

    if not results_dict:
        logger.error("No results found. Make sure training completed.")
        return

    # ── Cross-model comparison plots ──────────────────────────────
    if len(results_dict) > 1:
        plot_per_class_f1(results_dict,        str(plots_dir))
        plot_modality_comparison(results_dict, str(plots_dir))

    # ── CSV export ────────────────────────────────────────────────
    save_metrics_csv(results_dict, str(met_dir))

    # ── Final summary ─────────────────────────────────────────────
    logger.info(f"\n{'='*66}")
    logger.info(f"{'EXPERIMENT':<15} {'Acc':>9} {'UAR':>9} {'F1-W':>9} {'F1-M':>9}")
    logger.info("-" * 66)
    for name, m in results_dict.items():
        logger.info(
            f"{name:<15} "
            f"{100*m['acc']:>8.2f}%  "
            f"{m['uar']:>9.4f}  "
            f"{m['f1_weighted']:>9.4f}  "
            f"{m['f1_macro']:>9.4f}"
        )
    logger.info("=" * 66)
    logger.info(f"\nAll results saved to:  {cfg['paths']['results']}/")


if __name__ == "__main__":
    main()