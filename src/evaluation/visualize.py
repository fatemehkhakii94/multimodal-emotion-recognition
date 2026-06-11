"""
Visualization Suite
===================
All plots and result exports for the emotion recognition system.

Functions
---------
plot_confusion_matrix        — normalized heatmap per model
plot_per_class_f1            — grouped bar chart across all models
plot_modality_comparison     — summary bar chart (Acc / UAR / F1)
plot_training_curves         — loss & metric curves from TensorBoard logs
plot_tsne                    — 2D t-SNE of audio/visual/fused embeddings
plot_roc_curves              — one-vs-rest ROC + AUC per class
save_metrics_csv             — CSV export of all results
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "figure.dpi"       : 120,
    "savefig.dpi"      : 150,
    "savefig.bbox"     : "tight",
})

EMOTION_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
PALETTE       = sns.color_palette("tab10", 10)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm:          np.ndarray,
    run_name:    str,
    output_dir:  str,
    label_names: List[str] = EMOTION_NAMES,
    normalize:   bool      = True,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_plot,
        annot       = True,
        fmt         = fmt,
        cmap        = "Blues",
        xticklabels = label_names,
        yticklabels = label_names,
        linewidths  = 0.5,
        vmin        = 0,
        vmax        = vmax,
        ax          = ax,
        annot_kws   = {"size": 10},
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label",      fontweight="bold")
    ax.set_title(f"Confusion Matrix — {run_name}", fontweight="bold", pad=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    path = str(out / f"confusion_matrix_{run_name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-class F1 Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(
    results_dict: Dict[str, Dict],
    output_dir:   str,
    label_names:  List[str] = EMOTION_NAMES,
) -> str:
    """
    results_dict: {run_name: metrics_dict}
    Each metrics_dict must have 'per_class_f1_dict'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_names   = list(results_dict.keys())
    n_classes   = len(label_names)
    n_models    = len(run_names)
    bar_width   = 0.8 / n_models
    x           = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors  = PALETTE[:n_models]

    for i, (run, metrics) in enumerate(results_dict.items()):
        f1s    = [metrics["per_class_f1_dict"].get(c, 0.0) for c in label_names]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, f1s, bar_width * 0.9,
                        label=run, color=colors[i], alpha=0.85)
        for b, v in zip(bars, f1s):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xlabel("Emotion Class",      fontweight="bold")
    ax.set_ylabel("F1-Score",           fontweight="bold")
    ax.set_title("Per-Class F1 Score — All Models", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = str(out / "per_class_f1_all_models.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. Modality Comparison (summary bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_modality_comparison(
    results_dict: Dict[str, Dict],
    output_dir:   str,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        ("acc",         "Accuracy",    lambda v: v * 100),
        ("uar",         "UAR",         lambda v: v * 100),
        ("f1_weighted", "F1 Weighted", lambda v: v * 100),
        ("f1_macro",    "F1 Macro",    lambda v: v * 100),
    ]

    run_names = list(results_dict.keys())
    n_runs    = len(run_names)
    n_metrics = len(metrics_to_plot)
    x         = np.arange(n_runs)
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, n_runs * 2), 5.5))
    colors  = PALETTE[:n_metrics]

    for i, (key, label, fn) in enumerate(metrics_to_plot):
        vals   = [fn(results_dict[r].get(key, 0.0)) for r in run_names]
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width * 0.9,
                        label=label, color=colors[i], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.4,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xlabel("Experiment",  fontweight="bold")
    ax.set_ylabel("Score (%)",   fontweight="bold")
    ax.set_title("Model Comparison — Accuracy / UAR / F1", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = str(out / "modality_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training Curves  (read from TensorBoard event files)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    log_dir:    str,
    run_name:   str,
    output_dir: str,
) -> Optional[str]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard not importable — skipping training curves")
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_log = Path(log_dir) / run_name
    if not run_log.exists():
        logger.warning(f"TensorBoard log not found: {run_log}")
        return None

    ea = EventAccumulator(str(run_log))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    def _get(tag):
        if tag not in tags:
            return [], []
        events = ea.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Training Curves — {run_name}", fontweight="bold", y=1.01)

    pairs = [
        ("Loss/train", "Loss/val",       axes[0], "Loss"),
        ("Acc/train",  "Acc/val",        axes[1], "Accuracy"),
        ("Metrics/F1_weighted", "Metrics/UAR", axes[2], "Val Metrics"),
    ]

    for tag_a, tag_b, ax, ylabel in pairs:
        xa, ya = _get(tag_a)
        xb, yb = _get(tag_b)

        # Fix tag names — SummaryWriter uses dict keys for add_scalars
        # e.g. 'Loss/train' and 'Loss/val' are stored as 'Loss/train' etc.
        # Try alternate patterns
        if not xa:
            xa, ya = _get(tag_a.replace("/", "_"))
        if not xb:
            xb, yb = _get(tag_b.replace("/", "_"))

        # Try the format used in our trainer: "Loss train" etc.
        for alt_a in [tag_a, f"{ylabel.lower()}/train", f"{ylabel}/train"]:
            if not xa:
                xa, ya = _get(alt_a)
        for alt_b in [tag_b, f"{ylabel.lower()}/val", f"{ylabel}/val"]:
            if not xb:
                xb, yb = _get(alt_b)

        label_a = tag_a.split("/")[-1]
        label_b = tag_b.split("/")[-1]

        if xa:
            ax.plot(xa, ya, label=label_a, color="steelblue",  linewidth=1.8)
        if xb:
            ax.plot(xb, yb, label=label_b, color="darkorange", linewidth=1.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = str(out / f"training_curves_{run_name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. t-SNE Embedding Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(
    embeddings:  np.ndarray,
    labels:      np.ndarray,
    title:       str,
    output_dir:  str,
    label_names: List[str] = EMOTION_NAMES,
    perplexity:  int       = 30,
    max_samples: int       = 2000,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Sub-sample if needed
    if len(embeddings) > max_samples:
        idx        = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]

    logger.info(f"Running t-SNE on {len(embeddings)} samples …")
    tsne    = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    reduced = tsne.fit_transform(embeddings)

    fig, ax    = plt.subplots(figsize=(8, 7))
    colors     = PALETTE[:len(label_names)]
    class_ids  = sorted(np.unique(labels))

    for cid in class_ids:
        mask = labels == cid
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c       = [colors[cid]],
            label   = label_names[cid] if cid < len(label_names) else str(cid),
            s       = 18,
            alpha   = 0.65,
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    safe_title = title.replace(" ", "_").replace("/", "_")
    path       = str(out / f"tsne_{safe_title}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    run_name:    str,
    output_dir:  str,
    label_names: List[str] = EMOTION_NAMES,
) -> str:
    """
    Args:
        y_true  : [N]       integer class labels
        y_score : [N, C]    softmax probabilities
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_classes  = y_score.shape[1]
    y_bin      = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax    = plt.subplots(figsize=(7.5, 6.5))
    colors     = PALETTE[:n_classes]

    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(
            fpr, tpr,
            color     = colors[i],
            linewidth = 1.8,
            label     = f"{label_names[i]}  AUC={roc_auc:.3f}",
        )

    # Macro-average AUC
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontweight="bold")
    ax.set_title(f"ROC Curves (one-vs-rest) — {run_name}\nMean AUC={np.mean(aucs):.3f}",
                 fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = str(out / f"roc_curves_{run_name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 7. CSV Export
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(
    results_dict: Dict[str, Dict],
    output_dir:   str,
    label_names:  List[str] = EMOTION_NAMES,
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Summary CSV ───────────────────────────────────────────────
    rows = []
    for run, m in results_dict.items():
        row = {
            "experiment"      : run,
            "accuracy"        : round(m.get("acc",         0.0), 4),
            "uar"             : round(m.get("uar",         0.0), 4),
            "f1_weighted"     : round(m.get("f1_weighted", 0.0), 4),
            "f1_macro"        : round(m.get("f1_macro",    0.0), 4),
            "precision_w"     : round(m.get("precision_weighted", 0.0), 4),
            "recall_w"        : round(m.get("recall_weighted",    0.0), 4),
        }
        # Add per-class F1
        for cls in label_names:
            row[f"f1_{cls}"] = round(m.get("per_class_f1_dict", {}).get(cls, 0.0), 4)
        rows.append(row)

    df   = pd.DataFrame(rows)
    path = str(out / "all_models_summary.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")

    # ── Per-class F1 CSV ──────────────────────────────────────────
    pc_rows = []
    for run, m in results_dict.items():
        pc = m.get("per_class_f1_dict", {})
        pc_rows.append({"experiment": run, **{k: round(v, 4) for k, v in pc.items()}})

    df_pc   = pd.DataFrame(pc_rows)
    path_pc = str(out / "per_class_f1.csv")
    df_pc.to_csv(path_pc, index=False)
    logger.info(f"Saved: {path_pc}")

    return path