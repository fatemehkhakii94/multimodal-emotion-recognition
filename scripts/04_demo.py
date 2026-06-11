#!/usr/bin/env python
"""
scripts/04_demo.py
==================
Emotion recognition demo — creates rich visualizations.

Modes
-----
  dataset : sample N clips from test set, predict, visualize  (default)
  file    : predict on a single video file

Usage
-----
  python scripts/04_demo.py                                          # default: 18 test samples
  python scripts/04_demo.py --n-samples 30 --run cross_modal
  python scripts/04_demo.py --mode file --input path/to/video.flv
  python scripts/04_demo.py --mode file  # auto-picks a random test clip
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict import EmotionPredictor, EMOTION_NAMES, EMOTION_LABELS, EMOTION_COLORS

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

BG_DARK  = "#1a1a2e"
BG_MID   = "#16213e"
BG_LIGHT = "#0f3460"


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_prediction(
    faces:       List[np.ndarray],      # RGB uint8
    wav:         np.ndarray,
    sample_rate: int,
    prediction:  Dict,
    true_label:  Optional[str],
    filename:    str,
    output_path: str,
) -> None:
    """
    Dark-themed visualization with:
      - Row of face frames
      - Audio waveform
      - Emotion probability bar chart
      - Prediction summary
    """
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(BG_DARK)

    gs = gridspec.GridSpec(
        3, 8,
        figure  = fig,
        hspace  = 0.5,
        wspace  = 0.35,
        left    = 0.04,
        right   = 0.97,
        top     = 0.88,
        bottom  = 0.08,
    )

    correct = (true_label == prediction["emotion"]) if true_label else None
    status  = " ✓" if correct else (" ✗" if correct is not None else "")
    tcolor  = "#2ECC71" if correct else ("#E74C3C" if correct is not None else "#3498DB")

    title = f"{filename}"
    if true_label:
        title += (
            f"   |   True: {true_label} ({EMOTION_LABELS[true_label]})"
            f"   |   Predicted: {prediction['emotion']} ({prediction['emotion_full']}){status}"
        )
    else:
        title += (
            f"   |   Predicted: {prediction['emotion']} ({prediction['emotion_full']})"
            f"   |   Confidence: {prediction['confidence']*100:.1f}%"
        )
    fig.suptitle(title, fontsize=12, color=tcolor, fontweight="bold", y=0.97)

    # ── Face frames ───────────────────────────────────────────────
    n_show  = min(8, len(faces))
    indices = np.linspace(0, len(faces) - 1, n_show, dtype=int) if faces else []

    for col_i in range(8):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(BG_MID)
        if col_i < n_show and len(faces) > 0:
            ax.imshow(faces[int(indices[col_i])])
            ax.set_title(f"f{int(indices[col_i])}", fontsize=6, color="#aaa", pad=1)
        else:
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, color="#555", fontsize=12)
        ax.axis("off")

    # ── Audio waveform ────────────────────────────────────────────
    ax_wav = fig.add_subplot(gs[1, :5])
    ax_wav.set_facecolor(BG_MID)
    if len(wav) > 0:
        t = np.linspace(0, len(wav) / sample_rate, len(wav))
        # Downsample for plotting speed
        step = max(1, len(wav) // 4000)
        ax_wav.plot(t[::step], wav[::step], color="#3498DB", linewidth=0.5, alpha=0.9)
        ax_wav.fill_between(t[::step], wav[::step], alpha=0.15, color="#3498DB")
    ax_wav.set_xlabel("Time (s)", color="#bbb", fontsize=8)
    ax_wav.set_ylabel("Amplitude", color="#bbb", fontsize=8)
    ax_wav.set_title("Audio Waveform", color="white", fontsize=10, fontweight="bold")
    ax_wav.tick_params(colors="#888", labelsize=7)
    for spine in ax_wav.spines.values():
        spine.set_color("#333")

    # ── Probability bar chart ─────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 5:])
    ax_bar.set_facecolor(BG_MID)
    probs  = prediction["probs"]
    labels = list(probs.keys())
    values = [probs[l] for l in labels]
    colors = [EMOTION_COLORS[l] for l in labels]

    bars = ax_bar.barh(
        [EMOTION_LABELS[l] for l in labels],
        values, color=colors, alpha=0.85, height=0.55,
    )

    # Highlight predicted
    pred_name  = prediction["emotion"]
    for bar, lbl, val in zip(bars, labels, values):
        lw = 2.5 if lbl == pred_name else 0
        bar.set_linewidth(lw)
        bar.set_edgecolor("white")
        ax_bar.text(
            min(val + 0.02, 0.92),
            bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center", ha="left", fontsize=7.5, color="white",
        )

    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Probability", color="#bbb", fontsize=8)
    ax_bar.set_title("Emotion Probabilities", color="white", fontsize=10, fontweight="bold")
    ax_bar.tick_params(colors="#bbb", labelsize=8)
    ax_bar.invert_yaxis()
    for spine in ax_bar.spines.values():
        spine.set_color("#333")

    # ── Confidence summary bar ─────────────────────────────────────
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.set_facecolor(BG_LIGHT)
    ax_info.axis("off")
    conf = prediction["confidence"]
    conf_color = "#2ECC71" if conf > 0.7 else "#F1C40F" if conf > 0.4 else "#E74C3C"
    ax_info.text(
        0.5, 0.5,
        f"Emotion: {prediction['emotion']}  ({prediction['emotion_full']})    "
        f"Confidence: {conf*100:.1f}%    "
        f"{'Correct ✓' if correct else ('Wrong ✗' if correct is not None else '')}",
        ha="center", va="center", fontsize=12,
        color=conf_color, fontweight="bold",
        transform=ax_info.transAxes,
    )

    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid
# ─────────────────────────────────────────────────────────────────────────────

def create_summary_grid(results: List[Dict], output_path: str) -> None:
    """Grid overview: one cell per sample, shows face + true/pred labels."""
    n      = len(results)
    n_cols = min(6, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
    fig.patch.set_facecolor(BG_DARK)
    fig.suptitle(
        "Emotion Recognition — Test Set Predictions",
        fontsize=15, color="white", fontweight="bold", y=1.02,
    )

    # Flatten axes
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i, res in enumerate(results):
        ax = axes_flat[i]
        ax.set_facecolor(BG_MID)

        # Show middle face
        if res.get("faces"):
            mid = res["faces"][len(res["faces"]) // 2]
            ax.imshow(mid)
        else:
            ax.text(0.5, 0.5, "No face", ha="center", va="center",
                    transform=ax.transAxes, color="#555")

        correct    = res["true"] == res["pred"]
        color      = "#2ECC71" if correct else "#E74C3C"
        icon       = "✓" if correct else "✗"
        pred_color = EMOTION_COLORS.get(res["pred"], "#fff")

        ax.set_title(
            f"True: {res['true']}  Pred: {res['pred']} {icon}\n"
            f"Conf: {res['conf']*100:.0f}%  "
            f"({res.get('filename', '')[:14]})",
            fontsize=7, color=color, pad=3,
        )
        for spine in ax.spines.values():
            spine.set_color(pred_color)
            spine.set_linewidth(2.5)
        ax.axis("off")

    # Hide empty cells
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy by emotion bar chart
# ─────────────────────────────────────────────────────────────────────────────

def create_demo_accuracy_chart(results: List[Dict], output_path: str) -> None:
    """Per-class accuracy bar chart from demo samples."""
    from collections import defaultdict
    correct_per = defaultdict(int)
    total_per   = defaultdict(int)

    for r in results:
        total_per[r["true"]] += 1
        if r["true"] == r["pred"]:
            correct_per[r["true"]] += 1

    emotions = sorted(total_per.keys())
    accs     = [correct_per[e] / total_per[e] for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#aaa") for e in emotions]
    labels   = [f"{e}\n({EMOTION_LABELS[e]})" for e in emotions]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_MID)

    bars = ax.bar(labels, [a * 100 for a in accs], color=colors, alpha=0.85, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc*100:.0f}%", ha="center", va="bottom",
            fontsize=10, color="white", fontweight="bold",
        )

    overall = sum(correct_per.values()) / max(sum(total_per.values()), 1)
    ax.axhline(overall * 100, color="#F1C40F", linestyle="--",
               linewidth=1.5, label=f"Overall: {overall*100:.1f}%")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=11)
    ax.set_title("Per-Class Accuracy (Demo Samples)", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    ax.legend(fontsize=9, facecolor=BG_MID, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset demo
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_demo(
    predictor:  EmotionPredictor,
    cfg:        Dict,
    n_samples:  int,
    output_dir: Path,
) -> None:
    test_csv = Path(cfg["data"]["splits_dir"]) / "test.csv"
    df       = pd.read_csv(test_csv)
    df       = df[df["label"] >= 0].reset_index(drop=True)

    # Sample evenly across all 6 emotions (pandas 3.x safe)
    n_per_class  = max(1, n_samples // 6)
    samples_list = []
    for emotion in sorted(df["emotion"].unique()):
        grp = df[df["emotion"] == emotion]
        samples_list.append(grp.sample(min(n_per_class, len(grp)), random_state=42))
    samples_df = (
        pd.concat(samples_list, ignore_index=True)
          .sample(frac=1, random_state=42)
          .reset_index(drop=True)
    )
    logger.info(f"\nRunning demo on {len(samples_df)} test samples …\n")

    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    results   = []
    n_correct = 0

    for _, row in samples_df.iterrows():
        fname      = str(row["filename"])
        true_label = str(row["emotion"])

        result = predictor.predict_from_processed(
            audio_path = str(row["audio_path"]),
            frames_dir = str(row["frames_dir"]) if pd.notna(row.get("frames_dir")) else "",
        )

        correct = result["emotion"] == true_label
        if correct:
            n_correct += 1
        icon = "✓" if correct else "✗"

        logger.info(
            f"  {fname:<30}  "
            f"True={true_label}  Pred={result['emotion']}  "
            f"Conf={result['confidence']*100:.1f}%  {icon}"
        )

        # Individual visualization
        out_png = str(sample_dir / f"{fname}.png")
        visualize_prediction(
            faces       = result.get("faces", []),
            wav         = result.get("wav",   np.zeros(100)),
            sample_rate = predictor.sample_rate,
            prediction  = result,
            true_label  = true_label,
            filename    = fname,
            output_path = out_png,
        )

        results.append({
            "filename": fname,
            "true"    : true_label,
            "pred"    : result["emotion"],
            "conf"    : result["confidence"],
            "faces"   : result.get("faces", []),
        })

    # Aggregate plots
    create_summary_grid(
        results,
        str(output_dir / "summary_grid.png"),
    )
    create_demo_accuracy_chart(
        results,
        str(output_dir / "demo_accuracy_by_class.png"),
    )

    overall = n_correct / max(len(results), 1)
    logger.info(
        f"\n{'='*55}\n"
        f"  Demo Accuracy: {n_correct}/{len(results)} = {overall*100:.1f}%\n"
        f"  Outputs saved to: {output_dir}/\n"
        f"{'='*55}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# File demo
# ─────────────────────────────────────────────────────────────────────────────

def run_file_demo(
    predictor:  EmotionPredictor,
    video_path: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing: {video_path}")

    result = predictor.predict_from_file(video_path)

    logger.info(f"\n  Predicted : {result['emotion']}  ({result['emotion_full']})")
    logger.info(f"  Confidence: {result['confidence']*100:.1f}%")
    logger.info("  All probabilities:")
    for emo, prob in sorted(result["probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 25)
        logger.info(f"    {emo}  ({EMOTION_LABELS[emo]:<8})  {prob*100:5.1f}%  {bar}")

    fname   = Path(video_path).stem
    out_png = str(output_dir / f"prediction_{fname}.png")
    visualize_prediction(
        faces       = result.get("faces", []),
        wav         = result.get("wav",   np.zeros(100)),
        sample_rate = predictor.sample_rate,
        prediction  = result,
        true_label  = None,
        filename    = fname,
        output_path = out_png,
    )
    logger.info(f"\n✅  Visualization saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--run", default="cross_modal",
        choices=["cross_modal", "early", "late", "audio_only", "visual_only"],
    )
    parser.add_argument(
        "--mode", default="dataset",
        choices=["dataset", "file"],
    )
    parser.add_argument("--n-samples", type=int, default=18)
    parser.add_argument("--input",     type=str, default=None,
                        help="Video file path for --mode file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["paths"]["results"]) / "demo" / args.run
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = EmotionPredictor(cfg, run_name=args.run)

    try:
        if args.mode == "dataset":
            run_dataset_demo(predictor, cfg, args.n_samples, output_dir)

        elif args.mode == "file":
            if args.input:
                video_path = args.input
            else:
                # Auto-pick a random test clip from raw data
                test_csv   = Path(cfg["data"]["splits_dir"]) / "test.csv"
                df         = pd.read_csv(test_csv).sample(1, random_state=7)
                fname      = df.iloc[0]["filename"]
                video_path = str(
                    Path(cfg["data"]["raw_dir"]) / "VideoFlash" / f"{fname}.flv"
                )
                logger.info(f"No --input given. Using: {video_path}")
            run_file_demo(predictor, video_path, output_dir)
    finally:
        predictor.close()


if __name__ == "__main__":
    main()