#!/usr/bin/env python
"""
scripts/01_prepare_data.py
==========================
End-to-end data preparation for CREMA-D.

Usage
-----
Full pipeline (download + preprocess + split):
    python scripts/01_prepare_data.py

Skip download (dataset already in data/raw/CREMAD):
    python scripts/01_prepare_data.py --skip-download

Force full re-download and re-process:
    python scripts/01_prepare_data.py --force
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

# ── make src/ importable when running from project root ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download   import download_cremad, get_file_list
from src.data.preprocess import preprocess_dataset, create_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def banner(msg: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {msg}")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare CREMA-D dataset")
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download (dataset already present in data/raw/CREMAD)")
    p.add_argument("--force",         action="store_true",
                   help="Force re-download and re-process everything")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_dir       = cfg["data"]["raw_dir"]        # data/raw/CREMAD
    processed_dir = cfg["data"]["processed_dir"]  # data/processed
    splits_dir    = cfg["data"]["splits_dir"]      # data/splits
    emotion_map   = cfg["data"]["emotion_map"]

    # ── STEP 1 : Download ─────────────────────────────────────────────
    banner("STEP 1  —  Download CREMA-D")
    if args.skip_download:
        cremad_dir = raw_dir
        logger.info(f"Skipping download  →  using {cremad_dir}")
    else:
        cremad_dir = download_cremad(cremad_dir=raw_dir, force=args.force)

    # ── STEP 2 : Parse file list ──────────────────────────────────────
    banner("STEP 2  —  Parse File List")
    samples = get_file_list(cremad_dir)

    # ── STEP 3 : Preprocess ───────────────────────────────────────────
    banner("STEP 3  —  Extract Frames & Process Audio")
    logger.info("⚠  This step takes ~30-60 min for all 7,442 clips.")
    logger.info("   It is resumable: re-run safely if interrupted.\n")
    processed_samples = preprocess_dataset(
        samples       = samples,
        processed_dir = processed_dir,
        config        = cfg,
        resume        = not args.force,
    )

    # ── STEP 4 : Create splits ────────────────────────────────────────
    banner("STEP 4  —  Actor-Based Train / Val / Test Splits")
    dfs = create_splits(
        samples     = processed_samples,
        splits_dir  = splits_dir,
        emotion_map = emotion_map,
        train_ratio = cfg["data"]["train_ratio"],
        val_ratio   = cfg["data"]["val_ratio"],
        seed        = cfg["project"]["seed"],
    )

    # ── Final summary ─────────────────────────────────────────────────
    banner("DATA PREPARATION COMPLETE")
    total     = sum(len(df) for df in dfs.values())
    n_video   = sum(df["n_frames"].gt(0).sum() for df in dfs.values())
    n_audio   = total - int(n_video)

    for split, df in dfs.items():
        logger.info(f"  {split:5s}: {len(df):5d} samples  ({100*len(df)/total:.1f}%)")

    logger.info(f"\n  Multimodal (audio + video) : {n_video}")
    logger.info(f"  Audio-only                 : {n_audio}")
    logger.info(f"\n  Split CSVs  →  {splits_dir}/")
    logger.info(f"  Frames      →  {processed_dir}/frames/")
    logger.info(f"  Audio       →  {processed_dir}/audio/")
    logger.info("\n  Next step  →  python scripts/02_train.py")

    # Save JSON summary
    summary = {
        "total": total, "train": len(dfs["train"]),
        "val": len(dfs["val"]), "test": len(dfs["test"]),
        "multimodal": int(n_video), "audio_only": int(n_audio),
    }
    out_json = Path(splits_dir) / "dataset_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary     →  {out_json}")


if __name__ == "__main__":
    main()