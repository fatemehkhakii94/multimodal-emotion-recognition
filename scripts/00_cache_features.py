#!/usr/bin/env python
"""
scripts/00_cache_features.py
============================
Pre-extract HuBERT + EfficientNet features once.
~20-40 min to run. After this, training is ~50x faster.

Usage:
    python scripts/00_cache_features.py
"""

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt = "%H:%M:%S",
)

from src.data.preprocess import extract_and_cache_features

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

extract_and_cache_features(
    splits_dir    = cfg["data"]["splits_dir"],
    processed_dir = cfg["data"]["processed_dir"],
    config        = cfg,
    device        = "cuda",
)
print("\n✅  Feature caching complete.")
print("    Next: python scripts/02_train.py --experiment all")