"""
Evaluation Metrics
==================
UAR  (Unweighted Average Recall)  — standard SER benchmark metric
Weighted-F1, Macro-F1, Accuracy, per-class F1
"""

import numpy as np
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

EMOTION_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Unweighted Average Recall — standard metric in SER literature.
    Average per-class recall, ignoring class support size.
    """
    cm               = confusion_matrix(y_true, y_pred)
    per_class_recall = cm.diagonal() / cm.sum(axis=1).clip(min=1e-9)
    return float(per_class_recall.mean())


def compute_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute full evaluation metrics dict.

    Args:
        y_true       : ground-truth class indices [N]
        y_pred       : predicted  class indices   [N]
        label_names  : list of class name strings

    Returns dict keys:
        acc, uar, f1_weighted, f1_macro,
        f1_per_class (ndarray), per_class_f1_dict,
        precision_weighted, recall_weighted,
        confusion_matrix (ndarray)
    """
    if label_names is None:
        label_names = EMOTION_NAMES

    acc         = float(accuracy_score(y_true, y_pred))
    uar         = compute_uar(y_true, y_pred)
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_pc       = f1_score(y_true, y_pred, average=None, zero_division=0)   # ndarray
    prec_w      = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_w       = float(recall_score(y_true,    y_pred, average="weighted", zero_division=0))
    cm          = confusion_matrix(y_true, y_pred)

    per_class_f1_dict = {
        label_names[i]: float(f1_pc[i])
        for i in range(min(len(f1_pc), len(label_names)))
    }

    return {
        "acc"               : acc,
        "uar"               : uar,
        "f1_weighted"       : f1_weighted,
        "f1_macro"          : f1_macro,
        "f1_per_class"      : f1_pc,
        "per_class_f1_dict" : per_class_f1_dict,
        "precision_weighted": prec_w,
        "recall_weighted"   : rec_w,
        "confusion_matrix"  : cm,
    }