"""
Loss Functions
==============
Label Smoothing Cross-Entropy with optional class-weight correction.
Prevents overconfident predictions; especially useful on small datasets.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.

    Args:
        smoothing : float in [0, 1)  —  0.0 = standard cross-entropy
        weight    : optional class-weight Tensor [num_classes]
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        smoothing: float                  = 0.1,
        weight:    Optional[torch.Tensor] = None,
        reduction: str                    = "mean",
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.weight    = weight
        self.reduction = reduction

    def forward(
        self,
        logits:  torch.Tensor,   # [B, C]
        targets: torch.Tensor,   # [B]  long
    ) -> torch.Tensor:
        B, C = logits.shape

        log_probs = F.log_softmax(logits, dim=-1)                     # [B, C]

        # Smoothed target distribution
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / max(C - 1, 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth * log_probs).sum(dim=-1)                      # [B]

        # Per-sample class weighting
        if self.weight is not None:
            w    = self.weight.to(logits.device)
            loss = loss * w[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'


def build_criterion(
    config:        Dict,
    class_weights: Optional[torch.Tensor] = None,
) -> LabelSmoothingCrossEntropy:
    """
    Build loss from config.yaml settings.

    Args:
        config        : loaded config.yaml dict
        class_weights : optional [num_classes] tensor
    """
    return LabelSmoothingCrossEntropy(
        smoothing = config["training"]["label_smoothing"],
        weight    = class_weights,
    )