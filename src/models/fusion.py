"""
Fusion Modules
==============
Three fusion strategies:
  1. EarlyFusion       — concatenation + MLP
  2. LateFusion        — separate heads + learnable weighted average
  3. CrossModalFusion  — bidirectional cross-attention + gated MLP
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    LayerNorm → Linear → GELU → Dropout → Linear  →  logits [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Early Fusion
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusion(nn.Module):
    """
    Concatenate audio + visual embeddings → MLP classification head.
    Simplest fusion baseline.
    """

    def __init__(self, config: Dict):
        super().__init__()

        audio_dim   = config["audio"]["output_dim"]        # 256
        visual_dim  = config["visual"]["output_dim"]       # 256
        hidden_dim  = config["fusion"]["hidden_dim"]       # 256
        num_classes = config["classifier"]["num_classes"]  # 6
        dropout     = config["fusion"]["dropout"]          # 0.3

        self.head = ClassificationHead(
            input_dim   = audio_dim + visual_dim,          # 512
            hidden_dim  = hidden_dim,
            num_classes = num_classes,
            dropout     = dropout,
        )

    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Returns: (logits [B, C],  None)
        """
        fused  = torch.cat([audio_emb, visual_emb], dim=-1)  # [B, 512]
        logits = self.head(fused)
        return logits, None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Late Fusion
# ─────────────────────────────────────────────────────────────────────────────

class LateFusion(nn.Module):
    """
    Separate unimodal classifiers → learnable weighted average of logits.
    """

    def __init__(self, config: Dict):
        super().__init__()

        audio_dim   = config["audio"]["output_dim"]        # 256
        visual_dim  = config["visual"]["output_dim"]       # 256
        hidden_dim  = config["fusion"]["hidden_dim"]       # 256
        num_classes = config["classifier"]["num_classes"]  # 6
        dropout     = config["fusion"]["dropout"]          # 0.3

        self.audio_head  = ClassificationHead(audio_dim,  hidden_dim, num_classes, dropout)
        self.visual_head = ClassificationHead(visual_dim, hidden_dim, num_classes, dropout)

        # Learnable fusion weight: sigmoid keeps it in (0,1)
        # Init = 0.0  →  sigmoid(0.0) = 0.5  (equal weighting at start)
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """
        Returns: (logits [B, C],  None)
        """
        audio_logits  = self.audio_head(audio_emb)          # [B, C]
        visual_logits = self.visual_head(visual_emb)         # [B, C]

        w      = torch.sigmoid(self.fusion_weight)           # scalar in (0,1)
        logits = w * audio_logits + (1.0 - w) * visual_logits

        return logits, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Modal Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-modal attention:
      Audio  queries Visual  →  context-enriched audio  representation
      Visual queries Audio   →  context-enriched visual representation
    Gated concatenation → MLP classification head.

    Attention weights are returned for visualization.
    """

    def __init__(self, config: Dict):
        super().__init__()

        embed_dim   = config["visual"]["output_dim"]       # 256
        num_heads   = config["fusion"]["num_heads"]        # 8
        hidden_dim  = config["fusion"]["hidden_dim"]       # 256
        num_classes = config["classifier"]["num_classes"]  # 6
        dropout     = config["fusion"]["dropout"]          # 0.3

        # Audio attends to Visual
        self.audio_to_visual = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        # Visual attends to Audio
        self.visual_to_audio = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        self.norm_audio  = nn.LayerNorm(embed_dim)
        self.norm_visual = nn.LayerNorm(embed_dim)

        # Gated fusion: learn how much cross-modal context to use
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.Sigmoid(),
        )

        # Final classification head
        self.head = ClassificationHead(
            input_dim   = embed_dim * 2,                   # 512
            hidden_dim  = hidden_dim,
            num_classes = num_classes,
            dropout     = dropout,
        )

    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            audio_emb  : [B, 256]
            visual_emb : [B, 256]
        Returns:
            logits       : [B, num_classes]
            attn_weights : dict {'audio_to_visual', 'visual_to_audio'}
        """
        # Unsqueeze to seq_len=1 for MultiheadAttention: [B, 1, D]
        a = audio_emb.unsqueeze(1)                          # [B, 1, 256]
        v = visual_emb.unsqueeze(1)                         # [B, 1, 256]

        # Audio queries Visual → context-enriched audio
        a_ctx, a2v_w = self.audio_to_visual(query=a, key=v, value=v)
        a_out = self.norm_audio(a_ctx.squeeze(1) + audio_emb)   # [B, 256]

        # Visual queries Audio → context-enriched visual
        v_ctx, v2a_w = self.visual_to_audio(query=v, key=a, value=a)
        v_out = self.norm_visual(v_ctx.squeeze(1) + visual_emb) # [B, 256]

        # Gated fusion
        fused  = torch.cat([a_out, v_out], dim=-1)          # [B, 512]
        gate   = self.gate(fused)                            # [B, 512]
        fused  = gate * fused

        # Classify
        logits = self.head(fused)                            # [B, num_classes]

        attn_weights = {
            "audio_to_visual": a2v_w.detach().cpu(),        # [B, 1, 1]
            "visual_to_audio": v2a_w.detach().cpu(),
        }

        return logits, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

FUSION_REGISTRY = {
    "early"      : EarlyFusion,
    "late"       : LateFusion,
    "cross_modal": CrossModalFusion,
}


def build_fusion(fusion_type: str, config: Dict) -> nn.Module:
    """
    Instantiate a fusion module by name.

    Args:
        fusion_type : 'early' | 'late' | 'cross_modal'
        config      : loaded config.yaml dict

    Returns:
        Fusion nn.Module
    """
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion type '{fusion_type}'. "
            f"Choose from: {list(FUSION_REGISTRY)}"
        )
    module   = FUSION_REGISTRY[fusion_type](config)
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info(f"Fusion [{fusion_type}]  —  {n_params:,} trainable parameters")
    return module