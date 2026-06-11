"""
Visual Branch
=============
EfficientNet-B3  →  Frame Projection  →  Temporal Transformer  →  CLS token
Input : [B, T, C, H, W]
Output: [B, output_dim]
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class VisualBranch(nn.Module):
    """
    Per-frame feature extraction with EfficientNet-B3,
    followed by a CLS-token Transformer for temporal aggregation.
    """

    def __init__(self, config: Dict):
        super().__init__()

        variant    = config["visual"]["efficientnet_variant"]  # efficientnet_b3
        pretrained = config["visual"]["pretrained"]
        feat_dim   = config["visual"]["visual_feature_dim"]    # 1536
        embed_dim  = config["visual"]["temporal_embed_dim"]    # 256
        num_heads  = config["visual"]["temporal_num_heads"]    # 8
        num_layers = config["visual"]["temporal_num_layers"]   # 4
        dropout    = config["visual"]["temporal_dropout"]      # 0.1
        output_dim = config["visual"]["output_dim"]            # 256

        self.num_frames = config["visual"]["num_frames"]       # 15
        self.embed_dim  = embed_dim

        # ── EfficientNet-B3 backbone (frame-level features) ───────
        self.backbone = timm.create_model(
            variant,
            pretrained   = pretrained,
            num_classes  = 0,       # remove classification head
            global_pool  = "avg",   # output: [B, 1536]
        )
        self._freeze_early_layers()

        # ── Frame projection: 1536 → 256 ─────────────────────────
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── CLS token ─────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ── Learnable positional encoding (T + 1 for CLS) ────────
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_frames + 1, embed_dim) * 0.02
        )

        # ── Temporal Transformer Encoder ──────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        # ── Output projection ─────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"VisualBranch  —  {n_params:,} trainable parameters")

    # ─────────────────────────────────────────────────────────────

    def _freeze_early_layers(self):
        """Freeze EfficientNet stem + first 2 blocks (generic low-level features)."""
        freeze_prefixes = ["conv_stem", "bn1", "blocks.0", "blocks.1"]
        n_frozen = 0
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(p) for p in freeze_prefixes):
                param.requires_grad = False
                n_frozen += 1
        logger.info(f"VisualBranch  —  froze {n_frozen} backbone parameter tensors")

    # ─────────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : Tensor [B, T, C, H, W]
        Returns:
            visual_emb : Tensor [B, output_dim]
        """
        B, T, C, H, W = frames.shape

        # ── Per-frame feature extraction (batched over B*T) ───────
        feats = self.backbone(frames.view(B * T, C, H, W))  # [B*T, 1536]
        feats = feats.view(B, T, -1)                         # [B,   T, 1536]

        # ── Project to embed_dim ──────────────────────────────────
        x = self.dropout(self.frame_proj(feats))             # [B, T, 256]

        # ── Prepend CLS token ─────────────────────────────────────
        cls   = self.cls_token.expand(B, -1, -1)            # [B, 1,   256]
        x     = torch.cat([cls, x], dim=1)                   # [B, T+1, 256]

        # ── Positional embedding ──────────────────────────────────
        x = x + self.pos_embedding                           # [B, T+1, 256]

        # ── Temporal Transformer ──────────────────────────────────
        x = self.temporal_transformer(x)                     # [B, T+1, 256]

        # ── CLS token as temporal summary ─────────────────────────
        cls_out    = x[:, 0, :]                              # [B, 256]
        visual_emb = self.output_proj(cls_out)               # [B, 256]

        return visual_emb