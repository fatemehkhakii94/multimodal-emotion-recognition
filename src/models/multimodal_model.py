"""
MultimodalEmotionModel
======================
Two variants:
  MultimodalEmotionModel       — end-to-end (raw audio + frames)
  CachedMultimodalEmotionModel — uses pre-extracted features (fast training)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .visual_branch import VisualBranch
from .audio_branch  import AudioBranch
from .fusion        import build_fusion, ClassificationHead

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Original end-to-end model (kept for inference / demo)
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalEmotionModel(nn.Module):
    def __init__(self, config: Dict, fusion_type: str = "cross_modal"):
        super().__init__()

        self.fusion_type = fusion_type
        num_classes = config["classifier"]["num_classes"]
        hidden_dim  = config["classifier"]["hidden_dim"]
        dropout     = config["classifier"]["dropout"]
        audio_dim   = config["audio"]["output_dim"]
        visual_dim  = config["visual"]["output_dim"]

        self.visual_branch    = VisualBranch(config)
        self.audio_branch     = AudioBranch(config)
        self.fusion           = build_fusion(fusion_type, config)
        self.audio_only_head  = ClassificationHead(audio_dim,  hidden_dim, num_classes, dropout)
        self.visual_only_head = ClassificationHead(visual_dim, hidden_dim, num_classes, dropout)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*50}\n"
            f"  MultimodalEmotionModel  [{fusion_type}]\n"
            f"  Total params     : {total:,}\n"
            f"  Trainable params : {trainable:,}\n"
            f"{'='*50}"
        )

    def forward(
        self,
        audio:  torch.Tensor,
        frames: torch.Tensor,
        mode:   str = "multimodal",
    ) -> Tuple[torch.Tensor, Dict]:
        audio_emb    = None
        visual_emb   = None
        attn_weights = None

        if mode in ("multimodal", "audio_only"):
            audio_emb = self.audio_branch(audio)
        if mode in ("multimodal", "visual_only"):
            visual_emb = self.visual_branch(frames)

        if mode == "audio_only":
            logits = self.audio_only_head(audio_emb)
        elif mode == "visual_only":
            logits = self.visual_only_head(visual_emb)
        else:
            logits, attn_weights = self.fusion(audio_emb, visual_emb)

        return logits, {
            "audio_emb"   : audio_emb,
            "visual_emb"  : visual_emb,
            "attn_weights": attn_weights,
        }

    @torch.no_grad()
    def get_embeddings(self, audio, frames):
        return {
            "audio_emb" : self.audio_branch(audio),
            "visual_emb": self.visual_branch(frames),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fast cached model (trains on pre-extracted features)
# ─────────────────────────────────────────────────────────────────────────────

class CachedMultimodalEmotionModel(nn.Module):
    """
    Lightweight model that trains on pre-extracted backbone features.

    Input:
        audio_feat : [B, 1536]           HuBERT mean+std pooled
        frame_feat : [B, num_frames, 1536]  EfficientNet per-frame features

    Architecture:
        Audio   : 1536 → LayerNorm → Linear(256) → audio_emb [B,256]
        Visual  : Linear(256) per frame → CLS Transformer → visual_emb [B,256]
        Fusion  : early | late | cross_modal
    """

    def __init__(self, config: Dict, fusion_type: str = "cross_modal"):
        super().__init__()

        self.fusion_type = fusion_type
        num_frames  = config["visual"]["num_frames"]
        embed_dim   = config["visual"]["temporal_embed_dim"]   # 256
        num_heads   = config["visual"]["temporal_num_heads"]   # 8
        num_layers  = config["visual"]["temporal_num_layers"]  # 4
        dropout     = config["visual"]["temporal_dropout"]     # 0.1
        num_classes = config["classifier"]["num_classes"]      # 6
        hidden_dim  = config["classifier"]["hidden_dim"]       # 256
        cls_dropout = config["classifier"]["dropout"]          # 0.3

        hubert_feat_dim = 1536   # mean + std of 768
        effnet_feat_dim = 1536   # EfficientNet-B3 global avg pool

        # ── Audio projection ──────────────────────────────────────
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(hubert_feat_dim),
            nn.Linear(hubert_feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Visual: per-frame projection + temporal transformer ───
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(effnet_feat_dim),
            nn.Linear(effnet_feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.visual_out_proj      = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── Fusion module ─────────────────────────────────────────
        self.fusion = build_fusion(fusion_type, config)

        # ── Unimodal heads (ablation) ─────────────────────────────
        self.audio_only_head  = ClassificationHead(embed_dim, hidden_dim, num_classes, cls_dropout)
        self.visual_only_head = ClassificationHead(embed_dim, hidden_dim, num_classes, cls_dropout)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*50}\n"
            f"  CachedMultimodalEmotionModel  [{fusion_type}]\n"
            f"  Total/Trainable  : {total:,} / {trainable:,}\n"
            f"{'='*50}"
        )

    def encode_audio(self, audio_feat: torch.Tensor) -> torch.Tensor:
        """[B, 1536] → [B, 256]"""
        return self.audio_proj(audio_feat)

    def encode_visual(self, frame_feat: torch.Tensor) -> torch.Tensor:
        """[B, T, 1536] → [B, 256]"""
        B, T, _ = frame_feat.shape
        x       = self.frame_proj(frame_feat)                  # [B, T, 256]
        cls     = self.cls_token.expand(B, -1, -1)
        x       = torch.cat([cls, x], dim=1) + self.pos_embedding
        x       = self.temporal_transformer(x)
        return self.visual_out_proj(x[:, 0, :])               # CLS token

    def forward(
        self,
        audio_feat: torch.Tensor,
        frame_feat: torch.Tensor,
        mode:       str = "multimodal",
    ) -> Tuple[torch.Tensor, Dict]:
        audio_emb  = None
        visual_emb = None

        if mode in ("multimodal", "audio_only"):
            audio_emb = self.encode_audio(audio_feat)           # [B, 256]
        if mode in ("multimodal", "visual_only"):
            visual_emb = self.encode_visual(frame_feat)         # [B, 256]

        if mode == "audio_only":
            logits = self.audio_only_head(audio_emb)
            attn   = None
        elif mode == "visual_only":
            logits = self.visual_only_head(visual_emb)
            attn   = None
        else:
            logits, attn = self.fusion(audio_emb, visual_emb)

        return logits, {
            "audio_emb"   : audio_emb,
            "visual_emb"  : visual_emb,
            "attn_weights": attn,
        }