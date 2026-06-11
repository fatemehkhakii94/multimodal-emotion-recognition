"""
Audio Branch
============
HuBERT-base  →  Mean+Std pooling  →  Projection head
Input : [B, max_samples]  (raw 16 kHz waveform)
Output: [B, output_dim]
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from transformers import HubertModel

logger = logging.getLogger(__name__)


class AudioBranch(nn.Module):
    """
    Speech emotion representation using HuBERT-base.
    Mean + Std pooling over token sequence gives a rich fixed-length embedding.
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_name = config["audio"]["model_name"]          # facebook/hubert-base-ls960
        hubert_dim = config["audio"]["hubert_output_dim"]   # 768
        proj_dim   = config["audio"]["projection_dim"]      # 256
        output_dim = config["audio"]["output_dim"]          # 256

        # ── HuBERT backbone ───────────────────────────────────────
        logger.info(f"Loading HuBERT: {model_name}  (first run downloads ~360 MB)")
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hubert.gradient_checkpointing_enable() 
        self._freeze_feature_extractor()

        # ── Projection: 768×2 → 256 ───────────────────────────────
        # Mean+Std pooling doubles dim: [B,768] + [B,768] → [B,1536]
        self.projection = nn.Sequential(
            nn.Linear(hubert_dim * 2, proj_dim * 2),   # 1536 → 512
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim * 2, output_dim),        # 512  → 256
            nn.LayerNorm(output_dim),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"AudioBranch  —  {n_params:,} trainable parameters")

    # ─────────────────────────────────────────────────────────────

    def _freeze_feature_extractor(self):
        """
        Freeze HuBERT's CNN feature extractor + feature projection.
        Transformer layers remain trainable for SER fine-tuning.
        """
        n_frozen = 0
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        logger.info(f"AudioBranch  —  froze {n_frozen:,} HuBERT CNN parameters")

    # ─────────────────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio : Tensor [B, max_samples]  float32 waveform
        Returns:
            audio_emb : Tensor [B, output_dim]
        """
        # ── HuBERT forward ────────────────────────────────────────
        hidden = self.hubert(
            input_values = audio,
            return_dict  = True,
        ).last_hidden_state                                  # [B, seq_len, 768]

        # ── Mean + Std pooling ────────────────────────────────────
        mean   = hidden.mean(dim=1)                          # [B, 768]
        std    = hidden.std(dim=1).clamp(min=1e-6)           # [B, 768]
        pooled = torch.cat([mean, std], dim=-1)              # [B, 1536]

        # ── Project to output_dim ─────────────────────────────────
        audio_emb = self.projection(pooled)                  # [B, 256]

        return audio_emb