"""PrimitiveNet — a lightweight set-transformer over CAD primitives.

Each primitive (line/arc/circle) is a token with a geometric feature
vector; self-attention over all primitives in a plan lets the model use
context ("this short arc near a wall gap is a door"). Outputs a class per
primitive. Far lighter than DPSS (no image branch, no Mask2Former
decoder) — designed to train on one GPU in well under 3 hours, while
producing the same *kind* of output: per-primitive semantic labels that
group into instances downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PrimitiveNet(nn.Module):
    def __init__(self, num_classes: int = 5, feat_dim: int = 17, dim: int = 256,
                 depth: int = 4, heads: int = 8, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, feats, key_padding_mask=None):
        # feats: [B,N,FEAT_DIM]; key_padding_mask: [B,N] True where padded
        x = self.embed(feats)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(x)  # [B,N,num_classes]
