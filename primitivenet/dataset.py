"""Dataset of parsed CAD plans -> per-primitive feature tokens + labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = -100  # CE ignore index for padded tokens


class PrimitiveDataset(Dataset):
    def __init__(self, root: Path, split: str, max_prims: int = 4096):
        self.files = sorted((Path(root) / split).glob("*.json"))
        self.max_prims = max_prims
        if not self.files:
            raise FileNotFoundError(f"No parsed plans under {root}/{split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = json.loads(self.files[i].read_text())
        prims = d["primitives"]
        if len(prims) > self.max_prims:
            idx = np.random.choice(len(prims), self.max_prims, replace=False)
            prims = [prims[j] for j in idx]
        feat = torch.tensor([p["feat"] for p in prims], dtype=torch.float32)
        sem = torch.tensor([p["sem"] for p in prims], dtype=torch.long)
        return feat, sem


def collate(batch):
    """Pad a batch of variable-length primitive sets. Returns
    feats [B,N,F], labels [B,N] (pad=IGNORE), key_padding_mask [B,N] (True=pad)."""
    B = len(batch)
    N = max(f.shape[0] for f, _ in batch)
    F = batch[0][0].shape[1]                 # feature dim inferred from data (12 SVG / 17 DXF)
    feats = torch.zeros(B, N, F)
    labels = torch.full((B, N), IGNORE, dtype=torch.long)
    mask = torch.ones(B, N, dtype=torch.bool)  # True where padded
    for b, (f, s) in enumerate(batch):
        n = f.shape[0]
        feats[b, :n] = f
        labels[b, :n] = s
        mask[b, :n] = False
    return feats, labels, mask
