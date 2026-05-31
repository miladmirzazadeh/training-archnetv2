"""ArchNetv2 — Xu, Jha, Mehadi, Mandal (Autom. Constr. 165 (2024) 105486).

Builds the model on top of Ultralytics YOLOv8 by:
  1) loading a custom YAML (cfg/archnetv2.yaml) that defines the 4-head
     ArchNetv2 topology with C2f modules as placeholders, and
  2) post-process replacing the four placeholder C2f blocks at indices
     [18, 21, 24, 27] with AC_CBAM (Aggregated C2f + CBAM) modules.

The 4-head Detect, channel widths, and stride auto-detection are all
handled by Ultralytics.
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, CBAM


# Indices in the YAML where C2f should become AC_CBAM (one per detection head).
AC_CBAM_INDICES = (18, 21, 24, 27)

CFG_PATH = Path(__file__).resolve().parent / "cfg" / "archnetv2.yaml"


class AC_CBAM(nn.Module):
    """Aggregated C2f + CBAM block (Fig. 7 of the paper).

    Structure: C2f(n=3) -> CBAM -> C2f(n=1).
    The first C2f extracts features, CBAM re-weights spatial+channel
    importance, the second C2f re-tunes the gated features.
    """

    def __init__(self, c1: int, c2: int, n1: int = 3, n2: int = 1,
                 shortcut: bool = True, cbam_kernel: int = 7):
        super().__init__()
        self.c2f1 = C2f(c1, c2, n=n1, shortcut=shortcut)
        self.cbam = CBAM(c2, kernel_size=cbam_kernel)
        self.c2f2 = C2f(c2, c2, n=n2, shortcut=shortcut)

    def forward(self, x):
        return self.c2f2(self.cbam(self.c2f1(x)))


def _swap_c2f_for_ac_cbam(model: nn.Module, indices=AC_CBAM_INDICES) -> nn.Module:
    """Replace `model.model[i]` (a C2f) with an AC_CBAM of matching channels."""
    seq = model.model  # nn.Sequential of layers
    for idx in indices:
        old = seq[idx]
        if not isinstance(old, C2f):
            raise RuntimeError(
                f"Expected C2f at index {idx}, got {type(old).__name__}. "
                "The YAML structure changed; update AC_CBAM_INDICES."
            )
        c1 = old.cv1.conv.in_channels   # C2f.cv1 is a 1x1 Conv
        c2 = old.cv2.conv.out_channels  # C2f.cv2 produces the C2f output
        new = AC_CBAM(c1=c1, c2=c2, n1=3, n2=1, shortcut=True).to(
            next(old.parameters()).device
        )
        # Preserve Ultralytics' bookkeeping attributes (i, f, type, np).
        new.i, new.f = old.i, old.f
        new.type = f"{__name__}.AC_CBAM"
        new.np = sum(p.numel() for p in new.parameters())
        seq[idx] = new
    return model


def build_archnetv2(scale: str = "l", nc: int = 2, weights: str | None = None,
                    verbose: bool = True) -> YOLO:
    """Construct the ArchNetv2 model.

    Args:
        scale: only 'l' is defined in cfg/archnetv2.yaml (paper uses YOLOv8l).
        nc: number of detection classes (paper: 13).
        weights: optional .pt path to load weights from after build.
        verbose: pass through to YOLO YAML parser.
    """
    cfg = str(CFG_PATH)
    # Ultralytics picks the scale from the cfg filename suffix (e.g. 'n','s','l').
    # We embed it explicitly:
    yolo = YOLO(cfg, task="detect", verbose=verbose)
    # Override nc on the underlying nn.Module if user passed a different value
    if nc != yolo.model.yaml.get("nc"):
        yolo.model.yaml["nc"] = nc
        yolo.model.nc = nc

    _swap_c2f_for_ac_cbam(yolo.model)

    # Re-run stride detection: walk a dummy input through the model so the
    # Detect head re-anchors its strides to the new modules.
    yolo.model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 640, 640)
        _ = yolo.model(dummy)

    if weights:
        yolo = YOLO(weights)  # if user has a trained .pt, just load that
    return yolo


if __name__ == "__main__":
    m = build_archnetv2(verbose=True)
    total = sum(p.numel() for p in m.model.parameters())
    print(f"\nArchNetv2 built. Total parameters: {total/1e6:.2f}M")
    print(f"Detect strides: {m.model.stride.tolist()}")
