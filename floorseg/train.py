"""Train a 4-class floor-plan segmentation UNet (bg/wall/door/window).

Input: (image PNG, class-index mask PNG) pairs from synth_to_masks.py
(+ optionally FloorPlanCAD/CubiCasa rasterized the same way).

Single GPU, resume + wall-clock time budget. Saves best.pt by val mIoU.
Download best.pt from RunPod with `runpodctl send`.

Usage:
  python -m floorseg.train --images-train data/images/train --masks-train data/masks/train \\
      --images-val data/images/val --masks-val data/masks/val \\
      --classes 4 --epochs 80 --batch 8 --imgsz 640 --time-budget 2.5
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_NAMES = ["background", "wall", "door", "window", "room"]


def _is_val(stem, val_frac):
    return (int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000) / 1000.0 < val_frac


class SegDataset(Dataset):
    """Items = mask stems (flat masks_dir); image found by rglob under images_root."""
    def __init__(self, images_root, masks_dir, imgsz, augment, val_frac, split):
        self.images_root, self.masks_dir = Path(images_root), Path(masks_dir)
        stems = sorted(p.stem for p in self.masks_dir.glob("*.png"))
        self.items = [s for s in stems
                      if (_is_val(s, val_frac) == (split == "val"))]
        self._img = {}
        for s in self.items:
            hits = list(self.images_root.rglob(f"{s}.png"))
            if hits:
                self._img[s] = hits[0]
        self.items = [s for s in self.items if s in self._img]
        self.imgsz, self.augment = imgsz, augment
        if not self.items:
            raise FileNotFoundError(
                f"No {split} image/mask pairs ({images_root} / {masks_dir})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        img = Image.open(self._img[s]).convert("RGB").resize(
            (self.imgsz, self.imgsz))
        m = Image.open(self.masks_dir / f"{s}.png").resize(
            (self.imgsz, self.imgsz), Image.NEAREST)
        x = np.asarray(img, np.float32) / 255.0
        y = np.asarray(m, np.int64)
        if self.augment:
            if np.random.rand() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5: x, y = x[::-1], y[::-1]
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y))
        return x, y


def build_model(classes, encoder="resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=classes)


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval(); inter = np.zeros(classes); union = np.zeros(classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        for c in range(classes):
            pc, gc = p == c, y == c
            inter[c] += (pc & gc).sum().item(); union[c] += (pc | gc).sum().item()
    iou = inter / np.maximum(union, 1)
    present = union > 0
    return iou, float(iou[present].mean() if present.any() else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Images root (searched recursively).")
    ap.add_argument("--masks", required=True, help="Flat dir of class-index mask PNGs.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=2.5)
    ap.add_argument("--out", type=Path, default=Path("runs/floorseg"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt = a.out / "last.pt", a.out / "best.pt"

    tr = DataLoader(SegDataset(a.images, a.masks, a.imgsz, True, a.val_frac, "train"),
                    batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True)
    va = DataLoader(SegDataset(a.images, a.masks, a.imgsz, False, a.val_frac, "val"),
                    batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    pin_memory=True)

    model = build_model(a.classes, a.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # down-weight background; openings are rare → keep their weight up
    w = torch.ones(a.classes, device=device); w[0] = 0.2
    crit = nn.CrossEntropyLoss(weight=w)

    start, best = 0, 0.0
    if last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["epoch"] + 1; best = ck["best"]
        print(f"[resume] epoch {start}, best mIoU {best:.4f}")
    print(f"train={len(tr.dataset)} val={len(va.dataset)} device={device} classes={a.classes}")

    t0 = time.time()
    for epoch in range(start, a.epochs):
        model.train(); run = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        iou, miou = evaluate(model, va, device, a.classes)
        names = CLASS_NAMES[:a.classes]
        per = "  ".join(f"{n}:{iou[c]:.2f}" for c, n in enumerate(names))
        print(f"ep {epoch+1}/{a.epochs} loss {run/len(tr):.3f} mIoU {miou:.4f} "
              f"[{per}] ({(time.time()-t0)/60:.1f}m)")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "best": best,
              "args": vars(a), "class_names": names}
        torch.save(ck, last_pt)
        if miou > best:
            best = miou; ck["best"] = best; torch.save(ck, best_pt)
            print(f"  new best mIoU {best:.4f}")
        if (time.time() - t0) / 3600 >= a.time_budget:
            print("[time] budget reached; stopping."); break
    print(f"[done] best mIoU {best:.4f} -> {best_pt}")


if __name__ == "__main__":
    main()
