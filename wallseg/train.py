"""Wall semantic-segmentation training (UNet) for FloorPlanCAD walls.

A binary segmentation model (wall vs. not-wall) trained on the (image,
mask) pairs produced by `wallseg.prepare_floorplancad_walls convert`.

Built for Kaggle background runs: a wall-clock `--time-budget` stops the
session gracefully (<12 h), and the whole run dir is checkpointed to a
Kaggle Dataset so the next session resumes. Mirrors the openings
detector's resume strategy.

Usage:
    python -m wallseg.train \\
        --data /kaggle/working/walls_seg \\
        --ckpt-dataset <user>/wallseg-ckpt \\
        --time-budget 11.0 --epochs 100 --batch 8
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class WallDataset(Dataset):
    """(image, mask) pairs. Images are RGB PNGs; masks are 0/255 L PNGs."""

    def __init__(self, root: Path, split: str, imgsz: int, augment: bool):
        self.img_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        self.items = sorted(p.stem for p in self.img_dir.glob("*.png")
                            if (self.mask_dir / p.name).exists())
        self.imgsz = imgsz
        self.augment = augment
        if not self.items:
            raise FileNotFoundError(f"No image/mask pairs under {root}/{split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        stem = self.items[i]
        img = Image.open(self.img_dir / f"{stem}.png").convert("RGB").resize(
            (self.imgsz, self.imgsz))
        mask = Image.open(self.mask_dir / f"{stem}.png").convert("L").resize(
            (self.imgsz, self.imgsz), Image.NEAREST)
        x = np.asarray(img, np.float32) / 255.0
        y = (np.asarray(mask, np.float32) > 127).astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:  # horizontal flip
                x, y = x[:, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:  # vertical flip (plans have no up/down)
                x, y = x[::-1], y[::-1]
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y))[None]
        return x, y


# --------------------------------------------------------------------------- #
# Loss / metrics
# --------------------------------------------------------------------------- #
def dice_bce_loss(logits, target, eps=1e-6):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    union = p.sum((2, 3)) + target.sum((2, 3))
    dice = 1 - ((2 * inter + eps) / (union + eps)).mean()
    return bce + dice


@torch.no_grad()
def iou_score(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((2, 3))
    union = ((p + target) > 0).float().sum((2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


# --------------------------------------------------------------------------- #
# Checkpoint dataset restore / upload (same pattern as the openings trainer)
# --------------------------------------------------------------------------- #
def restore_checkpoint(ckpt_dataset: str | None, run_dir: Path) -> None:
    if not ckpt_dataset:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    name = ckpt_dataset.split("/")[-1]
    mounted = Path("/kaggle/input") / name
    zips = list(mounted.glob("*.zip")) if mounted.exists() else []
    if zips:
        print(f"[ckpt] unpacking mounted {zips[0]}")
        shutil.unpack_archive(str(zips[0]), str(run_dir))
        return
    print(f"[ckpt] trying API download of {ckpt_dataset}")
    subprocess.call(["kaggle", "datasets", "download", "-d", ckpt_dataset,
                     "-p", str(run_dir), "--unzip"])


def upload_checkpoint(ckpt_dataset: str | None, run_dir: Path) -> None:
    if not ckpt_dataset:
        print("[ckpt] no --ckpt-dataset; run dir at", run_dir)
        return
    staging = Path("/kaggle/working/_wallseg_ckpt")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.make_archive(str(staging / "run"), "zip", run_dir.parent, run_dir.name)
    (staging / "dataset-metadata.json").write_text(
        f'{{"title": "{ckpt_dataset.split("/")[-1]}", "id": "{ckpt_dataset}", '
        '"licenses": [{"name": "CC0-1.0"}]}\n')
    rc = subprocess.call(["kaggle", "datasets", "version", "-p", str(staging),
                          "-m", "wallseg checkpoint", "--dir-mode", "zip"])
    if rc != 0:
        subprocess.call(["kaggle", "datasets", "create", "-p", str(staging),
                         "--dir-mode", "zip"])


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def build_model(encoder: str):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-budget", type=float, default=11.0)
    ap.add_argument("--project", default="/kaggle/working/runs/wallseg")
    ap.add_argument("--name", default="wallseg_unet")
    ap.add_argument("--ckpt-dataset", default=None)
    args = ap.parse_args()

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt = run_dir / "last.pt", run_dir / "best.pt"
    restore_checkpoint(args.ckpt_dataset, run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch, best_iou = 0, 0.0
    if last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_iou = ck["epoch"] + 1, ck.get("best_iou", 0.0)
        print(f"[resume] from epoch {start_epoch}, best_iou={best_iou:.4f}")

    tr = DataLoader(WallDataset(args.data, "train", args.imgsz, True),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True)
    va = DataLoader(WallDataset(args.data, "val", args.imgsz, False),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True)
    print(f"train={len(tr.dataset)}  val={len(va.dataset)}  device={device}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                ious.append(iou_score(model(x), y))
        val_iou = float(np.mean(ious)) if ious else 0.0
        print(f"epoch {epoch+1}/{args.epochs}  loss={run_loss/len(tr):.4f}  "
              f"val_IoU={val_iou:.4f}  ({(time.time()-t0)/60:.1f} min)")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "best_iou": best_iou,
              "args": vars(args)}
        torch.save(ck, last_pt)
        if val_iou > best_iou:
            best_iou = val_iou
            ck["best_iou"] = best_iou
            torch.save(ck, best_pt)
            print(f"  new best val_IoU={best_iou:.4f} -> {best_pt}")

        if (time.time() - t0) / 3600.0 >= args.time_budget:
            print(f"[time] budget {args.time_budget} h reached; stopping.")
            break

    upload_checkpoint(args.ckpt_dataset, run_dir)
    print(f"[done] best val_IoU={best_iou:.4f}; weights in {run_dir}")


if __name__ == "__main__":
    main()
