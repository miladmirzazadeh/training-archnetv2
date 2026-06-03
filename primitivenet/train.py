"""Train PrimitiveNet (per-primitive classifier) on parsed FloorPlanCAD.

Single-GPU, fast (<3h). Per-primitive cross-entropy with optional
down-weighting of the dominant background class. Reports val accuracy +
macro mIoU over classes. Time-budget stop + checkpoint/resume.

Usage:
  python -m primitivenet.train --data data/fpc_json \\
      --num-classes 36 --epochs 60 --batch 8 --time-budget 2.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from primitivenet.dataset import PrimitiveDataset, collate, IGNORE
from primitivenet.model import PrimitiveNet


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    correct = total = sub_correct = sub_total = 0
    for feats, sem, sub, mask in loader:
        feats, sem, sub, mask = feats.to(device), sem.to(device), sub.to(device), mask.to(device)
        lc, ls = model(feats, key_padding_mask=mask)
        pred, spred = lc.argmax(-1), ls.argmax(-1)
        valid = sem != IGNORE
        p = pred[valid].cpu().numpy(); g = sem[valid].cpu().numpy()
        correct += (p == g).sum(); total += g.size
        for c in range(num_classes):
            pc, gc = p == c, g == c
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
        sv = (sub != IGNORE) & (sub > 0)           # subtype acc only where a real subtype exists
        sub_correct += (spred[sv] == sub[sv]).sum().item(); sub_total += int(sv.sum().item())
    iou = inter / np.maximum(union, 1); present = union > 0
    return (correct / max(total, 1)), float(iou[present].mean() if present.any() else 0.0), \
           (sub_correct / max(sub_total, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--num-classes", type=int, default=7)   # clutter/wall/door/window/column/hatch/duplicate
    ap.add_argument("--num-subtypes", type=int, default=18) # parse_dxf.SUBTYPE_NAMES
    ap.add_argument("--sub-weight", type=float, default=0.5, help="loss weight for the subtype head")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-prims", type=int, default=4096)
    ap.add_argument("--bg-weight", type=float, default=0.1,
                    help="Loss weight for the background/clutter class (id 0). "
                         "Down-weights the dominant bg.")
    ap.add_argument("--open-weight", type=float, default=2.0,
                    help="Loss weight for door(2)+window(3) — the small, hard "
                         "classes. >1 boosts opening precision/recall.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=2.5, help="Max hours.")
    ap.add_argument("--out", type=Path, default=Path("runs/primitivenet"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt = args.out / "last.pt", args.out / "best.pt"

    tr = DataLoader(PrimitiveDataset(args.data, "train", args.max_prims),
                    batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, pin_memory=True)
    va_split = "val" if (args.data / "val").exists() else "test"
    va = DataLoader(PrimitiveDataset(args.data, va_split, args.max_prims),
                    batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True)

    feat_dim = int(tr.dataset[0][0].shape[1])               # infer from data (DXF schema = 18)
    print("feat_dim:", feat_dim)
    model = PrimitiveNet(num_classes=args.num_classes, num_subtypes=args.num_subtypes,
                         feat_dim=feat_dim, dim=args.dim, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    w = torch.ones(args.num_classes, device=device); w[0] = args.bg_weight
    for c in (2, 3):                              # door, window — boost the hard classes
        if c < args.num_classes: w[c] = args.open_weight
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    crit_sub = nn.CrossEntropyLoss(ignore_index=IGNORE)

    start, best = 0, 0.0
    if last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["epoch"] + 1; best = ck["best"]
        print(f"[resume] epoch {start}, best mIoU {best:.4f}")
    print(f"train={len(tr.dataset)} val({va_split})={len(va.dataset)} device={device}")

    t0 = time.time()
    for epoch in range(start, args.epochs):
        model.train(); run = 0.0
        for feats, sem, sub, mask in tr:
            feats, sem, sub, mask = feats.to(device), sem.to(device), sub.to(device), mask.to(device)
            opt.zero_grad()
            lc, ls = model(feats, key_padding_mask=mask)
            loss = crit(lc.reshape(-1, args.num_classes), sem.reshape(-1)) \
                + args.sub_weight * crit_sub(ls.reshape(-1, args.num_subtypes), sub.reshape(-1))
            loss.backward(); opt.step(); run += loss.item()
        sched.step()
        acc, miou, sub_acc = evaluate(model, va, device, args.num_classes)
        print(f"epoch {epoch+1}/{args.epochs} loss {run/len(tr):.4f} "
              f"val_acc {acc:.4f} val_mIoU {miou:.4f} sub_acc {sub_acc:.4f} ({(time.time()-t0)/60:.1f}m)")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "best": best,
              "args": {**vars(args), "feat_dim": feat_dim}}
        torch.save(ck, last_pt)
        if miou > best:
            best = miou; ck["best"] = best; torch.save(ck, best_pt)
            print(f"  new best val_mIoU {best:.4f}")
        if (time.time() - t0) / 3600 >= args.time_budget:
            print("[time] budget reached; stopping."); break
    print(f"[done] best val_mIoU {best:.4f} -> {best_pt}")


if __name__ == "__main__":
    main()
