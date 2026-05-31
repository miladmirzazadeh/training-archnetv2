"""Train ArchNetv2-openings on a YOLO-format dataset.

Paper hyperparameters (Section 3.3):
    input size       640 x 640 x 3
    batch size       4
    epochs           500
    early stopping   patience = 100 epochs
    optimizer        SGD (Ultralytics default for YOLOv8)
    augmentation     4 rotations (0, 90, 180, 270) x horizontal flip = 8x

Ultralytics' built-in augmentations cover horizontal flip natively
(fliplr=0.5). Multiples of 90 deg rotation are not in YOLOv8's default
pipeline; if you need exact paper-style 8x augmentation, run
`scripts/expand_dataset.py` once before training instead of relying on
runtime augmentation.

Usage:
    python -m archnetv2.train --data /path/to/cubicasa5k_yolo/data.yaml \\
                              --epochs 500 --batch 4 --device 0 \\
                              --name archnetv2_openings_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from archnetv2.model import build_archnetv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="Path to data.yaml (Ultralytics format).")
    p.add_argument("--epochs", type=int, default=500, help="Paper: 500.")
    p.add_argument("--batch", type=int, default=4, help="Paper: 4 on a T4.")
    p.add_argument("--imgsz", type=int, default=640, help="Paper: 640.")
    p.add_argument("--patience", type=int, default=100,
                   help="Early-stop patience in epochs. Paper: 100.")
    p.add_argument("--device", default="", help="cuda device, e.g. '0' or 'cpu'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/archnetv2",
                   help="Ultralytics runs directory.")
    p.add_argument("--name", default="train",
                   help="Sub-folder under --project for this run.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last.pt in the project/name folder.")
    p.add_argument("--weights", default=None,
                   help="Optional .pt to initialise from (e.g. yolov8l.pt for "
                        "partial backbone reuse).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip backbone-only weight loading from yolov8l.pt.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = build_archnetv2(nc=2, verbose=True)

    # Try to seed weights from yolov8l.pt for the backbone. This is the same
    # approach the paper uses (build on top of YOLOv8). Ultralytics' .load()
    # silently skips layers whose shapes don't match — perfect for our case
    # because our head differs but the backbone does not.
    if args.weights:
        print(f"[init] Loading weights from {args.weights}")
        model.load(args.weights)
    elif not args.no_pretrained:
        try:
            print("[init] Warm-starting backbone from yolov8l.pt (mismatched "
                  "head layers will be skipped).")
            model.load("yolov8l.pt")
        except Exception as e:
            print(f"[init] Could not load yolov8l.pt ({e}); training from scratch.")

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation: matches paper's flip; mosaic/HSV kept on as YOLOv8
        # defaults already produce strong, well-tested augmentation. Disable
        # mosaic in the last 10 epochs (Ultralytics default) to stabilize.
        fliplr=0.5,
        flipud=0.0,        # paper flips only horizontally
        degrees=0.0,       # 90-degree rotations should be pre-baked offline
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,  # floor plans are near-grayscale
        # Loss weights are YOLOv8 defaults; paper doesn't override them.
        plots=True,
        save=True,
    )

    # Evaluate on the test split too (if defined in data.yaml).
    print("\n[val] Running validation on test split...")
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz,
                        device=args.device or None)
    print(f"\nFinal mAP50: {metrics.box.map50:.4f}")
    print(f"Final mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
