"""Apply the paper's 8x augmentation (4 rotations x horizontal flip) to
a YOLO-format dataset in-place. Run once before training; do NOT also
enable runtime rotation augmentation in the trainer.

Usage:
    python -m archnetv2.scripts.expand_dataset --root /path/to/dataset_yolo \\
                                               --splits train val
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def rotate_box(cx, cy, w, h, angle_deg):
    """Rotate a YOLO-normalized box around the image center."""
    if angle_deg == 0:
        return cx, cy, w, h
    if angle_deg == 90:
        return 1 - cy, cx, h, w
    if angle_deg == 180:
        return 1 - cx, 1 - cy, w, h
    if angle_deg == 270:
        return cy, 1 - cx, h, w
    raise ValueError(angle_deg)


def flip_h(cx, cy, w, h):
    return 1 - cx, cy, w, h


def read_labels(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append((int(parts[0]), *(float(x) for x in parts[1:])))
    return out


def write_labels(p: Path, labels):
    p.write_text(
        "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                  for (c, cx, cy, w, h) in labels) + "\n"
    )


def expand_split(root: Path, split: str) -> int:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    originals = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                       and "_aug" not in p.stem)
    written = 0
    for img_path in originals:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels = read_labels(lbl_path)
        try:
            img = Image.open(img_path)
        except OSError:
            continue
        for rot in (0, 90, 180, 270):
            rotated = img.rotate(-rot, expand=True) if rot else img
            for flipped in (False, True):
                if rot == 0 and not flipped:
                    continue  # original already on disk
                final_img = rotated.transpose(Image.FLIP_LEFT_RIGHT) if flipped else rotated
                tag = f"_aug_r{rot}{'_f' if flipped else ''}"
                out_img = img_dir / f"{img_path.stem}{tag}{img_path.suffix}"
                out_lbl = lbl_dir / f"{img_path.stem}{tag}.txt"
                final_img.save(out_img)
                new_labels = []
                for (c, cx, cy, w, h) in labels:
                    cx2, cy2, w2, h2 = rotate_box(cx, cy, w, h, rot)
                    if flipped:
                        cx2, cy2, w2, h2 = flip_h(cx2, cy2, w2, h2)
                    new_labels.append((c, cx2, cy2, w2, h2))
                write_labels(out_lbl, new_labels)
                written += 1
        img.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    for split in args.splits:
        n = expand_split(args.root.resolve(), split)
        print(f"  {split}: wrote {n} augmented images")


if __name__ == "__main__":
    main()
