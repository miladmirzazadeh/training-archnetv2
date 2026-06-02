"""Bridge to vtruvian: label each extracted vector line by the segmentation mask.

Takes vtruvian's `_lines.json` (line segments in pixel coords) + a predicted
class mask (from floorseg.infer) and tags every line wall/door/window by
sampling the mask along the segment. Output = labeled vectors that feed
vtruvian's existing DXF assembly (walls→centerlines, openings→blocks).

Usage:
  python -m floorseg.label_lines --lines plan_lines.json --mask plan_mask.png \\
      --out plan_lines_labeled.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {0: "background", 1: "wall", 2: "door", 3: "window", 4: "room"}


def sample_line(mask: np.ndarray, p0, p1, n=25):
    """Majority non-background class along the segment (with a small normal band)."""
    H, W = mask.shape
    xs = np.linspace(p0[0], p1[0], n); ys = np.linspace(p0[1], p1[1], n)
    votes = Counter()
    for x, y in zip(xs, ys):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xi, yi = int(round(x + dx)), int(round(y + dy))
                if 0 <= xi < W and 0 <= yi < H:
                    votes[int(mask[yi, xi])] += 1
    votes.pop(0, None)  # drop background
    return votes.most_common(1)[0][0] if votes else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True, type=Path, help="vtruvian _lines.json")
    ap.add_argument("--mask", required=True, type=Path, help="class-index mask PNG")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start-key", default="start")
    ap.add_argument("--end-key", default="end")
    a = ap.parse_args()

    mask = np.asarray(Image.open(a.mask))
    data = json.loads(a.lines.read_text())
    lines = data["lines"] if isinstance(data, dict) and "lines" in data else data

    # If the mask was produced at a different resolution than the lines'
    # coordinate frame, scale line coords to mask pixels.
    mh, mw = mask.shape
    iw = (data.get("image_width_px") if isinstance(data, dict) else None) or mw
    ih = (data.get("image_height_px") if isinstance(data, dict) else None) or mh
    sx, sy = mw / iw, mh / ih

    out = []
    for ln in lines:
        p0 = (ln[a.start_key][0] * sx, ln[a.start_key][1] * sy)
        p1 = (ln[a.end_key][0] * sx, ln[a.end_key][1] * sy)
        cls = sample_line(mask, p0, p1)
        rec = dict(ln); rec["label"] = CLASS_NAMES.get(cls, str(cls)); rec["class_id"] = cls
        out.append(rec)

    a.out.write_text(json.dumps(
        {"image_width_px": iw, "image_height_px": ih, "lines": out}, indent=1))
    c = Counter(r["label"] for r in out)
    print("labeled lines:", dict(c))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
