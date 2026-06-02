"""Convert the synthetic dataset (configs + rich_json) into per-class
segmentation masks aligned to the rendered PNGs.

Classes: 0=background, 1=wall, 2=door, 3=window  (4=room if --rooms).

Geometry sources (all per plan id `plan_XXXXX`):
  configs/plan_XXXXX.json   walls.polygon / rooms.polygon  (mm)
  rich_json/plan_XXXXX_rich.json  openings with BOTH _mm and _px points

The mm->px transform is solved per plan by least-squares from the
same-point pairs the rich_json already stores (hinge/leaf/p1/p2 mm<->px).
This reproduces the renderer's ModelTransform (rotation+scale+offset+flip)
to sub-pixel accuracy and is self-validating (we report the residual).

Usage:
  python -m floorseg.synth_to_masks \\
      --configs <dir>/configs --rich <dir>/rich_json --images <dir>/images/train \\
      --out <dir>/masks/train [--rooms] [--line-frac 0.25]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BG, WALL, DOOR, WINDOW, ROOM = 0, 1, 2, 3, 4
PLAN_RE = re.compile(r"(plan_\d+)")


def plan_id(path: str):
    m = PLAN_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def collect_pairs(rich: dict):
    """Same-point (mm, px) correspondences from openings."""
    mm, px = [], []
    for o in rich.get("openings", []):
        for a, b in (("hinge_point_mm", "hinge_point_px"),
                     ("leaf_end_mm", "leaf_end_px"),
                     ("p1_mm", "p1_px"), ("p2_mm", "p2_px")):
            if o.get(a) and o.get(b):
                mm.append(o[a]); px.append(o[b])
    return np.array(mm, float), np.array(px, float)


def solve_affine(mm: np.ndarray, px: np.ndarray):
    """Solve 2x3 affine A such that px ~= A @ [mm_x, mm_y, 1]. Returns (A, resid)."""
    if len(mm) < 3:
        return None, None
    M = np.hstack([mm, np.ones((len(mm), 1))])            # [N,3]
    sol, *_ = np.linalg.lstsq(M, px, rcond=None)          # [3,2]
    A = sol.T                                             # [2,3]
    pred = (M @ sol)
    resid = float(np.sqrt(((pred - px) ** 2).sum(1)).mean())
    return A, resid


def to_px(A, pts):
    pts = np.asarray(pts, float)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (h @ A.T)


def convert_one(cfg: dict, rich: dict, W: int, H: int, line_frac: float,
                rooms: bool):
    A, resid = solve_affine(*collect_pairs(rich))
    if A is None:
        return None, None
    mask = Image.new("L", (W, H), BG)
    d = ImageDraw.Draw(mask)

    if rooms:  # lowest layer
        for r in cfg.get("rooms", []):
            poly = r.get("polygon") or r.get("points")
            if poly and len(poly) >= 3:
                d.polygon([tuple(p) for p in to_px(A, poly)], fill=ROOM)

    for w in cfg.get("walls", []):                        # walls over rooms
        poly = w.get("polygon")
        if poly and len(poly) >= 3:
            d.polygon([tuple(p) for p in to_px(A, poly)], fill=WALL)

    # openings over walls — draw the symbol stroke (leaf / glazing) thick,
    # not the loose bbox (a door bbox includes empty swing area).
    for o in rich.get("openings", []):
        cls = DOOR if o.get("type") == "door" else WINDOW
        if cls == DOOR and o.get("hinge_point_px") and o.get("leaf_end_px"):
            a, b = o["hinge_point_px"], o["leaf_end_px"]
        elif cls == WINDOW and o.get("p1_px") and o.get("p2_px"):
            a, b = o["p1_px"], o["p2_px"]
        else:
            bb = o.get("bbox_px")
            if not bb:
                continue
            a, b = (bb[0], bb[1]), (bb[2], bb[3])
        length = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        width = max(4, int(line_frac * length))
        d.line([tuple(a), tuple(b)], fill=cls, width=width)
    return mask, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, type=Path)
    ap.add_argument("--rich", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path,
                    help="Rendered PNGs (to read exact image size).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rooms", action="store_true")
    ap.add_argument("--line-frac", type=float, default=0.25,
                    help="Opening stroke width as a fraction of its length.")
    ap.add_argument("--max-resid", type=float, default=3.0,
                    help="Skip a plan if the mm->px fit residual (px) exceeds this.")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    cfgs = {plan_id(p): p for p in glob.glob(str(a.configs / "plan_*.json"))}
    riches = {plan_id(p): p for p in glob.glob(str(a.rich / "*_rich.json"))}
    ids = sorted(set(cfgs) & set(riches))
    if a.limit:
        ids = ids[: a.limit]
    print(f"{len(ids)} plans with both config + rich_json.")

    done = skipped = 0
    resids = []
    for pid in ids:
        hits = list(Path(a.images).rglob(f"{pid}.png"))  # handles train/val subdirs
        if not hits:
            skipped += 1; continue
        with Image.open(hits[0]) as im:
            W, H = im.size
        cfg = json.loads(Path(cfgs[pid]).read_text())
        rich = json.loads(Path(riches[pid]).read_text())
        mask, resid = convert_one(cfg, rich, W, H, a.line_frac, a.rooms)
        if mask is None or (resid is not None and resid > a.max_resid):
            skipped += 1; continue
        mask.save(a.out / f"{pid}.png")
        resids.append(resid); done += 1
    print(f"wrote {done} masks, skipped {skipped}. "
          f"mean mm->px residual: {np.mean(resids):.3f} px" if resids else "no masks")
    if done == 0:
        raise SystemExit("ERROR: 0 masks written — check paths / rich_json pairs.")


if __name__ == "__main__":
    main()
