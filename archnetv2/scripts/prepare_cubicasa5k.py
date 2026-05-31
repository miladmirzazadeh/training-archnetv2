"""Convert CubiCasa5K (https://github.com/CubiCasa/CubiCasa5k) into
YOLO-format detection data for ArchNetv2-openings (door / window).

CubiCasa5K layout per plan (relative to dataset root):
    high_quality/<id>/F1_scaled.png   # rendered floor plan
    high_quality/<id>/model.svg       # SVG annotations (geometry + class)
    train.txt / val.txt / test.txt    # plan IDs per split

The SVG uses `<g class="Door FixedDoor"/>`, `<g class="Window"/>`, etc.
We extract the bounding box of each Door/Window polygon and write
YOLO-format labels (normalized cx cy w h).

Usage:
    python -m archnetv2.scripts.prepare_cubicasa5k \\
        --src /path/to/cubicasa5k \\
        --dst /path/to/cubicasa5k_yolo \\
        --quality high  # or "colorful" or "all"

Output layout (Ultralytics-compatible):
    dst/images/{train,val,test}/<id>.png
    dst/labels/{train,val,test}/<id>.txt
    dst/data.yaml
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

CLASS_MAP = {"door": 0, "window": 1}  # YOLO class ids
QUALITY_DIRS = {
    "high": ("high_quality",),
    "colorful": ("colorful",),
    "high_arch": ("high_quality_architectural",),
    "all": ("high_quality", "colorful", "high_quality_architectural"),
}

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def class_of(elem) -> str | None:
    """Return 'door' or 'window' if the SVG group represents one; else None.

    CubiCasa5K uses space-separated multi-class strings such as
    "Door FixedDoor", "Window", "Railing Wall" etc.
    """
    cls = (elem.get("class") or "").lower()
    if not cls:
        return None
    tokens = cls.split()
    # Skip walls/columns/railings even if they happen to contain "door"/"window"
    # substrings (defensive: CubiCasa class strings rarely do, but harmless).
    if "door" in tokens:
        return "door"
    if "window" in tokens:
        return "window"
    return None


def polygon_points(elem) -> list[tuple[float, float]]:
    """Pull (x, y) vertices from <polygon points="..."> or <path d="...">."""
    pts: list[tuple[float, float]] = []
    for poly in elem.iter("{http://www.w3.org/2000/svg}polygon"):
        nums = [float(n) for n in NUM.findall(poly.get("points", ""))]
        pts.extend(zip(nums[0::2], nums[1::2]))
    for path in elem.iter("{http://www.w3.org/2000/svg}path"):
        # Path coords are messy (cmds + nums). For an axis-aligned bbox we just
        # need *any* x/y on the path, so taking every number pair gives a
        # superset bbox — fine for door/window which are convex rectangles.
        nums = [float(n) for n in NUM.findall(path.get("d", ""))]
        pts.extend(zip(nums[0::2], nums[1::2]))
    return pts


def parse_transform(tf: str | None) -> tuple[float, float]:
    """Return (tx, ty) from an SVG transform="translate(tx ty) ...".

    CubiCasa SVGs use translate() to position element groups. We ignore
    rotate/scale because door/window groups in this dataset are
    axis-aligned post-translate.
    """
    if not tf:
        return 0.0, 0.0
    m = re.search(r"translate\(\s*(-?\d+\.?\d*)[ ,]+(-?\d+\.?\d*)", tf)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def boxes_from_svg(svg_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(cls_id, xmin, ymin, xmax, ymax), ...] in SVG pixel space."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    boxes: list[tuple[int, float, float, float, float]] = []
    # Walk all <g> descendants; CubiCasa nests Door inside Wall inside FloorPlan.
    for g in root.iter("{http://www.w3.org/2000/svg}g"):
        kind = class_of(g)
        if kind is None:
            continue
        pts = polygon_points(g)
        if not pts:
            continue
        tx, ty = parse_transform(g.get("transform"))
        xs = [p[0] + tx for p in pts]
        ys = [p[1] + ty for p in pts]
        if not xs:
            continue
        boxes.append((CLASS_MAP[kind], min(xs), min(ys), max(xs), max(ys)))
    return boxes


def to_yolo(box, w: int, h: int) -> str:
    cid, x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2) / w
    cy = ((y0 + y1) / 2) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    # Clamp into [0, 1] — pathological SVGs sometimes go slightly out of bounds.
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
    if bw <= 0 or bh <= 0:
        return ""
    return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def split_ids(src: Path, split: str) -> list[str]:
    """CubiCasa5K ships train.txt / val.txt / test.txt at the repo root."""
    txt = src / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split file missing: {txt}")
    ids: list[str] = []
    for line in txt.read_text().splitlines():
        line = line.strip().lstrip("/")
        if line:
            ids.append(line)
    return ids


def convert(src: Path, dst: Path, quality: str, limit: int | None,
            link_images: bool = False) -> None:
    quality_dirs = QUALITY_DIRS[quality]
    splits = ("train", "val", "test")
    for split in splits:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"plans": 0, "boxes": 0, "skipped": 0} for s in splits}

    for split in splits:
        ids = split_ids(src, split)
        if limit:
            ids = ids[:limit]
        for plan_id in ids:
            qd = next((q for q in quality_dirs if (src / q / plan_id).exists()), None)
            if qd is None:
                stats[split]["skipped"] += 1
                continue
            plan_dir = src / qd / plan_id
            png = plan_dir / "F1_scaled.png"
            svg = plan_dir / "model.svg"
            if not (png.exists() and svg.exists()):
                stats[split]["skipped"] += 1
                continue
            try:
                with Image.open(png) as im:
                    w, h = im.size
                boxes = boxes_from_svg(svg)
            except (ET.ParseError, OSError) as e:
                print(f"[skip] {plan_id}: {e}", file=sys.stderr)
                stats[split]["skipped"] += 1
                continue

            # Use a flat filename: replace '/' so the basename is unique.
            base = plan_id.replace("/", "_").rstrip("_")
            dst_img = dst / "images" / split / f"{base}.png"
            if link_images:
                # Symlink instead of copy — keeps the output tiny so it fits
                # within Kaggle's 20 GB /kaggle/working limit (the 9 GB source
                # PNGs stay where they are in /kaggle/input).
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(png.resolve())
            else:
                shutil.copy2(png, dst_img)
            label_lines = [
                line for line in (to_yolo(b, w, h) for b in boxes) if line
            ]
            (dst / "labels" / split / f"{base}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else "")
            )
            stats[split]["plans"] += 1
            stats[split]["boxes"] += len(label_lines)

    (dst / "data.yaml").write_text(
        "# Ultralytics dataset config — CubiCasa5K openings (door + window)\n"
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names: [door, window]\n"
    )

    print("\nDone.")
    for s in splits:
        st = stats[s]
        print(f"  {s:5s}: {st['plans']:5d} plans, {st['boxes']:6d} boxes,"
              f" {st['skipped']:4d} skipped")
    print(f"\nDataset config: {dst/'data.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Path to extracted CubiCasa5K dataset root.")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Output dir for the YOLO-format dataset.")
    ap.add_argument("--quality", choices=list(QUALITY_DIRS), default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on plans per split (for quick tests).")
    ap.add_argument("--link-images", action="store_true",
                    help="Symlink source PNGs instead of copying (saves disk; "
                         "recommended on Kaggle).")
    args = ap.parse_args()
    convert(args.src.resolve(), args.dst.resolve(), args.quality, args.limit,
            link_images=args.link_images)


if __name__ == "__main__":
    main()
