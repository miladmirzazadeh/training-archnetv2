"""Build a wall **segmentation** dataset from the original FloorPlanCAD
SVG release (the catwhisker `*.tar.xz` archives on Kaggle).

Two modes:

  probe    Extract a few SVGs and report the annotation schema — element
           tags, attribute names, the histogram of `semantic-id` values
           and their stroke colors, and whether each SVG has a sibling
           PNG. Run this FIRST to identify which id(s) mark walls.

  convert  For every SVG: take the drawing raster (sibling PNG if present,
           else render the SVG) as the input image, and rasterize the
           wall primitives into a binary mask aligned to that image.
           Outputs an images/ + masks/ tree with a deterministic split.

FloorPlanCAD encodes class per primitive via a `semantic-id` attribute
(1..35) and instance via `instance-id` (-1 for stuff like walls). The
exact wall id is confirmed from `probe` output and passed via --wall-ids.

Usage:
    # 1) discover the schema (paste the output back)
    python -m wallseg.prepare_floorplancad_walls probe \\
        --src /kaggle/input/datasets/catwhisker/floorplancad-dataset \\
        --work /kaggle/working/fpcad

    # 2) convert once the wall id is known
    python -m wallseg.prepare_floorplancad_walls convert \\
        --src /kaggle/input/datasets/catwhisker/floorplancad-dataset \\
        --work /kaggle/working/fpcad --dst /kaggle/working/walls_seg \\
        --wall-ids 1,2 --imgsz 640
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import re
import sys
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

# Numeric tokens (handles ints, floats, scientific, negatives) in path data.
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
# Coordinate-bearing SVG elements we know how to rasterize.
GEOM_TAGS = ("path", "polyline", "line", "polygon")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_tars(src: Path, work: Path) -> None:
    """Extract every *.tar.xz under `src` into `work` (idempotent)."""
    work.mkdir(parents=True, exist_ok=True)
    tars = sorted(src.rglob("*.tar.xz"))
    if not tars:
        print(f"WARNING: no *.tar.xz under {src}", file=sys.stderr)
    for t in tars:
        marker = work / (t.stem + ".done")
        if marker.exists():
            print(f"[extract] {t.name} already extracted.")
            continue
        print(f"[extract] {t.name} -> {work}")
        with tarfile.open(t, "r:xz") as tf:
            tf.extractall(work)
        marker.write_text("ok")


# --------------------------------------------------------------------------- #
# SVG parsing
# --------------------------------------------------------------------------- #
def iter_elements(svg_text: str):
    """Yield (tag, attr_dict) for each geometry element in an SVG string."""
    for m in re.finditer(r"<(\w+)\b([^>]*)>", svg_text):
        tag = m.group(1)
        if tag not in GEOM_TAGS:
            continue
        attrs = dict(re.findall(r'([\w:-]+)\s*=\s*"([^"]*)"', m.group(2)))
        yield tag, attrs


def element_points(tag: str, attrs: dict) -> list[tuple[float, float]]:
    """Extract an ordered list of (x, y) vertices from a geometry element."""
    if tag == "line":
        try:
            return [(float(attrs["x1"]), float(attrs["y1"])),
                    (float(attrs["x2"]), float(attrs["y2"]))]
        except (KeyError, ValueError):
            return []
    if tag in ("polyline", "polygon"):
        nums = [float(n) for n in NUM.findall(attrs.get("points", ""))]
    else:  # path
        nums = [float(n) for n in NUM.findall(attrs.get("d", ""))]
    return list(zip(nums[0::2], nums[1::2]))


def svg_canvas_size(svg_text: str) -> tuple[float, float] | None:
    """Return (width, height) from viewBox or width/height attributes."""
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text)
    if vb:
        parts = [float(x) for x in NUM.findall(vb.group(1))]
        if len(parts) == 4:
            return parts[2], parts[3]
    w = re.search(r'\bwidth\s*=\s*"([\d.]+)', svg_text)
    h = re.search(r'\bheight\s*=\s*"([\d.]+)', svg_text)
    if w and h:
        return float(w.group(1)), float(h.group(1))
    return None


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def probe(src: Path, work: Path, sample: int) -> None:
    extract_tars(src, work)
    svgs = sorted(work.rglob("*.svg"))[:sample]
    print(f"\nFound {len(list(work.rglob('*.svg')))} SVGs (showing {len(svgs)}).")
    if not svgs:
        print("ERROR: no SVGs extracted — check --src.", file=sys.stderr)
        sys.exit(2)

    sem_hist: collections.Counter = collections.Counter()
    sem_colors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    attr_names: collections.Counter = collections.Counter()
    png_hits = 0
    for s in svgs:
        txt = s.read_text(errors="ignore")
        if (s.with_suffix(".png")).exists():
            png_hits += 1
        size = svg_canvas_size(txt)
        for tag, attrs in iter_elements(txt):
            attr_names.update(attrs.keys())
            sid = attrs.get("semantic-id") or attrs.get("semanticId") or attrs.get("class")
            if sid is not None:
                sem_hist[sid] += 1
                color = attrs.get("stroke") or attrs.get("fill") or "?"
                sem_colors[sid][color] += 1
        print(f"  {s.name}: canvas={size}, sibling PNG={'yes' if s.with_suffix('.png').exists() else 'NO'}")

    print("\nATTRIBUTE NAMES seen on geometry elements:")
    print(" ", dict(attr_names.most_common(20)))
    print(f"\nSibling PNGs: {png_hits}/{len(svgs)} SVGs have a matching .png")
    print("\nsemantic-id histogram (id: count  [top stroke colors]):")
    for sid, cnt in sem_hist.most_common(40):
        cols = ", ".join(f"{c}×{n}" for c, n in sem_colors[sid].most_common(3))
        print(f"  {sid:>4}: {cnt:6d}   [{cols}]")
    print("\n>>> Identify the wall id(s) above (FloorPlanCAD walls are drawn "
          "in dark red). Pass them to `convert --wall-ids`.")


# --------------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------------- #
def render_svg_to_png(svg_path: Path, out_png: Path, imgsz: int) -> bool:
    """Best-effort SVG->PNG render (only used when no sibling PNG exists)."""
    try:
        import cairosvg
    except ImportError:
        return False
    try:
        cairosvg.svg2png(url=str(svg_path), write_to=str(out_png),
                         output_width=imgsz, output_height=imgsz)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[render fail] {svg_path.name}: {e}", file=sys.stderr)
        return False


def build_wall_mask(svg_text: str, wall_ids: set[str], imgsz: int,
                    line_w: int) -> Image.Image | None:
    """Rasterize wall primitives into an (imgsz×imgsz) binary mask."""
    size = svg_canvas_size(svg_text)
    if not size or size[0] <= 0 or size[1] <= 0:
        return None
    sx, sy = imgsz / size[0], imgsz / size[1]
    mask = Image.new("L", (imgsz, imgsz), 0)
    draw = ImageDraw.Draw(mask)
    drew = False
    for tag, attrs in iter_elements(svg_text):
        sid = attrs.get("semantic-id") or attrs.get("semanticId") or attrs.get("class")
        if sid not in wall_ids:
            continue
        pts = [(x * sx, y * sy) for x, y in element_points(tag, attrs)]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=line_w, joint="curve")
            drew = True
        elif len(pts) == 1:
            x, y = pts[0]
            r = line_w / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            drew = True
    return mask if drew else mask  # return even empty mask (valid negative)


def split_of(stem: str, val_frac: float) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 / 1000.0
    return "val" if h < val_frac else "train"


def convert(src: Path, work: Path, dst: Path, wall_ids: set[str], imgsz: int,
            line_w: int, val_frac: float, limit: int | None) -> None:
    extract_tars(src, work)
    svgs = sorted(work.rglob("*.svg"))
    if limit:
        svgs = svgs[:limit]
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "masks" / split).mkdir(parents=True, exist_ok=True)

    stats = {"pairs": 0, "wall_px": 0, "no_image": 0, "no_canvas": 0}
    for s in svgs:
        txt = s.read_text(errors="ignore")
        split = split_of(s.stem, val_frac)
        # input image: prefer sibling PNG, else render
        png = s.with_suffix(".png")
        out_img = dst / "images" / split / f"{s.stem}.png"
        if png.exists():
            Image.open(png).convert("RGB").resize((imgsz, imgsz)).save(out_img)
        elif not render_svg_to_png(s, out_img, imgsz):
            stats["no_image"] += 1
            continue
        mask = build_wall_mask(txt, wall_ids, imgsz, line_w)
        if mask is None:
            stats["no_canvas"] += 1
            out_img.unlink(missing_ok=True)
            continue
        mask.save(dst / "masks" / split / f"{s.stem}.png")
        stats["pairs"] += 1
        stats["wall_px"] += sum(mask.point(lambda p: 1 if p else 0)
                                .getdata())

    print("\nDone (wall segmentation dataset).")
    print(f"  pairs written:   {stats['pairs']}")
    print(f"  skipped no-image:{stats['no_image']}, no-canvas:{stats['no_canvas']}")
    if stats["pairs"]:
        frac = stats["wall_px"] / (stats["pairs"] * imgsz * imgsz)
        print(f"  mean wall pixel fraction: {frac:.4f} "
              f"(sanity: should be ~0.02–0.15; near 0 means wrong --wall-ids)")
    (dst / "meta.txt").write_text(
        f"wall_ids={sorted(wall_ids)} imgsz={imgsz} line_w={line_w}\n")
    if stats["pairs"] == 0:
        print("ERROR: 0 pairs — check --src / extraction.", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--sample", type=int, default=8)

    c = sub.add_parser("convert")
    c.add_argument("--src", required=True, type=Path)
    c.add_argument("--work", required=True, type=Path)
    c.add_argument("--dst", required=True, type=Path)
    c.add_argument("--wall-ids", required=True,
                   help="Comma-separated semantic-id values for walls, e.g. '1,2'.")
    c.add_argument("--imgsz", type=int, default=640)
    c.add_argument("--line-w", type=int, default=4,
                   help="Stroke width (px) used to rasterize wall lines.")
    c.add_argument("--val-frac", type=float, default=0.1)
    c.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "probe":
        probe(args.src.resolve(), args.work.resolve(), args.sample)
    else:
        wall_ids = {w.strip() for w in args.wall_ids.split(",") if w.strip()}
        convert(args.src.resolve(), args.work.resolve(), args.dst.resolve(),
                wall_ids, args.imgsz, args.line_w, args.val_frac, args.limit)


if __name__ == "__main__":
    main()
