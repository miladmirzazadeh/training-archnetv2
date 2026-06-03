"""Parse generator DXFs into per-primitive features + labels (no PNG needed).

Same output schema as parse_svg.convert so dataset.py / train.py / model.py work
unchanged:  {"width", "height", "primitives":[{"t","feat","sem","ins","g"}]}.

  feat : FEAT_DIM geometric features (see feature()).
  sem  : class id (0 clutter, 1 wall, 2 door, 3 window, 4 column)  <- the LABEL.
  g    : raw WCS geometry kept so inference can rebuild exact DXF coords.

Labels come from the DXF the generator wrote: each semantic class lives on its
own layer (and/or block name). Use `probe` to see the layers/blocks present,
then set LAYER2CLASS / BLOCK_RULES (or pass --layer-map a JSON file).

Subcommands:
  probe   --dxf-dir DIR [--sample N]          # inspect entity/layer/block names
  convert --dxf-dir DIR --out OUT --split train|test [--layer-map map.json] [--limit N]
"""
from __future__ import annotations
import argparse, collections, json, math, random, sys
from pathlib import Path

import ezdxf
from ezdxf import bbox

CLASS_NAMES = ["clutter", "wall", "door", "window", "column", "hatch"]
NAME2ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Vitruev_synthdata layer scheme (generator/style.py). Block references for
# doors/windows/columns are placed on the component layer, so the INSERT's layer
# is authoritative — no block-name rules needed.
LAYER2CLASS = {
    "a-wall-full": "wall",   # exterior / full-thickness walls
    "a-wall-intr": "wall",   # interior / partition walls
    "a-door": "door",        # door blocks (leaf + swing arc)
    "a-glaz": "window",      # glazing = windows
    "a-cols": "column",
    # everything else -> clutter (DEFAULT_CLASS):
    #   a-wall-patt (hatch), a-anno-dims/text/ttlb, a-furn, a-grid, a-misc, "0"
}
BLOCK_RULES = {}             # INSERT layer already distinguishes door/window/column
# Anything unmatched -> clutter (0): dropped at output, and the model learns to ignore it.
DEFAULT_CLASS = "clutter"

_TYPES = {"line": 0, "arc": 1, "circle": 2, "other": 3, "region": 4}
NUM_TYPES = 5
FEAT_DIM = NUM_TYPES + 13   # one-hot type(5) + [x0,y0,x1,y1,cx,cy,len,sin,cos,rad,sweep,bw,bh]


def _label(layer: str, block: str | None) -> int:
    if block:
        for k, v in BLOCK_RULES.items():
            if k in block.lower():
                return NAME2ID[v]
    return NAME2ID.get(LAYER2CLASS.get((layer or "").lower(), DEFAULT_CLASS), 0)


def feature(kind, x0, y0, x1, y1, cx, cy, length, ang, radius, sweep, bw, bh, W, H):
    diag = math.hypot(W, H) or 1.0
    oh = [0.0] * NUM_TYPES; oh[_TYPES.get(kind, 3)] = 1.0
    return oh + [x0 / W, y0 / H, x1 / W, y1 / H, cx / W, cy / H,
                 length / diag, math.sin(ang), math.cos(ang),
                 radius / diag, sweep / math.pi, bw / W, bh / H]


def _emit(kind, p0, p1, center, length, ang, radius, sweep, W, H, x0o, y0o):
    """Build (feat, raw-geom) with coords normalized to the plan bbox."""
    x0, y0 = p0[0] - x0o, p0[1] - y0o
    x1, y1 = p1[0] - x0o, p1[1] - y0o
    cx, cy = center[0] - x0o, center[1] - y0o
    bw, bh = abs(x1 - x0), abs(y1 - y0)
    feat = feature(kind, x0, y0, x1, y1, cx, cy, length, ang, radius, sweep, bw, bh, W, H)
    g = [round(v, 3) for v in (p0[0], p0[1], p1[0], p1[1], center[0], center[1], radius, sweep)]
    return [round(f, 5) for f in feat], g


def _region_emit(region, W, H, x0o, y0o):
    """A hatch region -> one 'region' token (type=region, label=hatch)."""
    pts = region.boundary
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    perim = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(len(pts)))
    feat = feature("region", minx - x0o, miny - y0o, maxx - x0o, maxy - y0o,
                   cx - x0o, cy - y0o, perim, 0.0, 0.0, 0.0, maxx - minx, maxy - miny, W, H)
    g = [round(v, 2) for p in pts for v in p]   # full polygon for reconstruction
    return [round(f, 5) for f in feat], g


def _walk(entity, forced_label, prims, W, H, x0o, y0o, depth=0):
    """Append primitives for one DXF entity (recursing INSERT blocks)."""
    t = entity.dxftype()
    layer = getattr(entity.dxf, "layer", "0")
    if t == "INSERT" and depth < 4:
        block = entity.dxf.name
        lab = _label(layer, block)
        try:
            for ve in entity.virtual_entities():
                _walk(ve, lab, prims, W, H, x0o, y0o, depth + 1)
        except Exception:
            pass
        return
    lab = forced_label if forced_label is not None else _label(layer, None)
    try:
        if t == "LINE":
            a, b = entity.dxf.start, entity.dxf.end
            length = math.hypot(b.x - a.x, b.y - a.y); ang = math.atan2(b.y - a.y, b.x - a.x)
            f, g = _emit("line", (a.x, a.y), (b.x, b.y), ((a.x + b.x) / 2, (a.y + b.y) / 2),
                         length, ang, 0.0, 0.0, W, H, x0o, y0o)
            prims.append((0, f, lab, g))
        elif t == "ARC":
            a, b = entity.start_point, entity.end_point
            c = entity.dxf.center; r = entity.dxf.radius
            sweep = math.radians((entity.dxf.end_angle - entity.dxf.start_angle) % 360)
            ang = math.atan2(b.y - a.y, b.x - a.x)
            f, g = _emit("arc", (a.x, a.y), (b.x, b.y), (c.x, c.y), r * sweep, ang, r, sweep, W, H, x0o, y0o)
            prims.append((1, f, lab, g))
        elif t in ("CIRCLE", "ELLIPSE"):
            c = entity.dxf.center; r = getattr(entity.dxf, "radius", 0.0) or 1.0
            f, g = _emit("circle", (c.x - r, c.y), (c.x + r, c.y), (c.x, c.y),
                         2 * math.pi * r, 0.0, r, 2 * math.pi, W, H, x0o, y0o)
            prims.append((2, f, lab, g))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = [(p[0], p[1]) for p in entity.get_points()] if t == "LWPOLYLINE" \
                  else [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if getattr(entity, "closed", False) and len(pts) > 2:
                pts = pts + [pts[0]]
            for a, b in zip(pts, pts[1:]):
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                if length == 0: continue
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                f, g = _emit("line", a, b, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                             length, ang, 0.0, 0.0, W, H, x0o, y0o)
                prims.append((0, f, lab, g))
    except Exception:
        pass


def primitives_of(dxf_path, hatch_layers=None):
    """Return (W, H, prims, regions). HatchDetector strips raw hatch and emits each
    hatch as ONE 'region' token (label='hatch'); the rest are labeled by layer.
    prims: list[(t, feat, label, geom)]; regions: list[HatchRegion]."""
    from primitivenet.hatch import HatchDetector
    try:
        hd = HatchDetector(dxf_path, hatch_layers=hatch_layers)
    except Exception:
        return 0, 0, [], []
    W, H, x0o, y0o = hd.W, hd.H, hd.x0, hd.y0
    if W <= 1e-6 or H <= 1e-6:
        return 0, 0, [], []
    prims = []
    for e in hd.clean_entities():            # hatch already removed
        _walk(e, None, prims, W, H, x0o, y0o)
    for r in hd.regions:                     # hatch -> region tokens
        f, g = _region_emit(r, W, H, x0o, y0o)
        prims.append((4, f, NAME2ID["hatch"], g))
    return W, H, prims, hd.regions


def probe(dxf_dir, sample):
    files = sorted(Path(dxf_dir).rglob("*.dxf"))[:sample]
    print(f"Probing {len(files)} DXFs…")
    types, layers, blocks = collections.Counter(), collections.Counter(), collections.Counter()
    for fp in files:
        doc = ezdxf.readfile(fp); msp = doc.modelspace()
        for e in msp:
            types[e.dxftype()] += 1
            layers[getattr(e.dxf, "layer", "0")] += 1
            if e.dxftype() == "INSERT":
                blocks[e.dxf.name] += 1
    print("ENTITY TYPES:", dict(types))
    print("LAYERS:", dict(layers.most_common(30)))
    print("BLOCK NAMES:", dict(blocks.most_common(30)))
    print("\n>>> Map LAYERS/BLOCKS to classes in LAYER2CLASS / BLOCK_RULES "
          "(or pass --layer-map a JSON {layer: class}).")


def convert(dxf_dir, out, split, layer_map, limit, hatch_layers=("A-WALL-PATT",)):
    if layer_map:
        global LAYER2CLASS
        LAYER2CLASS = {k.lower(): v for k, v in json.loads(Path(layer_map).read_text()).items()}
    files = sorted(Path(dxf_dir).rglob("*.dxf"))
    if limit: files = files[:limit]
    out_dir = Path(out) / split; out_dir.mkdir(parents=True, exist_ok=True)
    n_plans = n_prim = 0; hist = collections.Counter()
    for fp in files:
        W, H, prims, regions = primitives_of(str(fp), hatch_layers=list(hatch_layers))
        if not prims: continue
        rec = [{"feat": f, "sem": lab} for (_t, f, lab, _g) in prims]
        for r in rec: hist[r["sem"]] += 1
        (out_dir / f"{fp.stem}.json").write_text(json.dumps(
            {"width": round(W, 2), "height": round(H, 2), "primitives": rec,
             "hatch_regions": [r.to_dict() for r in regions]}))
        n_plans += 1; n_prim += len(rec)
    print(f"[{split}] wrote {n_plans} plans, {n_prim} primitives -> {out_dir}")
    print("  class histogram:", {CLASS_NAMES[k]: v for k, v in sorted(hist.items())})
    if n_plans == 0:
        print("ERROR: 0 plans converted.", file=sys.stderr); sys.exit(2)
    if set(hist) <= {0}:
        print("ERROR: every primitive is 'clutter' — layer/block map didn't match. "
              "Run `probe` and fix LAYER2CLASS/BLOCK_RULES.", file=sys.stderr); sys.exit(3)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe");   p.add_argument("--dxf-dir", required=True); p.add_argument("--sample", type=int, default=6)
    c = sub.add_parser("convert"); c.add_argument("--dxf-dir", required=True); c.add_argument("--out", required=True)
    c.add_argument("--split", default="train"); c.add_argument("--layer-map"); c.add_argument("--limit", type=int)
    c.add_argument("--hatch-layers", nargs="+", default=["A-WALL-PATT"])
    a = ap.parse_args()
    if a.cmd == "probe": probe(a.dxf_dir, a.sample)
    else: convert(a.dxf_dir, a.out, a.split, a.layer_map, a.limit, a.hatch_layers)


if __name__ == "__main__":
    main()
