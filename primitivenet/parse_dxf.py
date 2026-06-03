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
import argparse, collections, hashlib, json, math, random, sys
from pathlib import Path


def split_for(pid, seed=42, val_frac=0.15):      # matches render_dataset.split_for
    h = int(hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"

import ezdxf
from ezdxf import bbox

CLASS_NAMES = ["clutter", "wall", "door", "window", "column", "hatch", "duplicate"]
NAME2ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# 2nd head: subtype. Predicted independently of the coarse class, so a wrong
# subtype still leaves the object correctly a 'door'. No materials.
SUBTYPE_NAMES = ["none", "interior", "exterior",
                 "single", "double", "sliding", "pocket", "bifold", "french", "garage",
                 "casement", "fixed", "bay", "bow", "awning", "louvre", "clerestory", "corner"]
SUB2ID = {n: i for i, n in enumerate(SUBTYPE_NAMES)}
NUM_SUBTYPES = len(SUBTYPE_NAMES)

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


def _entity_label(e, dup_handles=frozenset(), sub_map=None):
    """(coarse, subtype) for one top-level entity.
    - duplicates -> (duplicate, none)
    - walls -> (wall, interior/exterior by layer)
    - doors/windows (INSERT) -> (door/window, subtype from sub_map[OPN_<id>])
    sub_map is None at inference (no scenario) -> opening subtype = none (model predicts)."""
    if getattr(e.dxf, "handle", None) in dup_handles:
        return NAME2ID["duplicate"], 0
    layer = getattr(e.dxf, "layer", "0")
    if e.dxftype() == "INSERT":
        coarse = _label(layer, e.dxf.name)
        sub = 0
        if sub_map and coarse in (NAME2ID["door"], NAME2ID["window"]):
            oid = str(e.dxf.name).split("_", 1)[-1]          # OPN_D5 -> D5
            sub = SUB2ID.get(str(sub_map.get(oid, "")).lower(), 0)
        return coarse, sub
    coarse = _label(layer, None)
    sub = 0
    if coarse == NAME2ID["wall"]:
        u = layer.upper()
        sub = SUB2ID["exterior"] if u == "A-WALL-FULL" else (SUB2ID["interior"] if u == "A-WALL-INTR" else 0)
    return coarse, sub


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


def _region_emit(pts, W, H, x0o, y0o):
    """A polygon -> one 'region' token (type=region). Used for hatch regions."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    perim = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(len(pts)))
    feat = feature("region", minx - x0o, miny - y0o, maxx - x0o, maxy - y0o,
                   cx - x0o, cy - y0o, perim, 0.0, 0.0, 0.0, maxx - minx, maxy - miny, W, H)
    g = [round(v, 2) for p in pts for v in p]   # full polygon for reconstruction
    return [round(f, 5) for f in feat], g


def _walk(entity, forced, prims, W, H, x0o, y0o, depth=0):
    """Append primitives for one entity as (t, feat, coarse, sub, geom). forced=
    (coarse, sub) is applied to the entity and recursed into INSERT children."""
    t = entity.dxftype()
    if t == "INSERT" and depth < 4:
        try:
            for ve in entity.virtual_entities():
                _walk(ve, forced, prims, W, H, x0o, y0o, depth + 1)
        except Exception:
            pass
        return
    coarse, sub = forced
    try:
        if t == "LINE":
            a, b = entity.dxf.start, entity.dxf.end
            length = math.hypot(b.x - a.x, b.y - a.y); ang = math.atan2(b.y - a.y, b.x - a.x)
            f, g = _emit("line", (a.x, a.y), (b.x, b.y), ((a.x + b.x) / 2, (a.y + b.y) / 2),
                         length, ang, 0.0, 0.0, W, H, x0o, y0o)
            prims.append((0, f, coarse, sub, g))
        elif t == "ARC":
            a, b = entity.start_point, entity.end_point
            c = entity.dxf.center; r = entity.dxf.radius
            sweep = math.radians((entity.dxf.end_angle - entity.dxf.start_angle) % 360)
            ang = math.atan2(b.y - a.y, b.x - a.x)
            f, g = _emit("arc", (a.x, a.y), (b.x, b.y), (c.x, c.y), r * sweep, ang, r, sweep, W, H, x0o, y0o)
            prims.append((1, f, coarse, sub, g))
        elif t in ("CIRCLE", "ELLIPSE"):
            c = entity.dxf.center; r = getattr(entity.dxf, "radius", 0.0) or 1.0
            f, g = _emit("circle", (c.x - r, c.y), (c.x + r, c.y), (c.x, c.y),
                         2 * math.pi * r, 0.0, r, 2 * math.pi, W, H, x0o, y0o)
            prims.append((2, f, coarse, sub, g))
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
                prims.append((0, f, coarse, sub, g))
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
        _walk(e, _entity_label(e), prims, W, H, x0o, y0o)   # sub_map=None at inference
    for r in hd.regions:                     # hatch -> region tokens
        f, g = _region_emit(r.boundary, W, H, x0o, y0o)
        prims.append((4, f, NAME2ID["hatch"], 0, g))
    return W, H, prims, hd.regions


def primitives_synth(dxf_path, dup_handles=frozenset(), add_hatch=True, sub_map=None,
                     wall_layers=("A-WALL-FULL", "A-WALL-INTR")):
    """Fast synthetic extraction (hatch-free DXF, no HatchDetector). Labels coarse+sub
    by layer / scenario, forces 'duplicate' for injected dup handles, and (if add_hatch)
    adds one 'hatch' region token per wall polygon (binary 'hatch present')."""
    doc = ezdxf.readfile(dxf_path); msp = doc.modelspace()
    ext = bbox.extents(msp)
    if ext is None or not ext.has_data:
        return 0, 0, []
    x0o, y0o = ext.extmin.x, ext.extmin.y
    W = max(1e-6, ext.extmax.x - x0o); H = max(1e-6, ext.extmax.y - y0o)
    prims = []
    for e in msp:
        _walk(e, _entity_label(e, dup_handles, sub_map), prims, W, H, x0o, y0o)
    if add_hatch:
        wl = {l.upper() for l in wall_layers}
        for e in msp:
            if (e.dxftype() == "LWPOLYLINE" and e.dxf.handle not in dup_handles
                    and getattr(e.dxf, "layer", "").upper() in wl):
                poly = [(p[0], p[1]) for p in e.get_points()]
                if len(poly) >= 3:
                    f, g = _region_emit(poly, W, H, x0o, y0o)
                    prims.append((4, f, NAME2ID["hatch"], 0, g))
    return W, H, prims


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


def convert(dxf_dir, out, layer_map, limit, hatch_layers=("A-WALL-PATT",), seed=42, val_frac=0.15):
    """Convert a dir of DXFs (e.g. render_dataset's dataset_10k/dxf/) -> primitive
    JSON, auto-split train/val per plan to match render_dataset.split_for."""
    if layer_map:
        global LAYER2CLASS
        LAYER2CLASS = {k.lower(): v for k, v in json.loads(Path(layer_map).read_text()).items()}
    files = sorted(Path(dxf_dir).rglob("*.dxf"))
    if limit: files = files[:limit]
    (Path(out) / "train").mkdir(parents=True, exist_ok=True)
    (Path(out) / "val").mkdir(parents=True, exist_ok=True)
    n_plans = n_prim = 0; hist = collections.Counter()
    for fp in files:
        W, H, prims, regions = primitives_of(str(fp), hatch_layers=list(hatch_layers))
        if not prims: continue
        rec = [{"feat": f, "sem": c, "sub": s} for (_t, f, c, s, _g) in prims]
        for r in rec: hist[r["sem"]] += 1
        split = split_for(fp.stem, seed, val_frac)
        (Path(out) / split / f"{fp.stem}.json").write_text(json.dumps(
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
    c.add_argument("--layer-map"); c.add_argument("--limit", type=int)
    c.add_argument("--hatch-layers", nargs="+", default=["A-WALL-PATT"])
    c.add_argument("--seed", type=int, default=42); c.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    if a.cmd == "probe": probe(a.dxf_dir, a.sample)
    else: convert(a.dxf_dir, a.out, a.layer_map, a.limit, a.hatch_layers, a.seed, a.val_frac)


if __name__ == "__main__":
    main()
