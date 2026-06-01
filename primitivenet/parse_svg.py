"""Parse FloorPlanCAD (catwhisker) SVGs into per-primitive features + labels.

This is the foundation of Option B: a per-primitive panoptic labeler.
FloorPlanCAD annotates every primitive with `semantic-id` (1..35) and
`instance-id` (-1 for stuff like walls). We turn each <path>/<circle>/
<ellipse>/<line> into a feature vector + its (semantic_id, instance_id).

Two modes:
  probe    Extract a few SVGs (handles .tar.xz) and print the schema:
           attribute names, semantic-id histogram + stroke colors, and a
           sample element. RUN THIS FIRST to confirm the format and lock
           the semantic-id -> name mapping.
  convert  Parse every SVG to a per-plan JSON: {width,height,primitives:[
           {type, feat:[...], sem, ins}, ...]}. Feature layout is documented
           in `primitive_feature()`.

Usage:
  python -m primitivenet.parse_svg probe   --src <dir-with-svgs-or-tarxz> --work /tmp/fpc
  python -m primitivenet.parse_svg convert --src <dir> --work /tmp/fpc --out data/fpc_json --split train
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import tarfile
from pathlib import Path

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
GEOM_TAGS = ("path", "line", "polyline", "circle", "ellipse")
TYPE_ID = {"line": 0, "polyline": 0, "path": 1, "circle": 2, "ellipse": 2}  # line, curve, round
NUM_TYPES = 3
FEAT_DIM = NUM_TYPES + 9  # one-hot type (3) + [x0,y0,x1,y1,cx,cy,len,sin,cos] (9)


# --------------------------------------------------------------------------- #
def extract_tars(src: Path, work: Path) -> Path:
    """Extract any *.tar.xz under src into work (idempotent). Returns the dir
    that actually contains SVGs (src if already extracted)."""
    if list(src.rglob("*.svg")):
        return src
    work.mkdir(parents=True, exist_ok=True)
    for t in sorted(src.rglob("*.tar.xz")):
        marker = work / (t.stem + ".done")
        if marker.exists():
            continue
        print(f"[extract] {t.name}")
        with tarfile.open(t, "r:xz") as tf:
            tf.extractall(work)
        marker.write_text("ok")
    return work


def iter_elements(svg_text: str):
    for m in re.finditer(r"<(\w+)\b([^>]*)>", svg_text):
        tag = m.group(1)
        if tag in GEOM_TAGS:
            yield tag, dict(re.findall(r'([\w:-]+)\s*=\s*"([^"]*)"', m.group(2)))


def canvas_size(svg_text: str) -> tuple[float, float]:
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text)
    if vb:
        p = [float(x) for x in NUM.findall(vb.group(1))]
        if len(p) == 4 and p[2] > 0 and p[3] > 0:
            return p[2], p[3]
    w = re.search(r'\bwidth\s*=\s*"([\d.]+)', svg_text)
    h = re.search(r'\bheight\s*=\s*"([\d.]+)', svg_text)
    return (float(w.group(1)), float(h.group(1))) if w and h else (0.0, 0.0)


def _endpoints(tag: str, a: dict):
    """Return (x0,y0,x1,y1,cx,cy) in raw SVG units, or None."""
    if tag == "line":
        try:
            x0, y0, x1, y1 = (float(a["x1"]), float(a["y1"]),
                              float(a["x2"]), float(a["y2"]))
        except (KeyError, ValueError):
            return None
    elif tag in ("circle", "ellipse"):
        try:
            cx, cy = float(a["cx"]), float(a["cy"])
        except (KeyError, ValueError):
            return None
        r = float(a.get("r", a.get("rx", 0)) or 0)
        return cx - r, cy, cx + r, cy, cx, cy
    else:  # path / polyline
        nums = [float(n) for n in NUM.findall(a.get("d", a.get("points", "")))]
        if len(nums) < 4:
            return None
        x0, y0, x1, y1 = nums[0], nums[1], nums[-2], nums[-1]
    return x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2


def primitive_feature(tag: str, a: dict, W: float, H: float):
    """Feature vector (FEAT_DIM): one-hot type(3) + normalized
    [x0,y0,x1,y1,cx,cy], normalized length, sin/cos of angle."""
    ep = _endpoints(tag, a)
    if ep is None or W <= 0 or H <= 0:
        return None
    x0, y0, x1, y1, cx, cy = ep
    onehot = [0.0] * NUM_TYPES
    onehot[TYPE_ID.get(tag, 1)] = 1.0
    dx, dy = (x1 - x0), (y1 - y0)
    length = math.hypot(dx, dy)
    ang = math.atan2(dy, dx)
    diag = math.hypot(W, H) or 1.0
    feat = onehot + [x0 / W, y0 / H, x1 / W, y1 / H, cx / W, cy / H,
                     length / diag, math.sin(ang), math.cos(ang)]
    return feat


def sem_ins(a: dict):
    sid = a.get("semantic-id", a.get("semanticId", a.get("semantic")))
    iid = a.get("instance-id", a.get("instanceId", a.get("instance")))
    try:
        sid = int(float(sid)) if sid is not None else 0
    except ValueError:
        sid = 0
    try:
        iid = int(float(iid)) if iid is not None else -1
    except ValueError:
        iid = -1
    return sid, iid


# --------------------------------------------------------------------------- #
def probe(src: Path, work: Path, sample: int) -> None:
    root = extract_tars(src, work)
    svgs = sorted(root.rglob("*.svg"))
    print(f"Found {len(svgs)} SVGs total.")
    if not svgs:
        print("ERROR: no SVGs found under", src, file=sys.stderr); sys.exit(2)
    attrs = collections.Counter()
    sem = collections.Counter()
    sem_color = collections.defaultdict(collections.Counter)
    tags = collections.Counter()
    for s in svgs[:sample]:
        txt = s.read_text(errors="ignore")
        print(f"  {s.name}: canvas={canvas_size(txt)}")
        for tag, a in iter_elements(txt):
            tags[tag] += 1
            attrs.update(a.keys())
            sid, _ = sem_ins(a)
            sem[sid] += 1
            sem_color[sid][a.get("stroke") or a.get("fill") or "?"] += 1
    print("\nELEMENT TAGS:", dict(tags))
    print("ATTRIBUTE NAMES:", dict(attrs.most_common(20)))
    print("\nsemantic-id histogram (id: count  [top colors]):")
    for sid, c in sem.most_common(40):
        cols = ", ".join(f"{k}×{v}" for k, v in sem_color[sid].most_common(3))
        print(f"  {sid:>4}: {c:6d}   [{cols}]")
    print("\n>>> Map these semantic-ids to names (wall/door/window/...) for the "
          "output. Training works on raw ids regardless.")


def convert(src: Path, work: Path, out: Path, split: str, limit) -> None:
    root = extract_tars(src, work)
    svgs = sorted(root.rglob("*.svg"))
    if limit:
        svgs = svgs[:limit]
    out_dir = out / split
    out_dir.mkdir(parents=True, exist_ok=True)
    n_plans = n_prim = 0
    sem_hist = collections.Counter()
    for s in svgs:
        txt = s.read_text(errors="ignore")
        W, H = canvas_size(txt)
        prims = []
        for tag, a in iter_elements(txt):
            feat = primitive_feature(tag, a, W, H)
            if feat is None:
                continue
            sid, iid = sem_ins(a)
            prims.append({"t": TYPE_ID.get(tag, 1), "feat": [round(f, 5) for f in feat],
                          "sem": sid, "ins": iid})
            sem_hist[sid] += 1
        if not prims:
            continue
        (out_dir / f"{s.stem}.json").write_text(json.dumps(
            {"width": W, "height": H, "primitives": prims}))
        n_plans += 1
        n_prim += len(prims)
    print(f"[{split}] wrote {n_plans} plans, {n_prim} primitives -> {out_dir}")
    print(f"  semantic-id histogram: {dict(sem_hist.most_common(40))}")
    if n_plans == 0:
        print("ERROR: 0 plans converted.", file=sys.stderr); sys.exit(2)
    if set(sem_hist) <= {0}:
        print("ERROR: every primitive has semantic-id 0 — labels were NOT "
              "parsed (attribute name/format mismatch). Run `probe` and check "
              "the attribute names before training.", file=sys.stderr)
        sys.exit(3)


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
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--split", default="train")
    c.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.src.resolve(), a.work.resolve(), a.sample)
    else:
        convert(a.src.resolve(), a.work.resolve(), a.out.resolve(), a.split, a.limit)


if __name__ == "__main__":
    main()
