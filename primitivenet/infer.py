"""Run PrimitiveNet on a CAD plan and emit DPSS-style output:
per-instance {label, points, primitives} JSON + a colored overlay.

Pipeline: parse primitives -> classify each -> group same-class primitives
that share endpoints into instances (panoptic 'things'); 'stuff' classes
(e.g. wall, background) are emitted as one semantic group.

Input: a FloorPlanCAD SVG (for validation). For vtruvian integration the
same model runs on your `_lines.json` once converted to the same feature
layout (see parse_svg.primitive_feature).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from primitivenet.model import PrimitiveNet
from primitivenet.parse_svg import (FEAT_DIM, canvas_size, iter_elements,
                                    primitive_feature, _endpoints, TYPE_ID)

# Default FloorPlanCAD-ish names; override with --names JSON {id: name}.
# Confirm exact ids from `parse_svg probe` output.
DEFAULT_NAMES = {0: "background"}
STUFF = {"wall", "background", "curtain wall", "railing"}
PALETTE = [(200, 30, 30), (30, 90, 220), (30, 160, 60), (210, 140, 20),
           (150, 40, 200), (0, 160, 160), (160, 110, 60), (200, 60, 160)]


def parse_plan(svg_text):
    W, H = canvas_size(svg_text)
    prims = []
    for tag, a in iter_elements(svg_text):
        f = primitive_feature(tag, a, W, H)
        ep = _endpoints(tag, a)
        if f is None or ep is None:
            continue
        prims.append({"type": TYPE_ID.get(tag, 1), "feat": f,
                      "p0": (ep[0], ep[1]), "p1": (ep[2], ep[3])})
    return W, H, prims


def group_instances(prims, classes, names, tol_frac=0.01, diag=1.0):
    """Union-find: same predicted class + shared endpoint -> one instance.
    Stuff classes are not instance-grouped (one group per stuff class)."""
    tol = tol_frac * diag
    parent = list(range(len(prims)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # bucket endpoints for proximity join within same class
    import collections
    buckets = collections.defaultdict(list)
    for i, p in enumerate(prims):
        if names.get(classes[i], str(classes[i])) in STUFF:
            continue
        for (x, y) in (p["p0"], p["p1"]):
            buckets[(round(x / max(tol, 1e-6)), round(y / max(tol, 1e-6)),
                     classes[i])].append(i)
    for ids in buckets.values():
        for k in range(1, len(ids)):
            union(ids[0], ids[k])

    inst = collections.defaultdict(list)
    for i, p in enumerate(prims):
        name = names.get(classes[i], str(classes[i]))
        if name == "background":
            continue
        key = name if name in STUFF else (name, find(i))
        inst[key].append(i)

    out = []
    for key, idxs in inst.items():
        name = key if isinstance(key, str) else key[0]
        pts = []
        for i in idxs:
            pts.append(list(prims[i]["p0"])); pts.append(list(prims[i]["p1"]))
        xs = [x for x, _ in pts]; ys = [y for _, y in pts]
        out.append({"label": name, "n_primitives": len(idxs),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "points": pts, "primitive_indices": idxs})
    return out


def overlay(W, H, prims, classes, names, scale=1.0):
    img = Image.new("RGB", (max(1, int(W * scale)), max(1, int(H * scale))), "white")
    d = ImageDraw.Draw(img)
    for i, p in enumerate(prims):
        name = names.get(classes[i], str(classes[i]))
        if name == "background":
            col = (220, 220, 220)
        else:
            col = PALETTE[hash(name) % len(PALETTE)]
        (x0, y0), (x1, y1) = p["p0"], p["p1"]
        d.line([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill=col, width=2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--svg", required=True, type=Path, help="FloorPlanCAD SVG.")
    ap.add_argument("--names", type=Path, default=None,
                    help="JSON {semantic_id: name}; from `parse_svg probe`.")
    ap.add_argument("--num-classes", type=int, default=36)
    ap.add_argument("--out", type=Path, default=Path("primnet_out"))
    args = ap.parse_args()

    names = dict(DEFAULT_NAMES)
    if args.names:
        names.update({int(k): v for k, v in json.loads(args.names.read_text()).items()})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location=device)
    nc = ck.get("args", {}).get("num_classes", args.num_classes)
    model = PrimitiveNet(num_classes=nc,
                         dim=ck.get("args", {}).get("dim", 256),
                         depth=ck.get("args", {}).get("depth", 4)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    W, H, prims = parse_plan(args.svg.read_text(errors="ignore"))
    if not prims:
        raise SystemExit("No primitives parsed from SVG.")
    feats = torch.tensor([p["feat"] for p in prims], dtype=torch.float32)[None].to(device)
    with torch.no_grad():
        classes = model(feats).argmax(-1)[0].cpu().tolist()

    diag = (W ** 2 + H ** 2) ** 0.5 or 1.0
    instances = group_instances(prims, classes, names, diag=diag)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.svg.stem
    (args.out / f"{stem}.json").write_text(json.dumps(
        {"width": W, "height": H, "instances": instances}, indent=1))
    overlay(W, H, prims, classes, names).save(args.out / f"{stem}_overlay.png")
    from collections import Counter
    print("instances by label:", dict(Counter(i["label"] for i in instances)))
    print(f"wrote {args.out}/{stem}.json (+ overlay)")


if __name__ == "__main__":
    main()
