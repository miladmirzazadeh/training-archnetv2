"""Run PrimitiveNet on a DXF -> per-primitive coarse class + subtype, with a preview.

  python -m primitivenet.infer_dxf --weights best.pt --dxf plan.dxf --out pred.png

Real hatch in the input is handled by HatchDetector (collapsed to region tokens);
the rest are classified. Output: printed class/subtype counts + a colored preview,
plus a labeled DXF (entities on layers PN-WALL / PN-DOOR / ... ) if --dxf-out given.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from primitivenet.model import PrimitiveNet
from primitivenet.parse_dxf import primitives_of, CLASS_NAMES, SUBTYPE_NAMES

COLOR = {0: "0.75", 1: "black", 2: "red", 3: "blue", 4: "green", 5: "orange", 6: "magenta"}
LW = {1: 2.2, 2: 2.0, 3: 2.0, 4: 1.6, 6: 1.4}   # wall/door/window/column/duplicate


def load_model(weights):
    ck = torch.load(weights, map_location="cpu")
    a = ck.get("args", {})
    m = PrimitiveNet(num_classes=int(a.get("num_classes", 7)),
                     num_subtypes=int(a.get("num_subtypes", 18)),
                     feat_dim=int(a.get("feat_dim", 18)),
                     dim=int(a.get("dim", 256)), depth=int(a.get("depth", 4)))
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def infer(model, dxf):
    W, H, prims, regions = primitives_of(dxf)          # HatchDetector handles real hatch
    if not prims:
        return None
    feats = torch.tensor([p[1] for p in prims], dtype=torch.float32)[None]
    lc, ls = model(feats)
    pc = lc.argmax(-1)[0].tolist(); ps = ls.argmax(-1)[0].tolist()
    return W, H, prims, pc, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dxf", required=True)
    ap.add_argument("--out", default="pred.png")
    ap.add_argument("--dxf-out", help="write a labeled DXF (PN-<CLASS> layers)")
    a = ap.parse_args()

    model = load_model(a.weights)
    res = infer(model, a.dxf)
    if res is None:
        print("no primitives in", a.dxf); return
    W, H, prims, pc, ps = res

    print("coarse:", {CLASS_NAMES[k]: v for k, v in sorted(Counter(pc).items())})
    dw = Counter((CLASS_NAMES[c], SUBTYPE_NAMES[s]) for (p, c, s) in zip(prims, pc, ps) if c in (2, 3))
    print("door/window subtypes:", dict(dw))
    walls = Counter(SUBTYPE_NAMES[s] for (p, c, s) in zip(prims, pc, ps) if c == 1)
    print("wall int/ext:", dict(walls))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 10))
    for (t, feat, _c, _s, g), c in zip(prims, pc):
        if c == 0:
            continue
        col = COLOR.get(c, "0.6")
        if t == 4:                                       # hatch region polygon
            poly = np.array(g, float).reshape(-1, 2)
            ax.fill(poly[:, 0], poly[:, 1], color="orange", alpha=0.12)
        else:
            ax.plot([g[0], g[2]], [g[1], g[3]], color=col, lw=LW.get(c, 1.0))
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("PrimitiveNet: wall=blk door=red window=blue column=grn hatch=org dup=mag")
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print("preview ->", a.out)

    if a.dxf_out:
        import ezdxf
        doc = ezdxf.new(); msp = doc.modelspace()
        for c in range(1, len(CLASS_NAMES)):
            doc.layers.add(f"PN-{CLASS_NAMES[c].upper()}", color=(0 if c == 1 else c))
        for (t, feat, _c, _s, g), c in zip(prims, pc):
            if c == 0 or t == 4:
                continue
            msp.add_line((g[0], g[1]), (g[2], g[3]), dxfattribs={"layer": f"PN-{CLASS_NAMES[c].upper()}"})
        doc.saveas(a.dxf_out); print("labeled DXF ->", a.dxf_out)


if __name__ == "__main__":
    main()
