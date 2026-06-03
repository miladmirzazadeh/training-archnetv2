"""Streaming generate+parse: scenario batches -> labeled primitive JSON.

For each plan:  FloorPlan -> write_dxf(TEMP) -> primitives -> JSON -> delete temp.
Never keeps the ~7 GB of DXFs on disk — only the small primitives (~1.5 GB raw /
~240 MB tar.gz for 10k). Run this LOCALLY (you have the generator + configs),
tar the output, upload to Kaggle, and train there.

  python -m primitivenet.make_dataset --configs <Vitruev>/configs --out prim_ds
  tar czf primitives.tar.gz -C prim_ds train val      # ~240 MB -> upload to Kaggle
"""
from __future__ import annotations
import argparse, json, os, random, sys, tempfile
from pathlib import Path

from generator.scenario_loader import load_scenarios
from generator.layout import FloorPlan
from primitivenet.parse_dxf import primitives_of, CLASS_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, help="configs root (contains render_batches/)")
    ap.add_argument("--batches", type=Path, help="render_batches dir directly")
    ap.add_argument("--out", required=True, type=Path, help="writes <out>/train and <out>/val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--clutter-mult", type=float, default=2.0)
    ap.add_argument("--clutter-min", type=int, default=80)
    a = ap.parse_args()

    batch_dir = a.batches or (a.configs / "render_batches" if a.configs else None)
    if not batch_dir or not Path(batch_dir).is_dir():
        print("ERROR: pass --batches <render_batches> or --configs <root>", file=sys.stderr); sys.exit(2)

    configs, warns = load_scenarios(str(batch_dir))
    print(f"loaded {len(configs)} configs ({len(warns)} conversion warnings)")
    if a.limit:
        configs = configs[:a.limit]

    rng = random.Random(0)
    (a.out / "train").mkdir(parents=True, exist_ok=True)
    (a.out / "val").mkdir(parents=True, exist_ok=True)
    hist: dict = {}; n_ok = n_prim = 0
    fd, tmp = tempfile.mkstemp(suffix=".dxf"); os.close(fd)
    try:
        for i, cfg in enumerate(configs):
            pid = cfg.get("plan_id") or cfg.get("id") or f"plan_{i:05d}"
            try:
                FloorPlan(cfg).write_dxf(tmp)
                W, H, prims = primitives_of(tmp)
            except Exception as e:
                print("skip", pid, repr(e), file=sys.stderr); continue
            if not prims:
                continue
            noncl = [p for p in prims if p[2] != 0]
            cl = [p for p in prims if p[2] == 0]
            cap = max(a.clutter_min, int(a.clutter_mult * len(noncl)))
            if len(cl) > cap:
                cl = rng.sample(cl, cap)
            keep = noncl + cl; rng.shuffle(keep)
            rec = [{"feat": f, "sem": lab} for (_t, f, lab, _g) in keep]
            for r in rec:
                hist[r["sem"]] = hist.get(r["sem"], 0) + 1
            split = "val" if rng.random() < a.val_frac else "train"
            (a.out / split / f"{pid}.json").write_text(json.dumps(
                {"width": round(W, 2), "height": round(H, 2), "primitives": rec}))
            n_ok += 1; n_prim += len(rec)
            if n_ok % 500 == 0:
                print(f"  …{n_ok} plans, {n_prim} prims")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"DONE: {n_ok} plans, {n_prim} primitives -> {a.out}")
    print("class histogram:", {CLASS_NAMES[k]: v for k, v in sorted(hist.items())})
    if n_ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
