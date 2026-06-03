"""Streaming generate+parse (PARALLEL): scenario batches -> labeled primitive JSON.

For each plan:  FloorPlan -> write_dxf -> primitives -> subsample clutter -> JSON.
Uses a multiprocessing pool (one plan per task), so it scales with CPU cores
(~4x on a Kaggle notebook). DXFs go to a temp per worker (deleted) unless
--keep-dxf is given, in which case each plan's DXF is written there and kept.

  python -m primitivenet.make_dataset --configs <Vitruev>/configs --out prim_ds \
         [--keep-dxf dxf_dir] [--workers N] [--limit N]
"""
from __future__ import annotations
import argparse, atexit, collections, hashlib, json, os, random, sys, tempfile
import multiprocessing as mp
from pathlib import Path

from generator.scenario_loader import load_scenarios
from generator.layout import FloorPlan
from primitivenet.parse_dxf import primitives_of, CLASS_NAMES

_G: dict = {}   # per-worker state (set in _init)


def _init(out, keep_dxf, clutter_mult, clutter_min, val_frac):
    _G.update(out=Path(out), keep=(Path(keep_dxf) if keep_dxf else None),
              cm=clutter_mult, cmin=clutter_min, vf=val_frac)
    if _G["keep"] is None:
        fd, tmp = tempfile.mkstemp(suffix=".dxf"); os.close(fd)
        _G["tmp"] = tmp
        atexit.register(lambda: os.path.exists(tmp) and os.remove(tmp))


def _work(cfg):
    pid = cfg.get("plan_id") or cfg.get("id") or "plan"
    dxf = str(_G["keep"] / f"{pid}.dxf") if _G["keep"] else _G["tmp"]
    try:
        FloorPlan(cfg).write_dxf(dxf)
        W, H, prims = primitives_of(dxf)
    except Exception as e:
        return (pid, None, repr(e))
    if not prims:
        return (pid, None, "empty")
    rng = random.Random(pid)                       # deterministic per plan
    if _G["cm"] > 0:                               # optional clutter cap (default OFF)
        noncl = [p for p in prims if p[2] != 0]
        cl = [p for p in prims if p[2] == 0]
        cap = max(_G["cmin"], int(_G["cm"] * len(noncl)))
        if len(cl) > cap:
            cl = rng.sample(cl, cap)
        keep = noncl + cl
    else:
        keep = list(prims)                         # keep ALL primitives (train/infer consistency)
    rng.shuffle(keep)
    rec = [{"feat": f, "sem": lab} for (_t, f, lab, _g) in keep]
    split = "val" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100 < _G["vf"] * 100 else "train"
    (_G["out"] / split / f"{pid}.json").write_text(json.dumps(
        {"width": round(W, 2), "height": round(H, 2), "primitives": rec}))
    return (pid, collections.Counter(r["sem"] for r in rec), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, help="configs root (contains render_batches/)")
    ap.add_argument("--batches", type=Path, help="render_batches dir directly")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--clutter-mult", type=float, default=0.0,
                    help="0 = keep ALL primitives (default; matches hatch-heavy inference). "
                         ">0 caps clutter to mult x non-clutter per plan.")
    ap.add_argument("--clutter-min", type=int, default=80)
    ap.add_argument("--keep-dxf", type=Path, help="write+KEEP each DXF here")
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    a = ap.parse_args()

    batch_dir = a.batches or (a.configs / "render_batches" if a.configs else None)
    if not batch_dir or not Path(batch_dir).is_dir():
        print("ERROR: pass --batches <render_batches> or --configs <root>", file=sys.stderr); sys.exit(2)

    configs, warns = load_scenarios(str(batch_dir))
    print(f"loaded {len(configs)} configs ({len(warns)} conversion warnings)")
    if a.limit:
        configs = configs[:a.limit]

    (a.out / "train").mkdir(parents=True, exist_ok=True)
    (a.out / "val").mkdir(parents=True, exist_ok=True)
    if a.keep_dxf:
        a.keep_dxf.mkdir(parents=True, exist_ok=True)

    workers = a.workers or os.cpu_count() or 1
    print(f"generating {len(configs)} plans with {workers} workers…")
    hist: dict = {}; n_ok = 0; n_skip = 0
    init_args = (str(a.out), str(a.keep_dxf) if a.keep_dxf else None, a.clutter_mult, a.clutter_min, a.val_frac)
    with mp.Pool(workers, initializer=_init, initargs=init_args) as pool:
        for i, (pid, cnt, err) in enumerate(pool.imap_unordered(_work, configs, chunksize=8), 1):
            if cnt is None:
                n_skip += 1
                if n_skip <= 20:
                    print("skip", pid, err, file=sys.stderr)
            else:
                n_ok += 1
                for k, v in cnt.items():
                    hist[k] = hist.get(k, 0) + v
            if i % 500 == 0:
                print(f"  …{i}/{len(configs)}  (ok {n_ok}, skip {n_skip})")

    print(f"DONE: {n_ok} plans ({n_skip} skipped) -> {a.out}")
    print("class histogram:", {CLASS_NAMES[k]: v for k, v in sorted(hist.items())})
    if n_ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
