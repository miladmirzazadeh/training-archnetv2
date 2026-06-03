"""Build the PrimitiveNet dataset from the 10k synthetic scenarios (PARALLEL).

Per plan:  (maybe drop hatch) -> FloorPlan -> write_dxf -> HatchDetector strips
hatch into region tokens -> labeled primitives -> JSON. Multiprocessing pool, so
it scales with CPU cores (~4x on a Kaggle notebook).

Saves three things (all kept on disk for Kaggle datasets):
  --out        prim_ds/{train,val}/<pid>.json   primitives + hatch_regions (training data)
  --keep-dxf   dxf/<pid>.dxf                     the generated DXFs
  --scenarios  scenarios/<pid>.json              the engine config used (with no-hatch edits)

10% of plans (random, seeded) are generated with NO wall hatch (clutter.hatch_walls
=False) so the model sees hatch-free plans; those scenarios are saved edited.

  python -m primitivenet.make_dataset --configs <Vitruev>/configs \
      --out prim_ds --keep-dxf dxf --scenarios scenarios [--no-hatch-frac 0.1] [--limit N]
"""
from __future__ import annotations
import argparse, atexit, collections, hashlib, json, math, os, random, sys, tempfile
import multiprocessing as mp
from pathlib import Path

from generator.scenario_loader import load_scenarios
from generator.layout import FloorPlan
from primitivenet.parse_dxf import primitives_synth, CLASS_NAMES


def _inject_flaws(doc, rng, wall_frac, dup_frac, wall_layers=("A-WALL-FULL", "A-WALL-INTR")):
    """DXF-only flaws (labels stay correct): ~wall_frac of wall lines get a gap/overshoot
    (still labeled wall), ~dup_frac get a near-coincident duplicate (labeled 'duplicate'
    so the agent's overkill can delete it). Returns the duplicate handles."""
    msp = doc.modelspace(); wl = set(wall_layers); dup = set()
    walls = [e for e in msp if getattr(e.dxf, "layer", "") in wl and e.dxftype() in ("LINE", "LWPOLYLINE")]
    for e in walls:
        if rng.random() < wall_frac:                       # detach / overshoot an endpoint
            try:
                if e.dxftype() == "LINE":
                    a, b = e.dxf.start, e.dxf.end
                    L = math.hypot(b.x - a.x, b.y - a.y) or 1.0
                    d = rng.uniform(30, 150) * (1 if rng.random() < 0.5 else -1)
                    e.dxf.end = (b.x + (b.x - a.x) / L * d, b.y + (b.y - a.y) / L * d, 0)
                else:
                    pts = e.get_points()
                    if len(pts) >= 2:
                        i = rng.randrange(len(pts)); v = list(pts[i])
                        v[0] += rng.uniform(-120, 120); v[1] += rng.uniform(-120, 120)
                        pts[i] = tuple(v); e.set_points(pts)
            except Exception:
                pass
        if rng.random() < dup_frac:                        # overkill: near-coincident duplicate
            try:
                c = e.copy()
                try: c.translate(rng.uniform(-15, 15), rng.uniform(-15, 15), 0)
                except Exception: pass
                msp.add_entity(c); dup.add(c.dxf.handle)
            except Exception:
                pass
    return dup

_G: dict = {}


def _init(out, keep_dxf, scenarios, no_hatch_frac, val_frac, seed, flaw_frac, dup_frac):
    _G.update(out=Path(out), keep=(Path(keep_dxf) if keep_dxf else None),
              scen=(Path(scenarios) if scenarios else None),
              nhf=no_hatch_frac, vf=val_frac, seed=seed,
              fw=flaw_frac, fd=dup_frac)
    if _G["keep"] is None:
        fd, tmp = tempfile.mkstemp(suffix=".dxf"); os.close(fd)
        _G["tmp"] = tmp
        atexit.register(lambda: os.path.exists(tmp) and os.remove(tmp))


def _split_for(pid, seed, val_frac):              # matches render_dataset.split_for
    h = int(hashlib.md5(f"{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def _no_hatch_for(pid, seed, frac):               # matches render_dataset.no_hatch_for
    h = int(hashlib.md5(f"nohatch:{seed}:{pid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def _work(cfg):
    pid = cfg.get("plan_id") or cfg.get("id") or "plan"
    no_hatch = _no_hatch_for(pid, _G["seed"], _G["nhf"])
    # ALWAYS render hatch-free (fast). The hatch is added as a LABEL token instead,
    # except for the no-hatch 10% which get no hatch token at all.
    cfg = dict(cfg); cfg["clutter"] = {**cfg.get("clutter", {}), "hatch_walls": False}
    dxf = str(_G["keep"] / f"{pid}.dxf") if _G["keep"] else _G["tmp"]
    rng = random.Random(pid)
    try:
        plan = FloorPlan(cfg)
        sub_map = {str(c.id): getattr(c, "subtype", None) for c in getattr(plan, "opening_components", [])}
        doc = plan.write_dxf(dxf)
        dup_handles = _inject_flaws(doc, rng, _G["fw"], _G["fd"])    # flaws live ONLY in the DXF
        doc.saveas(dxf)
        W, H, prims = primitives_synth(dxf, dup_handles, add_hatch=not no_hatch, sub_map=sub_map)
    except Exception as e:
        return (pid, None, repr(e), no_hatch)
    if not prims:
        return (pid, None, "empty", no_hatch)
    keep = list(prims); rng.shuffle(keep)
    rec = [{"feat": f, "sem": c, "sub": s} for (_t, f, c, s, _g) in keep]
    split = _split_for(pid, _G["seed"], _G["vf"])
    (_G["out"] / split / f"{pid}.json").write_text(json.dumps(
        {"width": round(W, 2), "height": round(H, 2), "no_hatch": no_hatch, "primitives": rec}))
    if _G["scen"]:
        try:
            (_G["scen"] / f"{pid}.json").write_text(json.dumps(cfg))   # clean scenario (no flaws)
        except Exception:
            pass
    return (pid, collections.Counter(r["sem"] for r in rec), None, no_hatch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, help="configs root (contains render_batches/)")
    ap.add_argument("--batches", type=Path, help="render_batches dir directly")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-dxf", type=Path, help="write+KEEP each DXF here (flawed, hatch-free)")
    ap.add_argument("--scenarios", type=Path, help="write the engine config (clean scenario) here")
    ap.add_argument("--no-hatch-frac", type=float, default=0.1,
                    help="fraction of plans with NO hatch label (the rest get a hatch region token)")
    ap.add_argument("--flaw-frac", type=float, default=0.05,
                    help="fraction of wall lines given a gap/overshoot in the DXF (label stays wall)")
    ap.add_argument("--dup-frac", type=float, default=0.05,
                    help="fraction of wall lines given a duplicate in the DXF (labeled 'duplicate')")
    ap.add_argument("--val-frac", type=float, default=0.15,    # matches render_dataset
                    help="val fraction (deterministic MD5 split, matches render_dataset)")
    ap.add_argument("--seed", type=int, default=42,            # matches render_dataset
                    help="split/no-hatch seed (use 42 to match render_dataset exactly)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="run N parallel notebooks, each a shard")
    ap.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    a = ap.parse_args()

    batch_dir = a.batches or (a.configs / "render_batches" if a.configs else None)
    if not batch_dir or not Path(batch_dir).is_dir():
        print("ERROR: pass --batches <render_batches> or --configs <root>", file=sys.stderr); sys.exit(2)

    configs, warns = load_scenarios(str(batch_dir))
    print(f"loaded {len(configs)} configs ({len(warns)} conversion warnings)")
    if a.limit:
        configs = configs[:a.limit]
    if a.num_shards > 1:                           # split across parallel notebooks
        configs = configs[a.shard::a.num_shards]
        print(f"shard {a.shard}/{a.num_shards}: {len(configs)} plans")

    (a.out / "train").mkdir(parents=True, exist_ok=True)
    (a.out / "val").mkdir(parents=True, exist_ok=True)
    for d in (a.keep_dxf, a.scenarios):
        if d:
            d.mkdir(parents=True, exist_ok=True)

    workers = a.workers or os.cpu_count() or 1
    print(f"generating {len(configs)} plans with {workers} workers "
          f"(hatch-free DXF; no-hatch-label {a.no_hatch_frac:.0%}; "
          f"flaws {a.flaw_frac:.0%}, dups {a.dup_frac:.0%})…")
    hist: dict = {}; n_ok = n_skip = n_nohatch = 0
    init_args = (str(a.out), str(a.keep_dxf) if a.keep_dxf else None,
                 str(a.scenarios) if a.scenarios else None,
                 a.no_hatch_frac, a.val_frac, a.seed, a.flaw_frac, a.dup_frac)
    with mp.Pool(workers, initializer=_init, initargs=init_args, maxtasksperchild=200) as pool:
        for i, (pid, cnt, err, nh) in enumerate(pool.imap_unordered(_work, configs, chunksize=8), 1):
            if cnt is None:
                n_skip += 1
                if n_skip <= 20:
                    print("skip", pid, err, file=sys.stderr)
            else:
                n_ok += 1; n_nohatch += int(nh)
                for k, v in cnt.items():
                    hist[k] = hist.get(k, 0) + v
            if i % 500 == 0:
                print(f"  …{i}/{len(configs)}  (ok {n_ok}, skip {n_skip}, no-hatch {n_nohatch})")

    print(f"DONE: {n_ok} plans ({n_skip} skipped, {n_nohatch} no-hatch) -> {a.out}")
    print("class histogram:", {CLASS_NAMES[k]: v for k, v in sorted(hist.items())})
    if n_ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
