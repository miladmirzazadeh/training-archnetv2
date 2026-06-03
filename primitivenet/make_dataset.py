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
import argparse, atexit, collections, hashlib, json, os, random, sys, tempfile
import multiprocessing as mp
from pathlib import Path

from generator.scenario_loader import load_scenarios
from generator.layout import FloorPlan
from primitivenet.parse_dxf import primitives_of, CLASS_NAMES

_G: dict = {}


def _init(out, keep_dxf, scenarios, hatch_layers, no_hatch_frac, val_frac, seed):
    _G.update(out=Path(out), keep=(Path(keep_dxf) if keep_dxf else None),
              scen=(Path(scenarios) if scenarios else None),
              hl=list(hatch_layers) if hatch_layers else None,
              nhf=no_hatch_frac, vf=val_frac, seed=seed)
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
    if no_hatch:                                   # match render_dataset: clutter.hatch_walls=False
        cfg = dict(cfg); cfg["clutter"] = {**cfg.get("clutter", {}), "hatch_walls": False}
    dxf = str(_G["keep"] / f"{pid}.dxf") if _G["keep"] else _G["tmp"]
    try:
        FloorPlan(cfg).write_dxf(dxf)
        W, H, prims, regions = primitives_of(dxf, hatch_layers=_G["hl"])
    except Exception as e:
        return (pid, None, repr(e), no_hatch)
    if not prims:
        return (pid, None, "empty", no_hatch)
    rng = random.Random(pid); keep = list(prims); rng.shuffle(keep)
    rec = [{"feat": f, "sem": lab} for (_t, f, lab, _g) in keep]
    split = _split_for(pid, _G["seed"], _G["vf"])
    (_G["out"] / split / f"{pid}.json").write_text(json.dumps(
        {"width": round(W, 2), "height": round(H, 2), "no_hatch": no_hatch,
         "primitives": rec, "hatch_regions": [r.to_dict() for r in regions]}))
    if _G["scen"]:
        try:
            (_G["scen"] / f"{pid}.json").write_text(json.dumps(cfg))
        except Exception:
            pass
    return (pid, collections.Counter(r["sem"] for r in rec), None, no_hatch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, help="configs root (contains render_batches/)")
    ap.add_argument("--batches", type=Path, help="render_batches dir directly")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-dxf", type=Path, help="write+KEEP each DXF here")
    ap.add_argument("--scenarios", type=Path, help="write the engine config (scenario) here")
    ap.add_argument("--hatch-layers", nargs="+", default=["A-WALL-PATT"],
                    help="layers HatchDetector treats as hatch (synthetic: A-WALL-PATT)")
    ap.add_argument("--no-hatch-frac", type=float, default=0.1,
                    help="fraction of plans with NO wall hatch (matches render_dataset)")
    ap.add_argument("--val-frac", type=float, default=0.15,    # matches render_dataset
                    help="val fraction (deterministic MD5 split, matches render_dataset)")
    ap.add_argument("--seed", type=int, default=42,            # matches render_dataset
                    help="split/no-hatch seed (use 42 to match render_dataset exactly)")
    ap.add_argument("--limit", type=int)
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
    for d in (a.keep_dxf, a.scenarios):
        if d:
            d.mkdir(parents=True, exist_ok=True)

    workers = a.workers or os.cpu_count() or 1
    print(f"generating {len(configs)} plans with {workers} workers "
          f"(no-hatch {a.no_hatch_frac:.0%}, hatch_layers={a.hatch_layers})…")
    hist: dict = {}; n_ok = n_skip = n_nohatch = 0
    init_args = (str(a.out), str(a.keep_dxf) if a.keep_dxf else None,
                 str(a.scenarios) if a.scenarios else None, a.hatch_layers,
                 a.no_hatch_frac, a.val_frac, a.seed)
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
