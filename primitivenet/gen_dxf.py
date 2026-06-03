"""Generate DXFs from scenario configs (NO PNG) — runs on RunPod.

This drives YOUR generator. Confirm the two integration points marked TODO
against your repo (the import path of FloorPlan, and how a config is loaded).
Everything downstream (parse_dxf) only needs the DXFs this writes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Vitruev_synthdata: FloorPlan takes the plan config dict, then .write_dxf(path).
from generator.layout import FloorPlan          # plan = FloorPlan(cfg); plan.write_dxf(path)


def load_config(p: Path):
    cfg = json.loads(p.read_text())
    cfg.setdefault("plan_id", p.stem)           # stable id/seed (config uses "id")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, type=Path, help="dir of plan_*.json scenario configs")
    ap.add_argument("--out", required=True, type=Path, help="output dir for .dxf")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfgs = sorted(a.configs.glob("plan_*.json"))
    if a.limit:
        cfgs = cfgs[:a.limit]
    if not cfgs:
        print(f"ERROR: no plan_*.json under {a.configs}", file=sys.stderr); sys.exit(2)

    ok = 0
    for cp in cfgs:
        try:
            plan = FloorPlan(load_config(cp))
            plan.write_dxf(str(a.out / f"{cp.stem}.dxf"))
            ok += 1
        except Exception as e:                   # one bad config shouldn't kill the run
            print("skip", cp.name, repr(e), file=sys.stderr)
        if ok % 500 == 0 and ok:
            print(f"  …{ok} DXFs")
    print(f"wrote {ok}/{len(cfgs)} DXFs -> {a.out}")
    if ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
