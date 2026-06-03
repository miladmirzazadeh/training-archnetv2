"""Generate DXFs from Vitruev scenario batches (NO PNG) — local or RunPod.

Mirrors render_dataset.py exactly:  load_scenarios(render_batches/) -> engine
config dicts -> FloorPlan(cfg) -> write_dxf().  (The raw plan_*.json files are a
different, higher-level schema and are NOT valid FloorPlan input — load_scenarios
is the required converter.)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from generator.scenario_loader import load_scenarios     # scenario schema -> engine config
from generator.layout import FloorPlan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, help="configs root (contains render_batches/)")
    ap.add_argument("--batches", type=Path, help="render_batches dir directly (overrides --configs)")
    ap.add_argument("--out", required=True, type=Path, help="output dir for .dxf")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    batch_dir = a.batches or (a.configs / "render_batches" if a.configs else None)
    if not batch_dir or not Path(batch_dir).is_dir():
        print("ERROR: pass --batches <render_batches dir> or --configs <root with render_batches/>",
              file=sys.stderr); sys.exit(2)

    a.out.mkdir(parents=True, exist_ok=True)
    configs, warns = load_scenarios(str(batch_dir))
    print(f"loaded {len(configs)} configs ({len(warns)} conversion warnings)")
    if a.limit:
        configs = configs[:a.limit]

    ok = 0
    for i, cfg in enumerate(configs):
        pid = cfg.get("plan_id") or cfg.get("id") or f"plan_{i:05d}"
        try:
            FloorPlan(cfg).write_dxf(str(a.out / f"{pid}.dxf"))
            ok += 1
        except Exception as e:
            print("skip", pid, repr(e), file=sys.stderr)
        if ok and ok % 500 == 0:
            print(f"  …{ok} DXFs")
    print(f"wrote {ok}/{len(configs)} DXFs -> {a.out}")
    if ok == 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
