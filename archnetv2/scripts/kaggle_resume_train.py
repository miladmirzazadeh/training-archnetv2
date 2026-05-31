"""Resume-aware ArchNetv2 training for Kaggle (or any capped session).

Strategy
--------
Kaggle batch runs ("Save & Run All") run detached for up to ~12 h, then
get hard-killed. To span that, we:

  1. Restore the previous run directory from a checkpoint Kaggle Dataset
     (if it exists) so Ultralytics can `resume=True`.
  2. Train with a wall-clock `--time-budget` < 12 h so training stops
     *gracefully* (last.pt flushed) with time to spare for the upload.
  3. Zip the run dir and push it as a new version of the checkpoint
     Dataset via the Kaggle API.

Run "Save & Run All" repeatedly; each session resumes where the last
left off until it reaches `--epochs` or early-stops.

Usage (inside a Kaggle notebook cell):
    python -m archnetv2.scripts.kaggle_resume_train \\
        --data /kaggle/working/cubicasa5k_yolo/data.yaml \\
        --ckpt-dataset <kaggle-username>/archnetv2-ckpt \\
        --time-budget 11.0 --epochs 500 --batch 4

If --ckpt-dataset is omitted (or the Kaggle API has no creds), training
still runs and the run dir is left in /kaggle/working for manual reuse.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from archnetv2.model import build_archnetv2


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def restore_checkpoint(ckpt_dataset: str | None, runs_root: Path) -> None:
    """Download + unzip the checkpoint Dataset into `runs_root` if present.

    Tries a mounted input first (/kaggle/input/<name>), then the Kaggle API.
    """
    if not ckpt_dataset:
        return
    runs_root.mkdir(parents=True, exist_ok=True)
    name = ckpt_dataset.split("/")[-1]

    mounted = Path("/kaggle/input") / name
    zip_candidates = list(mounted.glob("*.zip")) if mounted.exists() else []
    if zip_candidates:
        print(f"[ckpt] Found mounted checkpoint: {zip_candidates[0]}")
        shutil.unpack_archive(str(zip_candidates[0]), str(runs_root))
        return

    # Fall back to the API (needs creds + internet).
    print(f"[ckpt] Attempting API download of dataset '{ckpt_dataset}'...")
    rc = _run(["kaggle", "datasets", "download", "-d", ckpt_dataset,
               "-p", str(runs_root), "--unzip"])
    if rc != 0:
        print("[ckpt] No existing checkpoint (first run, or API unavailable).")


def upload_checkpoint(ckpt_dataset: str | None, run_dir: Path) -> None:
    """Zip `run_dir` and push it as a new version of the checkpoint Dataset."""
    if not ckpt_dataset:
        print("[ckpt] --ckpt-dataset not set; skipping upload. Run dir is at",
              run_dir)
        return
    staging = Path("/kaggle/working/_ckpt_upload")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    archive = staging / "run.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=run_dir.parent,
                        base_dir=run_dir.name)
    user, slug = ckpt_dataset.split("/")
    meta = staging / "dataset-metadata.json"
    meta.write_text(
        '{\n'
        f'  "title": "{slug}",\n'
        f'  "id": "{ckpt_dataset}",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    # `version` if it already exists, else `create`.
    rc = _run(["kaggle", "datasets", "version", "-p", str(staging),
               "-m", "auto-resume checkpoint", "--dir-mode", "zip"])
    if rc != 0:
        print("[ckpt] version failed (dataset may not exist yet); trying create.")
        _run(["kaggle", "datasets", "create", "-p", str(staging),
              "--dir-mode", "zip"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--time-budget", type=float, default=11.0,
                    help="Max wall-clock hours for THIS session (<12 on Kaggle).")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default="/kaggle/working/runs/archnetv2")
    ap.add_argument("--name", default="archnetv2_openings")
    ap.add_argument("--ckpt-dataset", default=None,
                    help="Kaggle dataset slug '<user>/<name>' for checkpoints.")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    project = Path(args.project)
    run_dir = project / args.name
    last_pt = run_dir / "weights" / "last.pt"

    # 1) Restore prior run dir (if any) so we can resume.
    restore_checkpoint(args.ckpt_dataset, project)

    # 2) Train — resume if a checkpoint was restored, else start fresh.
    common = dict(
        data=str(args.data), imgsz=args.imgsz, device=args.device,
        workers=args.workers, project=str(project), name=args.name,
        save=True, plots=True, exist_ok=True,
    )
    if last_pt.exists():
        from ultralytics import YOLO
        print(f"[train] Resuming from {last_pt}")
        model = YOLO(str(last_pt))
        # On resume Ultralytics reuses the original args (epochs/time/etc.).
        model.train(resume=True)
    else:
        print("[train] Fresh run.")
        model = build_archnetv2(nc=2, verbose=True)
        if not args.no_pretrained:
            try:
                model.load("yolov8l.pt")
                print("[train] Warm-started backbone from yolov8l.pt.")
            except Exception as e:
                print(f"[train] yolov8l.pt warm-start skipped: {e}")
        model.train(
            epochs=args.epochs, batch=args.batch, patience=args.patience,
            time=args.time_budget,
            fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.1, scale=0.5,
            mosaic=1.0, close_mosaic=10, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
            **common,
        )

    # 3) Persist the run dir back to the checkpoint Dataset.
    upload_checkpoint(args.ckpt_dataset, run_dir)

    if last_pt.exists():
        print(f"\n[done] Checkpoint at {last_pt}")
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            print(f"[done] Best weights at {best}")


if __name__ == "__main__":
    main()
