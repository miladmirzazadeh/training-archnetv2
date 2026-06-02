#!/usr/bin/env bash
# floorseg (Stage 2) — train the 4-class segmentation UNet on your synthetic
# dataset, on a RunPod GPU. Single GPU, <3 h.
#
# Expects a RENDERED synthetic dataset (you provide it) with:
#   $SYNTH/images/...           rendered PNGs (any subdirs; searched recursively)
#   $SYNTH/rich_json/*_rich.json
#   $CONFIGS/plan_*.json        the 10k configs (walls geometry in mm)
#
# Usage (from repo root):
#   SYNTH=/workspace/synth CONFIGS=/workspace/synth/configs bash floorseg/cloud_train.sh
#
# Tunables: EPOCHS=80 BATCH=8 IMGSZ=640 TIME=2.5 CLASSES=4 ROOMS=0
set -euo pipefail

: "${SYNTH:?set SYNTH=/path/to/rendered/synthetic (images/ + rich_json/)}"
CONFIGS="${CONFIGS:-$SYNTH/configs}"
EPOCHS="${EPOCHS:-80}"; BATCH="${BATCH:-8}"; IMGSZ="${IMGSZ:-640}"
TIME="${TIME:-2.5}"; CLASSES="${CLASSES:-4}"; ROOMS="${ROOMS:-0}"
WORK="${WORK:-$(pwd)}"

echo "== deps =="
pip install -q segmentation-models-pytorch pillow numpy PyMuPDF

echo "== synthetic → per-class masks =="
ROOMFLAG=""; [ "$ROOMS" = "1" ] && ROOMFLAG="--rooms"
python -m floorseg.synth_to_masks \
  --configs "$CONFIGS" --rich "$SYNTH/rich_json" --images "$SYNTH/images" \
  --out "$WORK/masks" $ROOMFLAG

echo "== train 4-class segmentation UNet =="
python -m floorseg.train \
  --images "$SYNTH/images" --masks "$WORK/masks" --val-frac 0.1 \
  --classes "$CLASSES" --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
  --time-budget "$TIME" --out "$WORK/runs/floorseg"

echo
echo "DONE. Weights: $WORK/runs/floorseg/best.pt"
echo "Download with: runpodctl send $WORK/runs/floorseg/best.pt"
