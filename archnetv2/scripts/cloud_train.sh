#!/usr/bin/env bash
# Turnkey ArchNetv2-openings training on a rented single-GPU box
# (RunPod / Vast.ai / Lambda). One powerful GPU = no DDP, so the AC-CBAM
# model trains correctly — unlike Kaggle's 2×T4.
#
# Prereqs: a Kaggle API token (to fetch the public FloorPlanCAD-YOLO
# dataset). Set these two env vars before running:
#
#   export KAGGLE_USERNAME=your_kaggle_username
#   export KAGGLE_KEY=your_kaggle_key
#
# Then run (from the repo root, or it will clone itself):
#
#   bash archnetv2/scripts/cloud_train.sh
#
# Tunables (env vars, with defaults):
#   EPOCHS=100  BATCH=16  IMGSZ=640  FRACTION=1.0  TIME=2.0  PATIENCE=30
#   BATCH=12 if you're on a 24 GB card (RTX 4090); 16 fits a 40 GB A100.
set -euo pipefail

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME (Kaggle API token)}"
: "${KAGGLE_KEY:?set KAGGLE_KEY (Kaggle API token)}"

EPOCHS="${EPOCHS:-100}"; BATCH="${BATCH:-16}"; IMGSZ="${IMGSZ:-640}"
FRACTION="${FRACTION:-1.0}"; TIME="${TIME:-2.0}"; PATIENCE="${PATIENCE:-30}"
WORK="${WORK:-$(pwd)}"

echo "== installing deps =="
pip install -q ultralytics==8.4.58 kaggle

echo "== kaggle creds =="
mkdir -p ~/.kaggle
printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "== code =="
if [ ! -d "$WORK/training-archnetv2" ]; then
  git clone https://github.com/miladmirzazadeh/training-archnetv2.git "$WORK/training-archnetv2"
fi
cd "$WORK/training-archnetv2"

echo "== dataset (FloorPlanCAD YOLO) =="
if [ ! -d "$WORK/data" ]; then
  kaggle datasets download -d samirshabani/architecture -p "$WORK/data" --unzip
fi
SRC=$(find "$WORK/data" -type d -name FloorPlanCAD_YOLOv8_Full | head -1)
echo "dataset src: $SRC"

echo "== convert -> door/window YOLO =="
python -m archnetv2.scripts.prepare_floorplancad_yolo \
  --src "$SRC" --dst "$WORK/openings_yolo"

echo "== train (single GPU, full arc2) =="
python -m archnetv2.scripts.kaggle_resume_train \
  --data "$WORK/openings_yolo/data.yaml" \
  --project "$WORK/runs" --name archnetv2_openings_full \
  --device 0 --cache ram --workers 8 \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
  --patience "$PATIENCE" --fraction "$FRACTION" --time-budget "$TIME"

echo
echo "=========================================================="
echo "DONE. Weights:"
echo "  $WORK/runs/archnetv2_openings_full/weights/best.pt"
echo "Download it from the pod's file browser / Jupyter, or scp it."
echo "=========================================================="
