#!/usr/bin/env bash
# Option B — PrimitiveNet (per-primitive panoptic labeler) on FloorPlanCAD
# (catwhisker SVG). Single GPU, designed for <3 h. Run on RunPod.
#
# Prereq: Kaggle API token to fetch the dataset:
#   export KAGGLE_USERNAME=...   export KAGGLE_KEY=...
# Then from the repo root:
#   bash primitivenet/cloud_train.sh
#
# Tunables (env): EPOCHS=60 BATCH=8 TIME=2.5 NUMCLASSES=36
set -euo pipefail

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
: "${KAGGLE_KEY:?set KAGGLE_KEY}"
EPOCHS="${EPOCHS:-60}"; BATCH="${BATCH:-8}"; TIME="${TIME:-2.5}"
NUMCLASSES="${NUMCLASSES:-36}"; WORK="${WORK:-$(pwd)}"

echo "== deps =="
pip install -q kaggle pillow numpy

echo "== kaggle creds =="
mkdir -p ~/.kaggle
printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "== download FloorPlanCAD (catwhisker SVG) =="
if [ ! -d "$WORK/fpc_raw" ]; then
  kaggle datasets download -d catwhisker/floorplancad-dataset -p "$WORK/fpc_raw" --unzip
fi
# separate train/test tarballs so each maps to the right split
mkdir -p "$WORK/fpc_train" "$WORK/fpc_test"
mv "$WORK/fpc_raw"/train-*.tar.xz "$WORK/fpc_train"/ 2>/dev/null || true
mv "$WORK/fpc_raw"/test-*.tar.xz  "$WORK/fpc_test"/  2>/dev/null || true

echo "== PROBE (confirm SVG format / semantic-id histogram) =="
python -m primitivenet.parse_svg probe --src "$WORK/fpc_train" --work "$WORK/ext_train" --sample 6

echo "== convert SVG -> per-primitive JSON =="
python -m primitivenet.parse_svg convert --src "$WORK/fpc_train" --work "$WORK/ext_train" --out "$WORK/fpc_json" --split train
python -m primitivenet.parse_svg convert --src "$WORK/fpc_test"  --work "$WORK/ext_test"  --out "$WORK/fpc_json" --split test

echo "== train PrimitiveNet =="
python -m primitivenet.train \
  --data "$WORK/fpc_json" --num-classes "$NUMCLASSES" \
  --epochs "$EPOCHS" --batch "$BATCH" --time-budget "$TIME" \
  --out "$WORK/runs/primitivenet"

echo
echo "DONE. Weights: $WORK/runs/primitivenet/best.pt"
echo "Infer:  python -m primitivenet.infer --weights $WORK/runs/primitivenet/best.pt --svg <a_test>.svg --names names.json"
