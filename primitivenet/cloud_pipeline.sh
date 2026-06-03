#!/usr/bin/env bash
# PrimitiveNet end-to-end on RunPod — everything on the pod, nothing big from your Mac.
#
#   configs (Kaggle) -> [generator] DXF -> [parse_dxf] labeled primitives (JSON)
#                    -> push tiny dataset to Kaggle -> train PrimitiveNet -> push best.pt
#
# Prereqs (export before running):
#   export KAGGLE_API_TOKEN=KGAT_...           # your token
#   export GEN_REPO=https://github.com/<you>/<generator>.git   # repo with FloorPlan.write_dxf
# Optional:
#   CONFIGS_DS=miladmirzazadeh/floorseg-jsons  # dataset that holds configs/  (default)
#   PRIM_DS=miladmirzazadeh/floorplan-primitives  CKPT_DS=miladmirzazadeh/primnet-ckpt
#   LIMIT=  EPOCHS=80 BATCH=16 TIME=3.0 OPENW=2.0 LAYER_MAP=
set -euo pipefail
: "${KAGGLE_API_TOKEN:?export KAGGLE_API_TOKEN}"
: "${GEN_REPO:?export GEN_REPO=<your generator git url>}"
CONFIGS_DS="${CONFIGS_DS:-miladmirzazadeh/floorseg-jsons}"
PRIM_DS="${PRIM_DS:-miladmirzazadeh/floorplan-primitives}"
CKPT_DS="${CKPT_DS:-miladmirzazadeh/primnet-ckpt}"
EPOCHS="${EPOCHS:-80}"; BATCH="${BATCH:-16}"; TIME="${TIME:-3.0}"; OPENW="${OPENW:-2.0}"
W="${WORK:-/workspace}"; LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"

echo "== deps =="
pip install -q kaggle ezdxf torch numpy

echo "== 1. configs from Kaggle =="
kaggle datasets download -d "$CONFIGS_DS" -p "$W/cfg" --unzip
CFG_DIR="$(dirname "$(find "$W/cfg" -name 'plan_*.json' | head -1)")"
echo "   configs: $(ls "$CFG_DIR"/plan_*.json | wc -l) in $CFG_DIR"

echo "== 2. clone generator + this repo =="
[ -d "$W/gen" ] || git clone "$GEN_REPO" "$W/gen"
pip install -q -e "$W/gen" 2>/dev/null || pip install -q -r "$W/gen/requirements.txt" 2>/dev/null || true
# this primitivenet package (so -m primitivenet.* works):
[ -d "$W/fod/primitivenet" ] || \
  git clone https://github.com/miladmirzazadeh/training-archnetv2.git "$W/fod" || true
PKG_ROOT="$W/fod"; [ -d "$PKG_ROOT/primitivenet" ] || PKG_ROOT="$(pwd)"

echo "== 3. generate DXFs (configs -> DXF, on the pod) =="
( cd "$W/gen" && PYTHONPATH="$W/gen:$PKG_ROOT" \
    python "$PKG_ROOT/primitivenet/gen_dxf.py" --configs "$CFG_DIR" --out "$W/dxf" $LIM )

echo "== 4. PROBE layers/blocks (sanity — confirm the label map matches) =="
( cd "$PKG_ROOT" && python -m primitivenet.parse_dxf probe --dxf-dir "$W/dxf" --sample 6 )

echo "== 5. DXF -> labeled primitive JSON =="
LM=""; [ -n "${LAYER_MAP:-}" ] && LM="--layer-map $LAYER_MAP"
( cd "$PKG_ROOT" && python -m primitivenet.parse_dxf convert --dxf-dir "$W/dxf" --out "$W/prim" --split train $LM )
# carve a 10% val split
python - "$W/prim" <<'PY'
import sys, random, shutil, pathlib
root = pathlib.Path(sys.argv[1]); tr = root/"train"; va = root/"val"; va.mkdir(exist_ok=True)
files = sorted(tr.glob("*.json")); random.Random(0).shuffle(files)
for f in files[:max(1, len(files)//10)]: shutil.move(str(f), va/f.name)
print("split: train", len(list(tr.glob('*.json'))), "val", len(list(va.glob('*.json'))))
PY

echo "== 6. push tiny primitive dataset to Kaggle =="
STG="$W/_prim_push"; rm -rf "$STG"; mkdir -p "$STG"
( cd "$W/prim" && tar czf "$STG/primitives.tar.gz" train val )
printf '{"title":"%s","id":"%s","licenses":[{"name":"CC0-1.0"}]}\n' "${PRIM_DS##*/}" "$PRIM_DS" > "$STG/dataset-metadata.json"
kaggle datasets create -p "$STG" 2>/dev/null || kaggle datasets version -p "$STG" -m "primitives $(ls "$W/dxf"|wc -l) plans"

echo "== 7. train PrimitiveNet (5 classes: clutter/wall/door/window/column) =="
( cd "$PKG_ROOT" && python -m primitivenet.train \
    --data "$W/prim" --num-classes 5 --epochs "$EPOCHS" --batch "$BATCH" \
    --open-weight "$OPENW" --time-budget "$TIME" --out "$W/runs/primnet" )

echo "== 8. push best.pt to Kaggle =="
CK="$W/_ckpt"; rm -rf "$CK"; mkdir -p "$CK"
cp "$W/runs/primnet/best.pt" "$W/runs/primnet/last.pt" "$CK"/ 2>/dev/null || true
printf '{"title":"%s","id":"%s","licenses":[{"name":"CC0-1.0"}]}\n' "${CKPT_DS##*/}" "$CKPT_DS" > "$CK/dataset-metadata.json"
kaggle datasets create -p "$CK" 2>/dev/null || kaggle datasets version -p "$CK" -m "primnet best"

echo "DONE. primitives -> $PRIM_DS ; weights -> $CKPT_DS"
