# PrimitiveNet — lightweight per-primitive panoptic labeler (Option B)

A small set-transformer that labels **each vector primitive** (line / arc /
circle) of a CAD plan with a semantic class, then groups same-class
connected primitives into **instances** — producing the same *kind* of
output as DPSS (per-primitive masks + points + labels) without DPSS's
heavy two-stream + Mask2Former infra or 8-GPU training.

Trains on the **FloorPlanCAD SVG** (catwhisker), single GPU, **< 3 h**.

## Why this instead of the bbox detector
Bounding boxes can't represent a diagonal window or tell you *which lines*
are a door. PrimitiveNet labels the actual primitives, so the output is
the higher-level graph you hand GPT: `{instances:[{label, points, ...}]}`.

## Pipeline
```
parse_svg.py   FloorPlanCAD SVG -> per-primitive features + (semantic, instance)
dataset.py     parsed JSON -> padded feature-token batches
model.py       PrimitiveNet: set-transformer -> per-primitive class
train.py       per-primitive CE, val mIoU, time budget, resume
infer.py       classify -> group instances -> DPSS-style JSON + overlay
```

Feature per primitive (12-d): one-hot type (line/curve/round) + normalized
`[x0,y0,x1,y1,cx,cy]` + normalized length + sin/cos of angle.

## Run on RunPod (single GPU, <3 h)
```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...
bash primitivenet/cloud_train.sh          # download + probe + convert + train
```
Then inference on a held-out plan:
```bash
python -m primitivenet.infer \
    --weights runs/primitivenet/best.pt --svg some_test.svg \
    --names names.json --out primnet_out
```
`names.json` maps `{semantic_id: name}` (door/window/wall/...) — finalize it
from the `parse_svg probe` histogram. Training works on raw ids regardless;
names are only applied at inference for readable labels.

## Status
v1: per-primitive **semantic** classification + heuristic instance grouping
(same-class, shared endpoints). A learned instance head (contrastive /
query-based, like DPSS) is a future upgrade.
