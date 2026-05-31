# ArchNetv2 (openings-only)

Reimplementation of:

> Xu Z., Jha N., Mehadi S., Mandal M. **Multiscale object detection on
> complex architectural floor plans.** *Automation in Construction* 165
> (2024) 105486. DOI: [10.1016/j.autcon.2024.105486](https://doi.org/10.1016/j.autcon.2024.105486)

The paper's full model targets 13 classes (door, stairs, sink, toilet,
firebox, fridge, stove, dishwasher, bathtub, shower, dryer, washer,
window) and reports **93.5 % mAP@0.5** on a private CAFP dataset. This
folder ships a faithful 2-class (door + window) variant trained on the
public **CubiCasa5K** dataset.

The paper notes that wall detection is **out of scope** for ArchNetv2
(future work: semantic segmentation) — so walls remain handled by your
existing `wall_healer.py`.

## Architecture (matches Fig. 4 of the paper)

```
Input 640×640×3
  └─ Backbone (YOLOv8l): exports B3, B5, B7, B10   ← extra B3 vs. YOLOv8
  └─ MLP (neck) 6 stages:
       Stage 1 (C2f)         up(B10)+B7    → 512ch
       Stage 2 (C2f)         up(n1)+B5     → 256ch
       Stage 3 (AC-CBAM)     up(n2)+B3     → 128ch ── head N3 (stride 4)
       Stage 4 (AC-CBAM)     down(N3)+n2   → 256ch ── head N4 (stride 8)
       Stage 5 (AC-CBAM)     down(N4)+n1   → 512ch ── head N5 (stride 16)
       Stage 6 (AC-CBAM)     down(N5)+B10  → 512ch ── head N6 (stride 32)
  └─ Detect (4 heads) → DFL + classification → NMS
```

`AC_CBAM = C2f(n=3) → CBAM → C2f(n=1)` — see [`model.py`](model.py).

The four detection heads (stride 4/8/16/32) are the paper's main edge
for small objects like windows; standard YOLOv8 only uses 8/16/32.

## Files

| Path | Purpose |
| --- | --- |
| `cfg/archnetv2.yaml` | Ultralytics model spec (4-head topology). |
| `model.py` | `AC_CBAM` module + `build_archnetv2()` builder. |
| `scripts/prepare_cubicasa5k.py` | CubiCasa5K SVG → YOLO labels. |
| `scripts/expand_dataset.py` | Optional 8× offline aug (4 rotations × flip). |
| `train.py` | Training entrypoint with paper hyperparameters. |
| `predict.py` | Inference on PNG/JPG **or PDF** (or a directory); JSON + annotated PNGs. |
| `overlay.py` | Draw detections on a floor-plan image. |
| `notebooks/archnetv2_colab.ipynb` | End-to-end Colab/Runpod workflow. |
| `notebooks/archnetv2_kaggle.ipynb` | **Kaggle background training with auto-resume** (recommended — survives session closing). |
| `scripts/kaggle_resume_train.py` | Resume-aware trainer: restore run dir → train w/ time budget → push checkpoint. |

## End-to-end workflow

### Local (CPU): sanity-check the model builds

```bash
python -m archnetv2.model
# expect: "Detect strides: [4.0, 8.0, 16.0, 32.0]"
```

### Kaggle (free, background, recommended): train on CubiCasa5K

Use `notebooks/archnetv2_kaggle.ipynb`. Unlike Colab, Kaggle's
**Save & Run All (Commit)** runs detached for up to ~12 h — you can
close the browser. Because ArchNetv2 won't finish 500 epochs in one
12 h window, the notebook **auto-resumes across sessions**:

1. One-time setup (in the notebook's first markdown cell): enable GPU +
   Internet, add the `cubicasa5k` Kaggle dataset, create an empty
   `archnetv2-ckpt` dataset, and add your `KAGGLE_USERNAME` / `KAGGLE_KEY`
   as Secrets.
2. Edit the `CKPT = 'YOUR_USERNAME/archnetv2-ckpt'` line.
3. Click **Save Version → Save & Run All**. Repeat whenever the previous
   run finishes — each session trains ~11 h, then pushes the updated run
   dir back to `archnetv2-ckpt`, and the next session picks up from there.

`best.pt` / `last.pt` end up inside the `archnetv2-ckpt` dataset under
`runs/archnetv2/archnetv2_openings/weights/`; download from there.

### Colab/Runpod: train on CubiCasa5K

Open `notebooks/archnetv2_colab.ipynb`. The notebook handles:

1. installing ultralytics
2. cloning the vtruvian repo
3. downloading + unzipping CubiCasa5K (~9 GB)
4. converting SVG annotations to YOLO format (door + window only)
5. (optional) offline 8× augmentation
6. training with paper hyperparameters
7. running inference on your `openheimer_outputs/full_healed.png`
8. downloading the trained `best.pt`

### Local: label new plans using the trained checkpoint

Once you've downloaded `best.pt`, label a **PNG**:

```bash
python -m archnetv2.predict \
    --weights /path/to/best.pt \
    --source openheimer_outputs/full_healed.png \
    --json   openheimer_outputs/artifacts/openings_archnetv2.json \
    --save-vis
```

…or a **PDF** (every page is rasterized at `--dpi`, then labeled):

```bash
python -m archnetv2.predict \
    --weights /path/to/best.pt \
    --source openheimer_inputs/input.pdf \
    --dpi 250 \
    --json   openheimer_outputs/artifacts/openings_archnetv2.json \
    --save-vis
```

…or a **directory** containing any mix of images and PDFs (just point
`--source` at the folder). Annotated images land in
`runs/archnetv2/predict/`; rendered PDF pages are kept in
`runs/archnetv2/predict/rendered/`.

Then draw a clean overlay:

```bash
python -m archnetv2.overlay \
    --image openheimer_outputs/full_healed.png \
    --json  openheimer_outputs/artifacts/openings_archnetv2.json \
    --out   openheimer_outputs/artifacts/openings_archnetv2_overlay.png
```

The output JSON has `bbox_xyxy_px` and `bbox_cxcywh_norm` per detection.
It is **not** in the same shape as your existing `openings.json` (which
is wall-anchored in DXF coordinates) — converting pixel boxes back into
the wall-anchored format requires the rendering scale + wall geometry
used by `wall_healer.py`. Drop me a line if you want that bridge.

## Hyperparameters (from the paper)

| Parameter | Value |
| --- | --- |
| Input size | 640 × 640 × 3 |
| Batch | 4 |
| Epochs | 500 |
| Early-stop patience | 100 |
| Optimizer | SGD (Ultralytics default) |
| Augmentation | 4× rotation × H-flip (= 8×) |
| IoU threshold (eval) | 0.5 |

## Faithfulness notes / deviations

- **Backbone weights**: paper trains from scratch; `train.py` warm-starts
  the backbone from `yolov8l.pt` (shape-mismatched head layers are
  silently skipped). Pass `--no-pretrained` to match the paper exactly.
- **Parameter count**: my build reports ~47.7 M; the paper's Table 5
  reports 90.6 M for ArchNetv2 vs. 82.3 M for YOLOv8l, while
  Ultralytics' own YOLOv8l is 43.7 M. The paper appears to use a
  different counting convention; the *architecture* matches.
- **Rotation augmentation**: bake offline with `expand_dataset.py` to
  match the paper exactly (YOLOv8's runtime `degrees=` is continuous,
  not 90°-discretized).
- **Dataset**: CafF (paper) is private — we use CubiCasa5K instead.
  Distribution shift between SVG-rendered CubiCasa plans and your
  DXF→PNG renders will dominate the gap to paper-reported mAP. The
  fastest fix is to fine-tune on a small batch of your own annotated
  renders after the CubiCasa5K pretrain.
