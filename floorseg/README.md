# floorseg — Stage 2: 4-class floor-plan segmentation (the locked workflow)

Trains a **multi-class semantic segmentation UNet** (`background / wall /
door / window` [+ `room`]) on your **synthetic dataset**, then feeds labels
into vtruvian's existing vector→DXF machinery.

This is the model in the agreed full workflow:
```
PNG ─┬─► floorseg UNet ──► per-class masks            (semantics)
     └─► vtruvian line extractor (_lines.json) ──► vector lines  (geometry)
              └─ label_lines.py: tag each line by the mask ─► labeled vectors
                       └─► vtruvian structuring + ezdxf ─► DXF
```

## Files
| File | Purpose |
|---|---|
| `synth_to_masks.py` | synthetic `configs` (walls, mm) + `rich_json` (openings, px) → per-class mask PNGs. Solves the mm→px affine per plan from the rich_json's own mm↔px point pairs (self-validating). |
| `train.py` | 4-class UNet (segmentation-models-pytorch), CE w/ bg down-weight, val mIoU, resume + time budget. Internal train/val split by hashing filenames. |
| `infer.py` | PNG/PDF → class mask + colored overlay. |
| `label_lines.py` | overlay vtruvian `_lines.json` on a predicted mask → label every line wall/door/window (the vtruvian bridge). |
| `cloud_train.sh` | RunPod one-shot: masks → train. |

## Train on RunPod
Provide the rendered synthetic dataset (`images/`, `rich_json/`) + the 10k `configs/`:
```bash
SYNTH=/workspace/synth CONFIGS=/workspace/synth/configs bash floorseg/cloud_train.sh
# best.pt → runs/floorseg/best.pt ; download via: runpodctl send runs/floorseg/best.pt
```
Light model → typically **well under the 2.5 h budget** on one GPU.

## Use it (end to end)
```bash
# 1) masks + overlay on a real plan
python -m floorseg.infer --weights best.pt --source plan.pdf --dpi 200 --out out
# 2) label vtruvian's extracted lines using the mask
python -m floorseg.label_lines --lines plan_lines.json --mask out/plan_p0_mask.png \
       --out plan_lines_labeled.json
# 3) feed plan_lines_labeled.json into vtruvian's DXF assembly
```

## Why segmentation (not DPSS / bbox)
- Pixel masks are robust for *recognition*; your **clean extracted lines** provide *geometry* — so mask noise never hurts coordinates.
- Trains on **your in-domain synthetic data** (the fix for the domain gap that broke every public model on your plans).
- Light, single-GPU, reuses Ultralytics-free `smp` UNet — no DPSS install/compute.
