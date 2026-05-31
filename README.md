# training-archnetv2

A faithful reimplementation of **ArchNetv2** — the multiscale floor-plan
object detector from:

> Xu Z., Jha N., Mehadi S., Mandal M. **Multiscale object detection on
> complex architectural floor plans.** *Automation in Construction* 165
> (2024) 105486. DOI: [10.1016/j.autcon.2024.105486](https://doi.org/10.1016/j.autcon.2024.105486)

This repo is configured for the **openings** task — detecting **doors**
and **windows** — and trains on the public **CubiCasa5K** dataset.

```
pip install -r requirements.txt
python -m archnetv2.model          # sanity-check the architecture builds
```

The model is a YOLOv8l backbone with **4 detection heads** (strides
4/8/16/32) and **AC-CBAM** attention blocks in the neck — see
[`archnetv2/README.md`](archnetv2/README.md) for the full architecture
breakdown, training instructions, and the Kaggle auto-resume workflow.

## Two tracks

| Track | Task | Model | Data |
|---|---|---|---|
| **Openings** (`archnetv2/`) | door + window **detection** | ArchNetv2 (4-head YOLOv8 + AC-CBAM) | CubiCasa5K + FloorPlanCAD-YOLO |
| **Walls** (`wallseg/`) | wall **segmentation** | UNet (segmentation-models-pytorch) | FloorPlanCAD SVG release |

Walls are a separate task because bounding boxes can't represent wall
geometry — the ArchNetv2 paper itself excludes walls and points to
segmentation. Both tracks share the same PDF/PNG inference contract and
the same Kaggle auto-resume background-training pattern.

## Quick links

**Openings (doors/windows):**
- Architecture + full docs: [`archnetv2/README.md`](archnetv2/README.md)
- Kaggle background training: [`archnetv2/notebooks/archnetv2_kaggle.ipynb`](archnetv2/notebooks/archnetv2_kaggle.ipynb)
- Inference (PNG/PDF): `python -m archnetv2.predict --weights best.pt --source plan.pdf`

**Walls (segmentation):**
- Kaggle background training: [`wallseg/notebooks/wallseg_kaggle.ipynb`](wallseg/notebooks/wallseg_kaggle.ipynb)
- Inference (PNG/PDF): `python -m wallseg.predict --weights best.pt --source plan.pdf`

## License

Code: MIT. The original paper and CubiCasa5K dataset carry their own
licenses (CC BY-NC and CC BY-SA-NC respectively) — review them before
commercial use.
