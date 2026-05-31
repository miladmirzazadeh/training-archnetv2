# floorplan-openings-detector

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

## Quick links

- **Architecture + full docs:** [`archnetv2/README.md`](archnetv2/README.md)
- **Kaggle background training (recommended):** [`archnetv2/notebooks/archnetv2_kaggle.ipynb`](archnetv2/notebooks/archnetv2_kaggle.ipynb)
- **Colab training:** [`archnetv2/notebooks/archnetv2_colab.ipynb`](archnetv2/notebooks/archnetv2_colab.ipynb)
- **Inference (PNG/PDF):** `python -m archnetv2.predict --weights best.pt --source plan.pdf`

## License

Code: MIT. The original paper and CubiCasa5K dataset carry their own
licenses (CC BY-NC and CC BY-SA-NC respectively) — review them before
commercial use.
