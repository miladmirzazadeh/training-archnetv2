"""Run the floorseg UNet on a PNG/PDF → per-class mask + colored overlay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from floorseg.train import build_model, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES

PALETTE = np.array([[255, 255, 255], [90, 90, 90], [220, 30, 30],
                    [30, 90, 220], [60, 180, 75]], np.uint8)  # bg,wall,door,window,room


def render(source: Path, dpi: int):
    if source.suffix.lower() == ".pdf":
        import fitz
        z = dpi / 72.0; out = []
        with fitz.open(source) as doc:
            for i, pg in enumerate(doc):
                px = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
                out.append((f"{source.stem}_p{i}",
                            Image.frombytes("RGB", (px.width, px.height), px.samples)))
        return out
    return [(source.stem, Image.open(source).convert("RGB"))]


@torch.no_grad()
def predict_mask(model, img: Image.Image, imgsz, device):
    w, h = img.size
    x = np.asarray(img.convert("RGB").resize((imgsz, imgsz)), np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    pred = model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
    return np.asarray(Image.fromarray(pred).resize((w, h), Image.NEAREST))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("floorseg_out"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.weights, map_location=device)
    nc = ck.get("args", {}).get("classes", 4)
    model = build_model(nc, ck.get("args", {}).get("encoder", "resnet34")).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    a.out.mkdir(parents=True, exist_ok=True)
    for name, img in render(a.source, a.dpi):
        m = predict_mask(model, img, a.imgsz, device)
        Image.fromarray(m).save(a.out / f"{name}_mask.png")          # class indices
        color = PALETTE[m]                                            # colorized
        blend = (0.5 * np.asarray(img.convert("RGB")) + 0.5 * color).astype(np.uint8)
        Image.fromarray(blend).save(a.out / f"{name}_overlay.png")
        from collections import Counter
        c = Counter(m.flatten().tolist())
        print(name, {CLASS_NAMES[k]: int(v) for k, v in c.items() if k < len(CLASS_NAMES)})
    print(f"masks + overlays -> {a.out}")


if __name__ == "__main__":
    main()
