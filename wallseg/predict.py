"""Run a trained wall-segmentation UNet on a PNG or PDF floor plan and
write the predicted wall mask + a colored overlay.

PDFs are rasterized per page via PyMuPDF (same engine as the openings
predictor), so the two share an input contract.

Usage:
    python -m wallseg.predict --weights best.pt --source plan.pdf --dpi 250 \\
        --out-dir wall_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wallseg.train import IMAGENET_MEAN, IMAGENET_STD, build_model

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    z = dpi / 72.0
    paths = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            p = out_dir / f"{pdf_path.stem}_p{i}.png"
            pix.save(p)
            paths.append(p)
    return paths


def _sources(source: Path, render_dir: Path, dpi: int) -> list[Path]:
    cands = sorted(source.iterdir()) if source.is_dir() else [source]
    imgs = []
    for p in cands:
        if p.suffix.lower() == ".pdf":
            imgs += _render_pdf_pages(p, render_dir, dpi)
        elif p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs


@torch.no_grad()
def infer(model, img: Image.Image, imgsz: int, device: str, thr: float) -> np.ndarray:
    """Return a full-resolution binary wall mask for the input image."""
    w, h = img.size
    x = np.asarray(img.convert("RGB").resize((imgsz, imgsz)), np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (prob > thr).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))


def overlay(img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(img.convert("RGB"), np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    m = (mask > 0)[..., None]
    out = np.where(m, (1 - alpha) * base + alpha * red, base)
    return Image.fromarray(out.astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("wall_out"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--encoder", default="resnet34")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.weights, map_location=device)
    enc = ck.get("args", {}).get("encoder", args.encoder)
    model = build_model(enc).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = _sources(args.source, args.out_dir / "rendered", args.dpi)
    if not images:
        raise FileNotFoundError(f"No images/PDFs at {args.source}")
    for p in images:
        img = Image.open(p)
        mask = infer(model, img, args.imgsz, device, args.thr)
        Image.fromarray(mask).save(args.out_dir / f"{p.stem}_wallmask.png")
        overlay(img, mask).save(args.out_dir / f"{p.stem}_overlay.png")
        print(f"{p.name}: wall px = {(mask > 0).sum()}")
    print(f"Wrote masks + overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
