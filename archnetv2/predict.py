"""Run a trained ArchNetv2-openings checkpoint on floor-plan images or
PDFs and write the detections to JSON (+ optional annotated images).

Accepts PNG/JPG, a PDF (each page is rasterized first), or a directory
of any mix of those.

Usage (single image):
    python -m archnetv2.predict --weights runs/archnetv2/train/weights/best.pt \\
                                --source openheimer_outputs/full_healed.png \\
                                --save-vis

Usage (PDF — every page is rendered and labeled):
    python -m archnetv2.predict --weights ... --source plan.pdf --dpi 250 \\
                                --json out/openings_archnetv2.json --save-vis

Usage (directory of images and/or PDFs):
    python -m archnetv2.predict --weights ... --source openheimer_outputs/ \\
                                --json openheimer_outputs/artifacts/openings_archnetv2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = {0: "door", 1: "window"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    """Rasterize every page of a PDF to PNG using PyMuPDF (same engine as
    the rest of vtruvian). Returns the list of generated image paths."""
    import fitz  # PyMuPDF — already a vtruvian dependency

    out_dir.mkdir(parents=True, exist_ok=True)
    z = dpi / 72.0
    paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            out = out_dir / f"{pdf_path.stem}_p{i}.png"
            pix.save(out)
            paths.append(out)
    return paths


def _expand_source(source: Path, render_dir: Path, dpi: int) -> list[Path]:
    """Turn a file/dir/PDF source into a flat list of image paths,
    rendering any PDFs encountered."""
    images: list[Path] = []
    if source.is_dir():
        candidates = sorted(source.iterdir())
    else:
        candidates = [source]
    for p in candidates:
        if p.suffix.lower() == ".pdf":
            images.extend(_render_pdf_pages(p, render_dir, dpi))
        elif p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return images


def detect(weights: Path, source: Path, imgsz: int = 640, conf: float = 0.25,
           iou: float = 0.5, device: str = "", save_vis: bool = False,
           project: str = "runs/archnetv2", name: str = "predict",
           dpi: int = 250) -> list[dict]:
    """Detect openings. `source` may be an image, a PDF, or a directory of
    either; PDFs are rasterized at `dpi` before detection."""
    model = YOLO(str(weights))

    # Render any PDFs to PNGs in a persistent folder so the JSON image paths
    # remain valid afterward (needed by overlay.py).
    render_dir = Path(project) / name / "rendered"
    image_paths = _expand_source(source, render_dir, dpi)
    if not image_paths:
        raise FileNotFoundError(
            f"No images or PDFs found at {source}. Supported: "
            f"{sorted(IMG_EXTS)} and .pdf"
        )
    results = model.predict(
        source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf,
        iou=iou, device=device or None, save=save_vis,
        project=project, name=name, exist_ok=True, verbose=False,
    )
    return _collect(results)


def _collect(results) -> list[dict]:
    out: list[dict] = []
    for r in results:
        img_path = Path(r.path)
        h, w = r.orig_shape
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, s in zip(xyxy, cls, scores):
                boxes.append({
                    "class": CLASS_NAMES.get(int(c), str(int(c))),
                    "score": float(s),
                    "bbox_xyxy_px": [float(x0), float(y0), float(x1), float(y1)],
                    "bbox_cxcywh_norm": [
                        float((x0 + x1) / 2 / w),
                        float((y0 + y1) / 2 / h),
                        float((x1 - x0) / w),
                        float((y1 - y0) / h),
                    ],
                })
        out.append({
            "image": str(img_path),
            "width": int(w),
            "height": int(h),
            "detections": boxes,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path,
                    help="Image file, PDF, or directory of either.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="")
    ap.add_argument("--dpi", type=int, default=250,
                    help="Rasterization DPI for PDF pages (default 250).")
    ap.add_argument("--save-vis", action="store_true",
                    help="Save annotated images via Ultralytics.")
    ap.add_argument("--json", type=Path, default=None,
                    help="Write detections to this JSON path.")
    args = ap.parse_args()

    results = detect(args.weights, args.source, args.imgsz, args.conf,
                     args.iou, args.device, args.save_vis, dpi=args.dpi)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Wrote {sum(len(r['detections']) for r in results)} detections "
              f"across {len(results)} images -> {args.json}")
    else:
        for r in results:
            print(f"{r['image']}: {len(r['detections'])} openings")
            for d in r["detections"]:
                x0, y0, x1, y1 = d["bbox_xyxy_px"]
                print(f"    {d['class']:6s} conf={d['score']:.2f} "
                      f"[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")


if __name__ == "__main__":
    main()
