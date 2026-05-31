"""Overlay ArchNetv2 detections (from predict.py JSON) on top of a
rendered floor-plan PNG. Useful for visual comparison with the existing
vtruvian door_window_agent output.

Usage:
    python -m archnetv2.overlay \\
        --image openheimer_outputs/full_healed.png \\
        --json  openheimer_outputs/artifacts/openings_archnetv2.json \\
        --out   openheimer_outputs/artifacts/openings_archnetv2_overlay.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"door": (0, 180, 0), "window": (0, 120, 255)}


def overlay(image: Path, det_json: Path, out: Path,
            min_conf: float = 0.0) -> None:
    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Helvetica", 14)
    except OSError:
        font = ImageFont.load_default()

    payload = json.loads(det_json.read_text())
    # predict.py writes a list; pick the entry matching `image` if multiple.
    record = next(
        (r for r in payload if Path(r["image"]).resolve() == image.resolve()),
        payload[0] if payload else {"detections": []},
    )

    n = 0
    for det in record.get("detections", []):
        if det["score"] < min_conf:
            continue
        x0, y0, x1, y1 = det["bbox_xyxy_px"]
        color = COLORS.get(det["class"], (255, 0, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{det['class']} {det['score']:.2f}"
        draw.text((x0 + 2, max(0, y0 - 16)), label, fill=color, font=font)
        n += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Drew {n} detections -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()
    overlay(args.image, args.json, args.out, args.min_conf)


if __name__ == "__main__":
    main()
