"""Convert a pre-made YOLOv8 FloorPlanCAD export (e.g. Samir Shabani's
"Architecture" dataset on Kaggle) into our 2-class openings dataset
(door=0, window=1), appending into the SAME directory as the CubiCasa5K
output so the two sources train together.

Why name-driven? The export ships a data.yaml whose `names:` order we
don't control. Instead of hardcoding indices, we read data.yaml and map
any class whose name contains "door" -> 0 and "window" -> 1, dropping the
other 26 furniture/fixture classes. The detected mapping is printed so the
first run is self-verifying.

The source export is typically FLAT (images/ + labels/ + data.yaml, no
train/val/test split), so we assign a deterministic split by hashing each
file stem.

Usage:
    python -m archnetv2.scripts.prepare_floorplancad_yolo \\
        --src /kaggle/input/architecture/FloorPlanCAD_YOLOv8_Full \\
        --dst /kaggle/working/openings_yolo \\
        --link-images
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

IMG_EXTS = (".png", ".jpg", ".jpeg")


def load_names(src: Path) -> dict[int, str]:
    """Read data.yaml and return {index: class_name}. Handles both list and
    dict `names:` forms."""
    yamls = list(src.rglob("data.yaml"))
    if not yamls:
        raise FileNotFoundError(f"No data.yaml found under {src}")
    data = yaml.safe_load(yamls[0].read_text())
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unexpected `names:` format in {yamls[0]}: {names!r}")


def build_remap(names: dict[int, str]) -> dict[int, int]:
    """Map source class index -> {0:door, 1:window}; omit everything else.

    Substring matching is safe for the FloorPlanCAD taxonomy: only door
    classes contain 'door' and only window classes contain 'window'.
    """
    remap: dict[int, int] = {}
    for idx, name in names.items():
        low = name.lower()
        if "door" in low:
            remap[idx] = 0
        elif "window" in low:
            remap[idx] = 1
    return remap


def split_of(stem: str, val_frac: float, test_frac: float) -> str:
    """Deterministic train/val/test assignment by hashing the file stem."""
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def find_image(label_path: Path) -> Path | None:
    """Given .../labels/<stem>.txt, find the parallel .../images/<stem>.<ext>."""
    parts = list(label_path.parts)
    try:
        i = len(parts) - 1 - parts[::-1].index("labels")
    except ValueError:
        return None
    parts[i] = "images"
    base = Path(*parts).with_suffix("")
    for ext in IMG_EXTS:
        cand = base.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def convert(src: Path, dst: Path, val_frac: float, test_frac: float,
            link_images: bool, prefix: str, limit: int | None) -> None:
    names = load_names(src)
    remap = build_remap(names)
    print(f"Loaded {len(names)} classes from data.yaml.")
    print("Door/window mapping detected:")
    for idx, tgt in sorted(remap.items()):
        print(f"  src[{idx}] {names[idx]!r} -> {'door' if tgt == 0 else 'window'}")
    if not remap:
        print("ERROR: no door/window classes found in data.yaml names — "
              "cannot proceed.", file=sys.stderr)
        sys.exit(2)

    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_files = sorted(src.rglob("labels/*.txt"))
    if limit:
        label_files = label_files[:limit]

    stats = {"kept": 0, "skipped_empty": 0, "skipped_noimg": 0, "boxes": 0}
    for lf in label_files:
        out_lines: list[str] = []
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls in remap:
                out_lines.append(f"{remap[cls]} {' '.join(parts[1:5])}")
        if not out_lines:
            stats["skipped_empty"] += 1
            continue
        img = find_image(lf)
        if img is None:
            stats["skipped_noimg"] += 1
            continue

        split = split_of(lf.stem, val_frac, test_frac)
        base = f"{prefix}{lf.stem}"
        dst_img = dst / "images" / split / f"{base}{img.suffix}"
        if link_images:
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(img.resolve())
        else:
            shutil.copy2(img, dst_img)
        (dst / "labels" / split / f"{base}.txt").write_text("\n".join(out_lines) + "\n")
        stats["kept"] += 1
        stats["boxes"] += len(out_lines)

    # Write data.yaml (idempotent; identical to the CubiCasa5K converter's).
    (dst / "data.yaml").write_text(
        "# Combined openings dataset (door + window)\n"
        f"path: {dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names: [door, window]\n"
    )

    print("\nDone (FloorPlanCAD YOLO).")
    print(f"  kept:          {stats['kept']:6d} images, {stats['boxes']} boxes")
    print(f"  skipped (no door/window): {stats['skipped_empty']:6d}")
    print(f"  skipped (no image):       {stats['skipped_noimg']:6d}")
    print(f"\nDataset config: {dst/'data.yaml'}")
    if stats["kept"] == 0:
        print("ERROR: 0 images kept — check --src path / dataset layout.",
              file=sys.stderr)
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Root of the YOLOv8 FloorPlanCAD export "
                         "(contains images/, labels/, data.yaml).")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Combined openings dataset dir (shared with CubiCasa5K).")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--link-images", action="store_true",
                    help="Symlink source PNGs instead of copying (Kaggle).")
    ap.add_argument("--prefix", default="fpc_",
                    help="Filename prefix to avoid collisions with CubiCasa5K.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    convert(args.src.resolve(), args.dst.resolve(), args.val_frac,
            args.test_frac, args.link_images, args.prefix, args.limit)


if __name__ == "__main__":
    main()
