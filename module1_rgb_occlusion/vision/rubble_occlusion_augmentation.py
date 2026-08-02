"""
rubble_occlusion_augmentation.py

Module 1 - step 3: synthetic rubble/debris occlusion augmentation.

Takes person images (with YOLO-format labels) and pastes procedurally
generated rubble/debris textures over the person at controlled occlusion
levels (default 25/50/75%), to build an occlusion-robust training set for
YOLO11n / YOLO26n fine-tuning.

Why procedural textures instead of scraped images:
- No real "rubble occlusion" dataset exists off the shelf.
- Procedural = reproducible, license-free, and you can dial difficulty
  (curriculum learning) by tuning texture roughness / occlusion %.

YOLO label format assumed (one .txt per image, same basename):
    class_id x_center y_center width height      (all normalized 0-1)

Occlusion does NOT change the bbox (the person's true extent is unchanged,
they're just partially hidden) so labels are copied as-is. What changes is
logged separately in occlusion_log.csv as the ACTUAL measured occlusion %
(mask area / bbox area) -- use this column, not the nominal target level,
when you stratify mAP/precision/recall later. Nominal and actual will
differ a bit because blobs are randomized.

Usage:
    python rubble_occlusion_augmentation.py \
        --images_dir data/images --labels_dir data/labels \
        --out_dir data/occluded --levels 25 50 75

Demo mode (no real dataset yet, just to sanity check the pipeline):
    python rubble_occlusion_augmentation.py --demo --out_dir demo_out
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 1. Procedural rubble/debris texture generator
# ----------------------------------------------------------------------
def generate_rubble_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    Builds a BGR debris-like texture of size (h, w) by layering:
      a) multi-octave value noise for a mottled concrete/dirt base
      b) random dark irregular polygons for crack/shadow chunks
      c) random light/rust patches for brick and rust variety
      d) fine grain noise for surface roughness
    Returns uint8 BGR image, shape (h, w, 3).
    """
    # a) multi-octave value noise base (cheap alternative to Perlin noise)
    base = np.zeros((h, w), dtype=np.float32)
    for octave in range(1, 5):
        scale = 2 ** octave
        small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        layer = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        base += layer / octave
    base = (base - base.min()) / (base.max() - base.min() + 1e-6)

    # base color: mix of concrete-gray / dirt-brown, randomized per call
    gray = rng.uniform(80, 140)
    brown = np.array([rng.uniform(20, 60), rng.uniform(50, 90), rng.uniform(70, 110)])  # BGR
    tex = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        tex[:, :, c] = gray * (1 - base) + brown[c] * base

    # b) dark irregular "crack" chunks
    n_chunks = rng.integers(6, 14)
    for _ in range(n_chunks):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(min(h, w) // 20, min(h, w) // 6)
        n_pts = rng.integers(5, 9)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
        pts = np.array([
            [cx + int(r * np.cos(a) * rng.uniform(0.5, 1.2)),
             cy + int(r * np.sin(a) * rng.uniform(0.5, 1.2))]
            for a in angles
        ], dtype=np.int32)
        shade = rng.uniform(0.35, 0.65)
        overlay = tex.copy()
        cv2.fillPoly(overlay, [pts], (int(30 * shade), int(30 * shade), int(30 * shade)))
        tex = cv2.addWeighted(overlay, 0.55, tex, 0.45, 0)

    # c) rust/light patches for material variety
    n_patches = rng.integers(3, 8)
    for _ in range(n_patches):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        axes = (rng.integers(w // 15, w // 6), rng.integers(h // 15, h // 6))
        angle = rng.integers(0, 180)
        color = (rng.uniform(20, 50), rng.uniform(60, 100), rng.uniform(120, 170))  # rust BGR
        overlay = tex.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1)
        tex = cv2.addWeighted(overlay, 0.4, tex, 0.6, 0)

    # d) fine grain noise for surface roughness
    grain = rng.normal(0, 12, (h, w, 3)).astype(np.float32)
    tex = tex + grain
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (3, 3), 0)
    return tex


# ----------------------------------------------------------------------
# 2. Occlusion mask: irregular blob sized to hit a target area fraction
# ----------------------------------------------------------------------
def generate_occlusion_mask(bbox_w: int, bbox_h: int, target_frac: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Returns a binary mask (uint8, shape (bbox_h, bbox_w)) of an irregular
    blob (union of randomly placed ellipses) whose area is close to
    target_frac * (bbox_w * bbox_h).

    Placement bias: rubble tends to bury from the ground up / from one
    side, so blobs are weighted toward the bottom half or a random side
    rather than centered -- more realistic than a centered rectangle.
    """
    mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    target_area = target_frac * bbox_w * bbox_h

    # pick a burial direction: bottom, left, right, or scattered
    direction = rng.choice(["bottom", "left", "right", "scattered"])

    attempts = 0
    while mask.sum() / 255 < target_area and attempts < 60:
        attempts += 1
        if direction == "bottom":
            cy = int(rng.uniform(0.4, 1.0) * bbox_h)
            cx = int(rng.uniform(0.1, 0.9) * bbox_w)
        elif direction == "left":
            cy = int(rng.uniform(0.1, 0.9) * bbox_h)
            cx = int(rng.uniform(0.0, 0.6) * bbox_w)
        elif direction == "right":
            cy = int(rng.uniform(0.1, 0.9) * bbox_h)
            cx = int(rng.uniform(0.4, 1.0) * bbox_w)
        else:  # scattered
            cy = int(rng.uniform(0, bbox_h))
            cx = int(rng.uniform(0, bbox_w))

        remaining = max(target_area - mask.sum() / 255, bbox_w * bbox_h * 0.02)
        blob_area = min(remaining, target_area * rng.uniform(0.2, 0.5))
        r = max(2, int(np.sqrt(blob_area / np.pi)))
        axes = (max(2, int(r * rng.uniform(0.7, 1.4))), max(2, int(r * rng.uniform(0.7, 1.4))))
        angle = rng.integers(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    return mask


# ----------------------------------------------------------------------
# 3. Compositor: feathered alpha blend of texture into masked region
# ----------------------------------------------------------------------
def composite_occlusion(image: np.ndarray, bbox_xyxy, mask: np.ndarray,
                         texture: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    out = image.copy()
    roi = out[y1:y2, x1:x2]

    # feather mask edges so the patch doesn't look pasted
    feathered = cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32) / 255.0
    feathered = feathered[:, :, None]

    tex_resized = cv2.resize(texture, (roi.shape[1], roi.shape[0]))
    blended = roi.astype(np.float32) * (1 - feathered) + tex_resized.astype(np.float32) * feathered
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out


def actual_occlusion_fraction(mask: np.ndarray) -> float:
    return float(mask.sum() / 255) / float(mask.shape[0] * mask.shape[1])


# ----------------------------------------------------------------------
# 4. YOLO label I/O helpers
# ----------------------------------------------------------------------
def read_yolo_labels(label_path: Path):
    """Returns list of (class_id, xc, yc, w, h) all normalized floats."""
    if not label_path.exists():
        return []
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            rows.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return rows


def yolo_to_xyxy(row, img_w, img_h):
    _, xc, yc, w, h = row
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


# ----------------------------------------------------------------------
# 5. Main batch pipeline
# ----------------------------------------------------------------------
def process_dataset(images_dir: Path, labels_dir: Path, out_dir: Path,
                     levels=(25, 50, 75), seed=42):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []

    img_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in
                         (".jpg", ".jpeg", ".png")])

    for level in levels:
        (out_dir / f"occ_{level}pct" / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / f"occ_{level}pct" / "labels").mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        rows = read_yolo_labels(label_path)
        person_rows = [r for r in rows if r[0] == 0]  # assumes class 0 = person
        if not person_rows:
            continue

        for level in levels:
            occluded = image.copy()
            for row in person_rows:
                x1, y1, x2, y2 = yolo_to_xyxy(row, img_w, img_h)
                bw, bh = x2 - x1, y2 - y1
                if bw < 4 or bh < 4:
                    continue
                mask = generate_occlusion_mask(bw, bh, level / 100.0, rng)
                texture = generate_rubble_texture(bh, bw, rng)
                occluded = composite_occlusion(occluded, (x1, y1, x2, y2), mask, texture)
                actual_pct = actual_occlusion_fraction(mask) * 100
                log_rows.append({
                    "image": img_path.name,
                    "nominal_occlusion_pct": level,
                    "actual_occlusion_pct": round(actual_pct, 1),
                })

            out_img_path = out_dir / f"occ_{level}pct" / "images" / img_path.name
            cv2.imwrite(str(out_img_path), occluded)
            out_label_path = out_dir / f"occ_{level}pct" / "labels" / (img_path.stem + ".txt")
            with open(out_label_path, "w") as f:
                for r in rows:
                    f.write(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

    log_path = out_dir / "occlusion_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "nominal_occlusion_pct", "actual_occlusion_pct"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Done. {len(img_paths)} source images -> {len(levels)} occlusion tiers.")
    print(f"Occlusion log (use ACTUAL pct for stratified eval): {log_path}")


# ----------------------------------------------------------------------
# 6. Demo mode: synthetic person silhouettes, no real dataset needed
# ----------------------------------------------------------------------
def make_demo_dataset(demo_dir: Path, n_images=3, seed=0):
    """Creates fake 'person' images (simple silhouettes) + YOLO labels,
    just so the pipeline can be run and inspected before you plug in
    real COCO/CrowdHuman crops."""
    rng = np.random.default_rng(seed)
    img_dir = demo_dir / "raw" / "images"
    lbl_dir = demo_dir / "raw" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    W, H = 480, 640
    for i in range(n_images):
        img = np.full((H, W, 3), fill_value=(180, 190, 195), dtype=np.uint8)  # light bg
        # simple standing-person silhouette: head circle + body rectangle
        cx = W // 2 + rng.integers(-40, 40)
        top = H // 4 + rng.integers(-20, 20)
        body_w = 90
        body_h = 260
        head_r = 35
        color = (60, 60, 60)
        cv2.circle(img, (cx, top), head_r, color, -1)
        cv2.rectangle(img, (cx - body_w // 2, top + head_r),
                       (cx + body_w // 2, top + head_r + body_h), color, -1)

        x1, y1 = cx - body_w // 2 - 10, top - head_r - 10
        x2, y2 = cx + body_w // 2 + 10, top + head_r + body_h + 10
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        img_path = img_dir / f"demo_{i}.jpg"
        cv2.imwrite(str(img_path), img)

        xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        with open(lbl_dir / f"demo_{i}.txt", "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return img_dir, lbl_dir


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--labels_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--demo", action="store_true",
                         help="Generate synthetic person images first, then run the pipeline on them.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.demo:
        img_dir, lbl_dir = make_demo_dataset(out_dir, n_images=3, seed=args.seed)
        process_dataset(img_dir, lbl_dir, out_dir, levels=args.levels, seed=args.seed)
    else:
        assert args.images_dir and args.labels_dir, "Provide --images_dir and --labels_dir (or use --demo)."
        process_dataset(Path(args.images_dir), Path(args.labels_dir), out_dir,
                         levels=args.levels, seed=args.seed)
