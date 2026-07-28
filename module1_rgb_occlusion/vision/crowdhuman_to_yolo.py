"""
crowdhuman_to_yolo.py

Converts CrowdHuman's native annotation format (.odgt, one JSON object
per line) into YOLO-format label .txt files, so CrowdHuman can be used
with rubble_occlusion_augmentation.py and standard YOLO training.

CrowdHuman format background:
  Each line in annotation_train.odgt / annotation_val.odgt is a JSON
  object for one image:
    {
      "ID": "273271,c9db000d5146c15",
      "gtboxes": [
        {
          "tag": "person",                 # or "mask" for person-like non-human
          "fbox": [x, y, w, h],             # FULL body box (includes occluded parts)
          "vbox": [x, y, w, h],             # VISIBLE-only box
          "hbox": [x, y, w, h],             # head box
          "extra": {"ignore": 0, ...}
        },
        ...
      ]
    }

Design choice: we use fbox (full box), not vbox (visible-only box).
Reason: this project cares about where the whole person is, even when
most of their body is occluded by rubble -- that's the entire point of
Module 1. Training on vbox would teach the model to draw a box around
only the visible sliver, which is the opposite of what we want.

We skip:
  - tag == "mask" (non-human, e.g. mannequins/reflections -- CrowdHuman's
    own convention for "looks human, isn't")
  - extra.ignore == 1 (CrowdHuman's own flag for boxes not to be scored)

Usage:
    python crowdhuman_to_yolo.py \
        --odgt path/to/annotation_train.odgt \
        --images_dir path/to/CrowdHuman_train/Images \
        --labels_out path/to/CrowdHuman_train/labels_yolo

Demo mode (no real CrowdHuman data yet, just to sanity check the logic):
    python crowdhuman_to_yolo.py --demo --labels_out demo_labels_out
"""

import argparse
import json
from pathlib import Path

import cv2


def convert_odgt_line(record: dict, img_w: int, img_h: int):
    """
    Takes one parsed CrowdHuman JSON record + the real image's (w, h),
    returns a list of YOLO-format label lines: "0 xc yc w h" (all normalized).
    """
    lines = []
    for box in record.get("gtboxes", []):
        if box.get("tag") != "person":
            continue
        if box.get("extra", {}).get("ignore", 0) == 1:
            continue

        x, y, w, h = box["fbox"]

        # CrowdHuman boxes occasionally extend slightly outside the image
        # (annotation noise) -- clamp so normalized coords stay in [0, 1].
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        w_clamped, h_clamped = x2 - x1, y2 - y1
        if w_clamped <= 1 or h_clamped <= 1:
            continue  # degenerate box, skip

        xc = (x1 + w_clamped / 2) / img_w
        yc = (y1 + h_clamped / 2) / img_h
        wn = w_clamped / img_w
        hn = h_clamped / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return lines


def process_odgt(odgt_path: Path, images_dir: Path, labels_out: Path):
    labels_out.mkdir(parents=True, exist_ok=True)
    n_images, n_boxes, n_skipped_missing_img = 0, 0, 0

    with open(odgt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record["ID"]

            img_path = images_dir / f"{image_id}.jpg"
            if not img_path.exists():
                n_skipped_missing_img += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                n_skipped_missing_img += 1
                continue
            img_h, img_w = img.shape[:2]

            yolo_lines = convert_odgt_line(record, img_w, img_h)
            if not yolo_lines:
                continue

            out_path = labels_out / f"{image_id}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as out_f:
                out_f.write("\n".join(yolo_lines) + "\n")

            n_images += 1
            n_boxes += len(yolo_lines)

    print(f"Converted {n_images} images, {n_boxes} person boxes total.")
    if n_skipped_missing_img:
        print(f"Skipped {n_skipped_missing_img} entries (image file not found -- "
              f"check --images_dir path matches where you unzipped CrowdHuman).")


def make_demo_odgt(demo_dir: Path):
    """Creates a tiny fake .odgt + matching fake images, so the converter's
    logic can be verified without needing the real (large, registration-gated)
    CrowdHuman download."""
    img_dir = demo_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    W, H = 400, 300

    fake_records = [
        {
            "ID": "demo_img_0",
            "gtboxes": [
                {"tag": "person", "fbox": [50, 40, 80, 200], "extra": {"ignore": 0}},
                {"tag": "person", "fbox": [180, 60, 70, 190], "extra": {"ignore": 0}},
                {"tag": "mask", "fbox": [300, 100, 40, 100], "extra": {"ignore": 0}},  # should be skipped
            ],
        },
        {
            "ID": "demo_img_1",
            "gtboxes": [
                {"tag": "person", "fbox": [-5, 20, 60, 150], "extra": {"ignore": 0}},  # slightly out of bounds
                {"tag": "person", "fbox": [200, 200, 50, 60], "extra": {"ignore": 1}},  # should be skipped (ignore)
            ],
        },
    ]

    odgt_path = demo_dir / "demo_annotation.odgt"
    with open(odgt_path, "w") as f:
        for rec in fake_records:
            f.write(json.dumps(rec) + "\n")

    for rec in fake_records:
        img = np.full((H, W, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{rec['ID']}.jpg"), img)

    return odgt_path, img_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--odgt", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--labels_out", type=str, required=True)
    parser.add_argument("--demo", action="store_true",
                         help="Run against a tiny synthetic .odgt + fake images to verify logic.")
    args = parser.parse_args()

    if args.demo:
        demo_root = Path(args.labels_out).parent / "_demo_source"
        odgt_path, images_dir = make_demo_odgt(demo_root)
        process_odgt(odgt_path, images_dir, Path(args.labels_out))
    else:
        assert args.odgt and args.images_dir, "Provide --odgt and --images_dir (or use --demo)."
        process_odgt(Path(args.odgt), Path(args.images_dir), Path(args.labels_out))
