"""
label_images.py
----------------
Interactive labeling tool for drowning-detection sample images.

Workflow:
1. Put your raw (unlabeled) images into a folder called raw_images/
   (any filenames, e.g., downloaded from Google Images).
2. Run this script. For each image, it will:
     - Open the image in your default image viewer (Preview on Mac)
     - Ask you to type a label: "normal", "distress", or "skip"
3. The image is copied into samples/ with the correct prefix and a
   sequential number, e.g., normal_001.jpg, distress_001.jpg, etc.

This saves you from manually renaming each file one by one.
"""

import os
import glob
import shutil
import subprocess
import sys

RAW_DIR = "raw_images"
SAMPLES_DIR = "samples"
VALID_LABELS = ("normal", "distress")

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def open_image(filepath):
    """Opens the image using the OS default viewer (cross-platform best effort)."""
    if sys.platform == "darwin":
        subprocess.run(["open", filepath])
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", filepath])
    elif sys.platform == "win32":
        os.startfile(filepath)  # noqa: (Windows only)
    else:
        print(f"Could not auto-open image, please view manually: {filepath}")


def get_next_index(samples_dir, label):
    """
    Finds the next available sequential number for a given label,
    so existing files are never overwritten.
    """
    existing = glob.glob(os.path.join(samples_dir, f"{label}_*.jpg")) + \
        glob.glob(os.path.join(samples_dir, f"{label}_*.jpeg")) + \
        glob.glob(os.path.join(samples_dir, f"{label}_*.png"))
    return len(existing) + 1


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    raw_images = []
    for ext in IMAGE_EXTENSIONS:
        raw_images.extend(glob.glob(os.path.join(RAW_DIR, ext)))

    if len(raw_images) == 0:
        print(f"No images found in '{RAW_DIR}/'. Add some images there first.")
        return

    print("=" * 50)
    print("Image Labeling Tool - Drowning Detection")
    print("=" * 50)
    print(f"Found {len(raw_images)} images to label.\n")

    for filepath in raw_images:
        filename = os.path.basename(filepath)
        print(f"\nOpening: {filename}")
        open_image(filepath)

        label = input(
            "Label this image (normal / distress / skip / q to quit): "
        ).strip().lower()

        if label == "q":
            print("Stopping labeling session.")
            break

        if label == "skip":
            print("Skipped.")
            continue

        if label not in VALID_LABELS:
            print("Invalid label, skipping this image.")
            continue

        ext = os.path.splitext(filename)[1]
        index = get_next_index(SAMPLES_DIR, label)
        new_filename = f"{label}_{index:03d}{ext}"
        new_filepath = os.path.join(SAMPLES_DIR, new_filename)

        shutil.copy(filepath, new_filepath)
        print(f"Saved as: {new_filepath}")

    print("\nLabeling session complete.")


if __name__ == "__main__":
    main()