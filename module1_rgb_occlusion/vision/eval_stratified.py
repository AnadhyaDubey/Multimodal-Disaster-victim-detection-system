"""
eval_stratified.py

The step that actually proves the project's core claim: evaluates the
trained model separately across occlusion difficulty BUCKETS based on
ACTUAL measured occlusion % (from occlusion_log.csv), not just the
nominal tier folder (occ_25pct/occ_50pct/occ_75pct) an image came from.

Why this matters: a single blended accuracy number hides whether the
model improved on hard cases (75%+) or only easy ones. Bucketing by
actual occlusion -- e.g. images that landed at 68% even though they were
generated under the "75%" nominal target -- gives an honest picture of
where the model actually struggles.

Matching logic: since occlusion_log.csv doesn't store per-box coordinates
(only image name + nominal + actual occlusion %), this script uses the
MEAN actual occlusion % across all logged boxes for a given (image, level)
as that image's difficulty for bucketing purposes. Detection matching
(TP/FP/FN) is done separately via IoU between predicted and ground-truth
boxes -- so recall/precision are computed correctly per-box, only the
BUCKETING uses the image-level average.

Usage:
    python eval_stratified.py \
        --occluded_root crowdhuman_data/occluded \
        --weights runs_curriculum/stage_75pct/weights/best.pt \
        --log_csv crowdhuman_data/occluded/occlusion_log.csv \
        --levels 25 50 75
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe, no display needed (works in Colab and scripts alike)
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def yolo_txt_to_xyxy_boxes(label_path: Path, img_w: int, img_h: int):
    """Reads a YOLO label .txt, returns list of [x1,y1,x2,y2] pixel boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching: each prediction (assumed sorted by confidence desc)
    claims the highest-IoU unmatched ground-truth box above threshold.
    Returns (n_true_positive, n_false_positive, n_false_negative).
    """
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_idx = 0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            i = iou(pred, gt)
            if i > best_iou:
                best_iou, best_idx = i, idx
        if best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def run_stratified_eval(occluded_root: Path, weights: str, log_csv: Path,
                         levels, conf_thresh=0.25, iou_thresh=0.5,
                         buckets=((0, 20), (20, 40), (40, 60), (60, 80), (80, 101))):
    model = YOLO(weights)
    log = pd.read_csv(log_csv)

    # mean actual occlusion % per (image, nominal level) -- see module docstring
    image_difficulty = (log.groupby(["image", "nominal_occlusion_pct"])["actual_occlusion_pct"]
                         .mean().reset_index())

    bucket_stats = {b: {"tp": 0, "fp": 0, "fn": 0, "n_images": 0} for b in buckets}
    per_image_rows = []

    for level in levels:
        val_images_dir = occluded_root / f"occ_{level}pct" / "val" / "images"
        val_labels_dir = occluded_root / f"occ_{level}pct" / "val" / "labels"
        if not val_images_dir.exists():
            print(f"Skipping level {level} -- {val_images_dir} not found "
                  f"(did curriculum_train.py's split step run for this tier?)")
            continue

        for img_path in sorted(val_images_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            row = image_difficulty[(image_difficulty["image"] == img_path.name) &
                                    (image_difficulty["nominal_occlusion_pct"] == level)]
            if row.empty:
                continue  # image not in log (e.g. had zero person boxes), skip
            actual_pct = float(row["actual_occlusion_pct"].iloc[0])

            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            gt_boxes = yolo_txt_to_xyxy_boxes(val_labels_dir / (img_path.stem + ".txt"),
                                               img_w, img_h)
            if not gt_boxes:
                continue

            results = model(str(img_path), conf=conf_thresh, verbose=False)[0]
            pred_boxes = results.boxes.xyxy.cpu().numpy().tolist() if len(results.boxes) > 0 else []

            tp, fp, fn = match_detections(pred_boxes, gt_boxes, iou_thresh)

            for b_low, b_high in buckets:
                if b_low <= actual_pct < b_high:
                    bucket_stats[(b_low, b_high)]["tp"] += tp
                    bucket_stats[(b_low, b_high)]["fp"] += fp
                    bucket_stats[(b_low, b_high)]["fn"] += fn
                    bucket_stats[(b_low, b_high)]["n_images"] += 1
                    break

            per_image_rows.append({
                "image": img_path.name, "nominal_level": level,
                "actual_occlusion_pct": actual_pct, "tp": tp, "fp": fp, "fn": fn,
            })

    print(f"\n{'Occlusion bucket':<20}{'Images':<10}{'Recall':<10}{'Precision':<12}{'TP/FP/FN'}")
    print("-" * 70)
    summary_rows = []
    for (low, high), stats in bucket_stats.items():
        tp, fp, fn, n = stats["tp"], stats["fp"], stats["fn"], stats["n_images"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        label = f"{low}-{high}%"
        print(f"{label:<20}{n:<10}{recall:<10.3f}{precision:<12.3f}{tp}/{fp}/{fn}")
        summary_rows.append({"bucket": label, "n_images": n, "recall": recall,
                              "precision": precision, "tp": tp, "fp": fp, "fn": fn})

    out_dir = occluded_root / "eval_results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "stratified_summary.csv", index=False)
    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'stratified_summary.csv'} and per_image_results.csv")

    plot_stratified_bars(summary_rows, out_dir)
    plot_occlusion_distribution(log, out_dir)

    return summary_rows


def plot_stratified_bars(summary_rows, out_dir: Path):
    """
    Grouped bar chart: recall + precision side by side per occlusion bucket.
    This is the headline chart -- shows whether performance holds up as
    occlusion gets harder, or collapses.
    """
    df = pd.DataFrame(summary_rows)
    df = df.dropna(subset=["recall", "precision"])  # skip empty buckets (e.g. 0-20%, 80-100%)
    if df.empty:
        print("No non-empty buckets to plot.")
        return

    long_df = df.melt(id_vars=["bucket"], value_vars=["recall", "precision"],
                       var_name="metric", value_name="value")

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=long_df, x="bucket", y="value", hue="metric",
                palette=["#2a78d6", "#eb6834"], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Actual occlusion bucket")
    ax.set_ylabel("Score")
    ax.set_title("Recall and precision by actual occlusion severity")
    ax.legend(title="")

    # label each bar with its value -- easier to read than eyeballing bar height
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2)

    fig.tight_layout()
    out_path = out_dir / "stratified_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_occlusion_distribution(log: pd.DataFrame, out_dir: Path):
    """
    Histogram of ACTUAL occlusion % achieved, split by nominal target level.
    Secondary/supporting plot -- sanity-checks that the augmentation script's
    randomized blobs actually landed close to their intended targets.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=log, x="actual_occlusion_pct", hue="nominal_occlusion_pct",
                 bins=20, palette=["#2a78d6", "#1baf7a", "#eb6834"], ax=ax,
                 multiple="layer", alpha=0.6)
    ax.set_xlabel("Actual occlusion % (measured)")
    ax.set_ylabel("Count")
    ax.set_title("Actual vs nominal occlusion -- augmentation script sanity check")

    fig.tight_layout()
    out_path = out_dir / "occlusion_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--occluded_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    run_stratified_eval(Path(args.occluded_root), args.weights, Path(args.log_csv),
                         args.levels, args.conf_thresh, args.iou_thresh)