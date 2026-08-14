"""
curriculum_train.py

Curriculum-learning fine-tune of YOLO on occlusion-augmented data.

Trains in stages -- 25% occlusion (easy) -> 50% -> 75% (hard) -- with each
stage starting from the PREVIOUS stage's best checkpoint, not from scratch
and not with all difficulties mixed together from epoch 1. This mirrors how
a person learns: master the easy case first, then progressively harder ones.

Assumes data produced by rubble_occlusion_augmentation.py:
    <occluded_root>/occ_25pct/images/*.jpg, occ_25pct/labels/*.txt
    <occluded_root>/occ_50pct/images/*.jpg, occ_50pct/labels/*.txt
    <occluded_root>/occ_75pct/images/*.jpg, occ_75pct/labels/*.txt

What this script does NOT do (by design, kept separate):
- No stratified evaluation here -- that's eval_stratified.py's job, run
  AFTER this script finishes, using occlusion_log.csv's ACTUAL percentages.
- No MLflow logging wired in yet -- flagged as a TODO at the top of main(),
  add before your first real multi-stage run so every attempt is comparable.

Usage (Colab, GPU):
    python curriculum_train.py --occluded_root crowdhuman_data/occluded \
        --levels 25 50 75 --epochs_per_stage 10 --base_model yolo26n.pt

Usage (quick local smoke test, CPU, tiny data):
    python curriculum_train.py --occluded_root demo_out \
        --levels 25 50 75 --epochs_per_stage 1 --base_model yolo11n.pt --imgsz 128
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def split_and_prepare_yaml(tier_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """
    Given occ_XXpct/images + occ_XXpct/labels (flat, no train/val split yet),
    creates train/ and val/ subfolders and a data.yaml pointing at them --
    this is the layout ultralytics' YOLO.train() expects.

    Re-running this is safe: it re-copies into train/val each time rather
    than assuming a previous split still matches (useful if you re-run the
    augmentation script and get a different set of images).
    """
    images_dir = tier_dir / "images"
    labels_dir = tier_dir / "labels"
    all_images = sorted([p for p in images_dir.glob("*")
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir} -- did the "
                                 f"augmentation script actually run for this tier?")

    rng = random.Random(seed)
    rng.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_images = set(all_images[:n_val])
    train_images = set(all_images) - val_images

    for split_name, split_set in [("train", train_images), ("val", val_images)]:
        (tier_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (tier_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in split_set:
            label_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, tier_dir / split_name / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, tier_dir / split_name / "labels" / label_path.name)

    yaml_path = tier_dir / "data.yaml"
    data_yaml = {
        "path": str(tier_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"  {len(train_images)} train / {len(val_images)} val images -> {yaml_path}")
    return yaml_path


def run_curriculum(occluded_root: Path, levels, epochs_per_stage: int,
                    base_model: str, imgsz: int = 640, project_dir: str = "runs_curriculum"):
    current_weights = base_model  # starts as pretrained e.g. yolo26n.pt / yolo11n.pt
    stage_results = []

    for stage_idx, level in enumerate(levels):
        tier_dir = occluded_root / f"occ_{level}pct"
        print(f"\n=== Stage {stage_idx + 1}/{len(levels)}: {level}% occlusion "
              f"(starting from {current_weights}) ===")

        yaml_path = split_and_prepare_yaml(tier_dir)

        model = YOLO(current_weights)
        results = model.train(
            data=str(yaml_path),
            epochs=epochs_per_stage,
            imgsz=imgsz,
            project=project_dir,
            name=f"stage_{level}pct",
            exist_ok=True,
            verbose=False,
        )

        # Next stage starts from THIS stage's best checkpoint -- this is
        # the actual "curriculum" mechanism: knowledge carries forward.
        current_weights = str(Path(results.save_dir) / "weights" / "best.pt")
        stage_results.append((level, current_weights))
        print(f"Stage {level}% complete. Best weights saved: {current_weights}")

    print("\n=== Curriculum training complete ===")
    for level, weights_path in stage_results:
        print(f"  {level}% stage weights: {weights_path}")
    print(f"\nFinal model (use this for evaluation/demo): {stage_results[-1][1]}")
    return stage_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--occluded_root", type=str, required=True,
                         help="Folder containing occ_25pct/, occ_50pct/, occ_75pct/ "
                              "(output of rubble_occlusion_augmentation.py)")
    parser.add_argument("--levels", type=int, nargs="+", default=[25, 50, 75],
                         help="Order matters -- easiest first for curriculum learning")
    parser.add_argument("--epochs_per_stage", type=int, default=10)
    parser.add_argument("--base_model", type=str, default="yolo26n.pt",
                         help="yolo26n.pt or yolo11n.pt -- auto-downloads on first use")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project_dir", type=str, default="runs_curriculum",
                         help="Where checkpoints save -- point this at a Drive path in "
                              "Colab so each stage is protected the moment it finishes, "
                              "not just at the very end.")
    args = parser.parse_args()

    # TODO before your first real multi-stage run: wire in MLflow logging here
    # (mlflow.start_run() per stage, log epochs_per_stage/base_model/imgsz as
    # params, log final mAP as a metric) so runs are comparable later.

    run_curriculum(Path(args.occluded_root), args.levels, args.epochs_per_stage,
                    args.base_model, args.imgsz, args.project_dir)