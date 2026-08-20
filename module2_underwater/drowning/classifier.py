"""
classifier.py
--------------
Trains a classifier to distinguish "normal" swimming postures from
"distress" (drowning) postures, using the pose-based features extracted
by pose_extraction.py.

Workflow:
1. Loads all images from the samples/ folder.
2. Extracts pose-based distress features from each image.
3. Labels them based on filename prefix:
       "normal_..."   -> 0
       "distress_..." -> 1
4. Trains an SVM classifier on this data.
5. Saves the trained model to disk so it can be reused later
   (e.g., by fusion.py) without retraining every time.

Note: You need a reasonable number of samples (ideally 15-20+ per class)
for this to produce a meaningful model. With very few samples, the
accuracy numbers won't mean much yet - that's expected at this stage.

Naming convention for sample images in samples/:
    normal_001.jpg, normal_002.jpg, ...
    distress_001.jpg, distress_002.jpg, ...
"""

import os
import glob
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from pose_extraction import extract_features_from_image

SAMPLES_DIR = "samples"
MODEL_PATH = "drowning_classifier.joblib"
SCALER_PATH = "feature_scaler.joblib"

LABEL_MAP = {"normal": 0, "distress": 1}
LABEL_NAMES = ["normal", "distress"]

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def load_dataset(samples_dir=SAMPLES_DIR):
    """
    Loads all images from the samples folder, extracts pose features,
    and assigns labels based on the filename prefix.

    Returns: (X, y, filenames)
        X - 2D numpy array of feature vectors
        y - 1D numpy array of labels (0 = normal, 1 = distress)
        filenames - list of filenames, in the same order as X and y
    """
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(samples_dir, ext)))

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{samples_dir}'. Add labeled sample "
            f"images first (normal_*.jpg / distress_*.jpg)."
        )

    X = []
    y = []
    filenames = []

    for filepath in image_paths:
        filename = os.path.basename(filepath)

        label = None
        for prefix, label_value in LABEL_MAP.items():
            if filename.startswith(prefix):
                label = label_value
                break

        if label is None:
            print(f"Skipping unrecognized file: {filename}")
            continue

        features = extract_features_from_image(filepath)
        if features is None:
            print(f"No person detected, skipping: {filename}")
            continue

        X.append(features)
        y.append(label)
        filenames.append(filename)

    X = np.array(X)
    y = np.array(y)

    return X, y, filenames


def train_classifier(X, y):
    """
    Trains an SVM classifier on the given features and labels.
    Scales features first (SVM is sensitive to feature scale).

    Returns: (trained_model, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel="rbf", probability=True, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Evaluation on held-out test split ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    return model, scaler


def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Saves the trained model and scaler to disk for later use."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Loads a previously trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_distress(image_path, model=None, scaler=None):
    """
    Predicts whether a given image shows a "normal" swimming posture
    or a "distress" (drowning) posture.

    If model/scaler are not provided, loads them from disk.

    Returns: (label, confidence)
        label - "normal" or "distress"
        confidence - probability of the predicted class (0 to 1)
        or (None, None) if no person was detected in the image.
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    features = extract_features_from_image(image_path)
    if features is None:
        return None, None

    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction]

    label = LABEL_NAMES[prediction]
    return label, confidence


def main():
    print("=" * 50)
    print("Drowning Posture Classifier - Training")
    print("=" * 50)

    print("\nLoading dataset...")
    X, y, filenames = load_dataset()

    counts = {name: int(np.sum(y == value)) for name, value in LABEL_MAP.items()}
    print(f"Loaded {len(y)} samples -> {counts}")

    if min(counts.values()) < 5:
        print(
            "\nWarning: You have very few samples in at least one class. "
            "Results below are not statistically meaningful yet - "
            "aim for 15-20+ per class for a reliable model."
        )

    print("\nTraining SVM classifier...")
    model, scaler = train_classifier(X, y)

    save_model(model, scaler)


if __name__ == "__main__":
    main()