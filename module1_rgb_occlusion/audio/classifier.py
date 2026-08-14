"""
classifier.py
--------------
Trains a classifier to distinguish between three sound types:
    - "tap"    : intentional tapping/knocking (survivor signal)
    - "voice"  : moaning / calling out / human voice distress sounds
    - "noise"  : background noise (not a survivor signal)

Workflow:
1. Loads all .wav files from the recordings/ folder.
2. Extracts statistical features from each file.
3. Labels them based on filename prefix:
       "tap_..."   -> 2
       "voice_..." -> 1
       "noise_..." -> 0
4. Trains an SVM classifier on this data.
5. Saves the trained model to disk so it can be reused later
   (e.g., by fusion.py) without retraining every time.

Note: You need a reasonable number of samples (ideally 15-20+ per class)
for this to produce a meaningful model. With very few samples, the
accuracy numbers won't mean much yet - that's expected at this stage.
"""

import os
import glob
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from feature_extraction import extract_features_from_file

RECORDINGS_DIR = "recordings"
MODEL_PATH = "tap_classifier.joblib"
SCALER_PATH = "feature_scaler.joblib"

# Label mapping used throughout this file
LABEL_MAP = {"noise": 0, "voice": 1, "tap": 2}
LABEL_NAMES = ["noise", "voice", "tap"]  # index-aligned with LABEL_MAP values


def load_dataset(recordings_dir=RECORDINGS_DIR):
    """
    Loads all .wav files from the recordings folder, extracts features,
    and assigns labels based on the filename prefix.

    Returns: (X, y, filenames)
        X - 2D numpy array of feature vectors
        y - 1D numpy array of labels (0 = noise, 1 = voice, 2 = tap)
        filenames - list of filenames, in the same order as X and y
    """
    wav_files = glob.glob(os.path.join(recordings_dir, "*.wav"))

    if len(wav_files) == 0:
        raise FileNotFoundError(
            f"No .wav files found in '{recordings_dir}'. "
            f"Record some samples first using record_audio.py."
        )

    X = []
    y = []
    filenames = []

    for filepath in wav_files:
        filename = os.path.basename(filepath)

        label = None
        for prefix, label_value in LABEL_MAP.items():
            if filename.startswith(prefix):
                label = label_value
                break

        if label is None:
            print(f"Skipping unrecognized file: {filename}")
            continue

        features = extract_features_from_file(filepath, feature_type="statistical")
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

    # Split into train/test sets - if the dataset is very small,
    # this split may be too small to be meaningful, but the pipeline
    # will still run correctly.
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


def predict_tap(filepath, model=None, scaler=None):
    """
    Predicts the sound category for a given audio file:
    "tap", "voice", or "noise".

    If model/scaler are not provided, loads them from disk.

    Returns: (label, confidence)
        label - one of "tap", "voice", "noise"
        confidence - probability of the predicted class (0 to 1)
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    features = extract_features_from_file(filepath, feature_type="statistical")
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction]

    label = LABEL_NAMES[prediction]
    return label, confidence


def main():
    print("=" * 50)
    print("Sound Classifier - Training (tap / voice / noise)")
    print("=" * 50)

    print("\nLoading dataset...")
    X, y, filenames = load_dataset()

    counts = {name: int(np.sum(y == value)) for name, value in LABEL_MAP.items()}
    print(f"Loaded {len(y)} samples -> {counts}")

    if min(counts.values()) < 5:
        print(
            "\nWarning: You have very few samples in at least one class. "
            "Results below are not statistically meaningful yet - "
            "record more samples (aim for 15-20+ per class) for a "
            "reliable model."
        )

    print("\nTraining SVM classifier...")
    model, scaler = train_classifier(X, y)

    save_model(model, scaler)


if __name__ == "__main__":
    main()