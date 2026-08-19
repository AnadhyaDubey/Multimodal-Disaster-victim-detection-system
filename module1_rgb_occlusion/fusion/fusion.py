"""
fusion.py
---------
Module 1 (RGB Occlusion Detection) — fuses visual confidence (from Person A's
YOLO occlusion-detection model) with audio confidence (from Person B's tap/
voice SVM classifier) into a single final confidence score for a survivor
detection.

Design:
- Vision side: loads the trained YOLO weights (best.pt from curriculum
  training, stage_75pct) via ultralytics, runs inference on an image, and
  extracts the highest-confidence "person" detection as visual_confidence.
- Audio side: reuses classifier.predict_tap() from the audio sub-module,
  which returns (label, confidence) for "tap" / "voice" / "noise". The
  audio model/scaler are loaded explicitly via absolute paths so this
  works no matter what directory fusion.py is run from.
- Fusion: weighted sum of the two confidences. Audio only contributes when
  it actually detected a survivor signal (tap/voice) — "noise" contributes
  zero audio confidence, since background sound isn't evidence of a person.

Usage:
    from fusion import fuse_detection

    result = fuse_detection(
        image_path="some_frame.jpg",
        audio_path="some_clip.wav",
    )
    print(result)
    # {'visual_confidence': 0.82, 'audio_label': 'tap',
    #  'audio_confidence': 0.61, 'final_confidence': 0.757, 'person_detected': True}

Weights (visual vs audio) are adjustable via VISUAL_WEIGHT / AUDIO_WEIGHT
below, or by passing visual_weight= to fuse_detection().
"""

import os
import sys
from pathlib import Path

from ultralytics import YOLO

# ---- Make the audio sub-module importable regardless of cwd ----
# fusion.py lives at module1_rgb_occlusion/fusion/fusion.py, audio code lives
# at module1_rgb_occlusion/audio/ — add that folder to sys.path so we can
# import classifier.py directly, without turning audio/ into a package.
THIS_DIR = Path(__file__).resolve().parent
AUDIO_DIR = THIS_DIR.parent / "audio"
if str(AUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIO_DIR))

from classifier import predict_tap, load_model  # noqa: E402  (import after sys.path edit, intentional)

# ---- Config ----

# Path to the trained YOLO weights. This is the file Person A downloads via
# the "files.download(...)" cell in her Colab notebook (stage_75pct/best.pt).
# Update this path once the real file is placed in the repo.
DEFAULT_WEIGHTS_PATH = str(
    THIS_DIR.parent / "vision" / "weights" / "best.pt"
)

# Audio model files always live in audio/, regardless of where fusion.py is
# run from — using absolute paths here avoids the "relative path breaks when
# cwd is fusion/ instead of audio/" bug.
AUDIO_MODEL_PATH = str(AUDIO_DIR / "tap_classifier.joblib")
AUDIO_SCALER_PATH = str(AUDIO_DIR / "feature_scaler.joblib")

PERSON_CLASS_ID = 0  # confirmed from data.yaml: {0: "person"}

# How much each modality contributes to the final score.
# Vision is weighted higher by default since it's the primary signal;
# audio acts as a confirming/boosting signal. Adjustable per call.
VISUAL_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3

# Audio labels that count as a "survivor signal" (contribute to fusion).
# "noise" is excluded on purpose — it is not evidence of a person.
SURVIVOR_AUDIO_LABELS = ("tap", "voice")

# Caches so repeated calls don't reload weights/model from disk every time.
_vision_model_cache = {}
_audio_model_cache = {}


def _load_vision_model(weights_path=DEFAULT_WEIGHTS_PATH):
    """
    Loads (and caches) the YOLO model for the given weights path.
    """
    if weights_path not in _vision_model_cache:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights not found at: {weights_path}\n"
                f"Place best.pt at this path, or pass weights_path= explicitly "
                f"to fuse_detection() / get_visual_confidence()."
            )
        _vision_model_cache[weights_path] = YOLO(weights_path)
    return _vision_model_cache[weights_path]


def _load_audio_model():
    """
    Loads (and caches) the audio SVM model + scaler using absolute paths,
    so this works regardless of the current working directory.
    """
    if "model" not in _audio_model_cache:
        if not os.path.exists(AUDIO_MODEL_PATH) or not os.path.exists(AUDIO_SCALER_PATH):
            raise FileNotFoundError(
                f"Audio model/scaler not found at:\n"
                f"  {AUDIO_MODEL_PATH}\n  {AUDIO_SCALER_PATH}\n"
                f"Run classifier.py's main() first (from audio/) to train and save them."
            )
        model, scaler = load_model(model_path=AUDIO_MODEL_PATH, scaler_path=AUDIO_SCALER_PATH)
        _audio_model_cache["model"] = model
        _audio_model_cache["scaler"] = scaler
    return _audio_model_cache["model"], _audio_model_cache["scaler"]


def get_visual_confidence(image_path, weights_path=DEFAULT_WEIGHTS_PATH, conf_threshold=0.25):
    """
    Runs YOLO inference on a single image and returns the visual confidence
    of the best "person" detection.

    Returns: (visual_confidence, person_detected)
        visual_confidence - float 0.0-1.0. 0.0 if no person detected.
        person_detected   - bool, True if at least one person box passed
                             conf_threshold.
    """
    model = _load_vision_model(weights_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, False

    # Filter to the "person" class only, then take the highest-confidence box.
    person_confidences = [
        float(boxes.conf[i])
        for i in range(len(boxes))
        if int(boxes.cls[i]) == PERSON_CLASS_ID
    ]

    if not person_confidences:
        return 0.0, False

    return max(person_confidences), True


def get_audio_confidence(audio_path):
    """
    Runs the audio SVM classifier on a single .wav clip.

    Returns: (audio_label, audio_confidence_for_fusion)
        audio_label                  - "tap", "voice", or "noise"
        audio_confidence_for_fusion  - the classifier's confidence if the
                                        label is a survivor signal (tap/voice),
                                        otherwise 0.0 (noise contributes nothing)
    """
    model, scaler = _load_audio_model()
    label, confidence = predict_tap(audio_path, model=model, scaler=scaler)

    if label in SURVIVOR_AUDIO_LABELS:
        return label, float(confidence)
    return label, 0.0


def fuse_detection(
    image_path,
    audio_path,
    weights_path=DEFAULT_WEIGHTS_PATH,
    visual_weight=VISUAL_WEIGHT,
    audio_weight=AUDIO_WEIGHT,
    conf_threshold=0.25,
):
    """
    Combines visual and audio confidence into a single final score.

    final_confidence = visual_weight * visual_confidence
                      + audio_weight * audio_confidence_for_fusion

    Returns a dict:
        {
            "visual_confidence": float,
            "person_detected": bool,
            "audio_label": str,        # "tap" / "voice" / "noise"
            "audio_confidence": float, # 0.0 if audio_label == "noise"
            "final_confidence": float,
        }
    """
    visual_confidence, person_detected = get_visual_confidence(
        image_path, weights_path=weights_path, conf_threshold=conf_threshold
    )
    audio_label, audio_confidence = get_audio_confidence(audio_path)

    final_confidence = (visual_weight * visual_confidence) + (audio_weight * audio_confidence)

    return {
        "visual_confidence": visual_confidence,
        "person_detected": person_detected,
        "audio_label": audio_label,
        "audio_confidence": audio_confidence,
        "final_confidence": final_confidence,
    }


if __name__ == "__main__":
    # Quick manual test - update these paths to real files before running.
    import argparse

    parser = argparse.ArgumentParser(description="Test fusion on one image + one audio clip")
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument("--audio", required=True, help="Path to a test .wav clip")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH, help="Path to best.pt")
    args = parser.parse_args()

    result = fuse_detection(args.image, args.audio, weights_path=args.weights)

    print("=" * 50)
    print("Fusion result")
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")