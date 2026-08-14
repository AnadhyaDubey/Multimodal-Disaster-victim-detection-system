"""
fusion.py
---------
Combines visual confidence (from the vision model) and audio confidence
(from the audio classifier) into a single final confidence score,
indicating how likely it is that a real survivor is present at a given
location.

=====================================================================
 FOR PERSON A (VISION) - READ THIS SECTION
=====================================================================
This file currently uses a PLACEHOLDER for the vision model output.
To plug in the real vision model, you only need to edit ONE function:
`get_visual_confidence()` below.

Your function must:
    - Accept an image path (string) as input
    - Return a single float between 0 and 1
      (0 = definitely not an occluded victim, 1 = definitely is)

Example of what to change:

    # BEFORE (placeholder):
    def get_visual_confidence(image_path):
        return 0.75

    # AFTER (your real model):
    from module1_rgb_occlusion.vision.detector import detect_occlusion

    def get_visual_confidence(image_path):
        return detect_occlusion(image_path)

Nothing else in this file needs to change - fuse_confidence() and the
weighting logic will keep working automatically once this function
returns real values.
=====================================================================
"""

import os
import sys

# Ensures classifier.py (in ../audio/) can be imported regardless of
# which folder this script is run from.
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio")
sys.path.append(AUDIO_DIR)

from classifier import predict_tap, load_model as load_audio_model  # noqa: E402

# ---- Fusion weights (adjustable) ----
VISUAL_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3

# Absolute paths to the trained audio model files (in ../audio/), so this
# script works correctly no matter which directory it's run from.
AUDIO_MODEL_PATH = os.path.join(AUDIO_DIR, "tap_classifier.joblib")
AUDIO_SCALER_PATH = os.path.join(AUDIO_DIR, "feature_scaler.joblib")

_audio_model = None
_audio_scaler = None


def _get_audio_model():
    """Lazily loads the trained audio model + scaler once, then reuses them."""
    global _audio_model, _audio_scaler
    if _audio_model is None:
        _audio_model, _audio_scaler = load_audio_model(
            model_path=AUDIO_MODEL_PATH, scaler_path=AUDIO_SCALER_PATH
        )
    return _audio_model, _audio_scaler


def get_visual_confidence(image_path):
    """
    PLACEHOLDER for the vision model output.
    See the module docstring above for integration instructions.
    """
    print(f"[placeholder] Pretending to run vision model on: {image_path}")
    return 0.75  # dummy fixed value until the real model is plugged in


def get_audio_confidence(audio_path):
    """
    Runs the trained audio classifier on an audio clip and converts
    its output into a single "distress confidence" score.

    Both "tap" and "voice" are treated as survivor signals; "noise" is not.
    Returns a confidence in the range 0 to 1.
    """
    model, scaler = _get_audio_model()
    label, confidence = predict_tap(audio_path, model=model, scaler=scaler)

    if label in ("tap", "voice"):
        return confidence
    else:
        # It's noise - so the "distress signal" confidence is low,
        # regardless of how confident the model is that it's noise.
        return 1 - confidence


def fuse_confidence(image_path, audio_path,
                     visual_weight=VISUAL_WEIGHT, audio_weight=AUDIO_WEIGHT):
    """
    Combines visual and audio confidence scores into one final score.

    Returns: dict with individual scores, the final fused confidence,
    and a simple verdict string for easy display in a demo.
    """
    visual_confidence = get_visual_confidence(image_path)
    audio_confidence = get_audio_confidence(audio_path)

    final_confidence = (
        visual_weight * visual_confidence + audio_weight * audio_confidence
    )

    if final_confidence >= 0.7:
        verdict = "LIKELY SURVIVOR DETECTED"
    elif final_confidence >= 0.4:
        verdict = "UNCERTAIN - needs further check"
    else:
        verdict = "LIKELY NO SURVIVOR"

    return {
        "visual_confidence": visual_confidence,
        "audio_confidence": audio_confidence,
        "final_confidence": final_confidence,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # Quick manual test - update these paths to real files to try it out.
    test_image = "sample_image.jpg"  # placeholder path - vision side will update this
    test_audio = os.path.join(AUDIO_DIR, "recordings", "tap_20260802_111418_00.wav")

    print("Running fusion on sample inputs (audio confidence is real, "
          "visual confidence is a placeholder for now)...\n")

    result = fuse_confidence(test_image, test_audio)

    print("\n--- Fusion Result ---")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")