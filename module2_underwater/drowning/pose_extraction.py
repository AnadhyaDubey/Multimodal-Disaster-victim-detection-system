"""
pose_extraction.py
-------------------
Extracts body pose landmarks from an image using MediaPipe's newer
Tasks API (mediapipe.tasks), and computes a small set of features useful
for distinguishing normal swimming from drowning/distress postures.

NOTE: MediaPipe removed the old `mp.solutions.pose` API in recent
versions. This script uses the current `PoseLandmarker` Tasks API instead,
which requires a model file (.task) to be downloaded once - see setup
instructions below.

Key visual cues used for distress detection:
- Body orientation (vertical vs horizontal in water)
- Arm position relative to shoulders (raised/flailing vs propelling)
- Head tilt angle

This module is meant to be imported by classifier.py, but can also be
run directly on a single image for quick testing.

SETUP (run once):
    curl -o pose_landmarker.task -L \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
"""

import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

# Landmark indices (MediaPipe Pose has 33 total landmarks, same ordering
# as the old API - these indices are documented and stable).
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

_detector = None  # lazily initialized, reused across calls


def _get_detector():
    """
    Creates (once) and returns the PoseLandmarker detector.
    Reusing the same detector avoids reloading the model on every call.
    """
    global _detector
    if _detector is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Download it first:\n"
                f"curl -o pose_landmarker.task -L "
                f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                f"pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
        )
        _detector = mp_vision.PoseLandmarker.create_from_options(options)
    return _detector


def extract_landmarks(image_path):
    """
    Runs the PoseLandmarker on an image file.

    Returns: a list of normalized landmarks (x, y, z per point) for the
    first detected person, or None if no person was detected.
    """
    detector = _get_detector()
    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None

    # pose_landmarks is a list (one entry per detected person) - take the first
    return result.pose_landmarks[0]


def _get_point(landmarks, index):
    """Helper: returns (x, y) for a given landmark index, normalized 0-1 coords."""
    point = landmarks[index]
    return np.array([point.x, point.y])


def compute_distress_features(landmarks):
    """
    Computes a small feature vector from pose landmarks, capturing cues
    relevant to distress/drowning detection.

    Returns: 1D numpy array of features:
        [body_tilt_angle, wrist_height_relative_to_shoulder,
         shoulder_hip_vertical_ratio, head_tilt_angle]
    """
    left_shoulder = _get_point(landmarks, LEFT_SHOULDER)
    right_shoulder = _get_point(landmarks, RIGHT_SHOULDER)
    left_hip = _get_point(landmarks, LEFT_HIP)
    right_hip = _get_point(landmarks, RIGHT_HIP)
    left_wrist = _get_point(landmarks, LEFT_WRIST)
    right_wrist = _get_point(landmarks, RIGHT_WRIST)
    nose = _get_point(landmarks, NOSE)

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2

    # 1. Body tilt angle: angle of the shoulder-hip line relative to vertical.
    body_vector = hip_mid - shoulder_mid
    body_tilt_angle = np.degrees(np.arctan2(abs(body_vector[0]), abs(body_vector[1]) + 1e-6))

    # 2. Average wrist height relative to shoulder.
    avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
    wrist_height_relative = shoulder_mid[1] - avg_wrist_y

    # 3. Shoulder-hip vertical ratio.
    shoulder_hip_vertical_ratio = abs(shoulder_mid[1] - hip_mid[1])

    # 4. Head tilt angle.
    head_vector = nose - shoulder_mid
    head_tilt_angle = np.degrees(np.arctan2(abs(head_vector[0]), abs(head_vector[1]) + 1e-6))

    features = np.array([
        body_tilt_angle,
        wrist_height_relative,
        shoulder_hip_vertical_ratio,
        head_tilt_angle,
    ])

    return features


def extract_features_from_image(image_path):
    """
    Convenience function: loads an image file, extracts pose landmarks,
    and computes the distress feature vector.

    Returns: 1D numpy array of features, or None if no person was detected.
    """
    landmarks = extract_landmarks(image_path)
    if landmarks is None:
        return None

    return compute_distress_features(landmarks)


if __name__ == "__main__":
    test_image_path = "samples/image.png"

    if os.path.exists(test_image_path):
        features = extract_features_from_image(test_image_path)
        if features is None:
            print("No person detected in the image.")
        else:
            print(f"Feature vector: {features}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Add a sample image to test, then update the path above.")