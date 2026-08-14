"""
feature_extraction.py
----------------------
Converts recorded .wav audio files into features suitable for classification.

Two feature types are provided:
1. Mel-spectrogram (2D) - useful if you plan to use a CNN later.
2. Flattened statistical features (1D) - useful for classical ML models
   like SVM, which expect a fixed-length feature vector per sample.

This module is meant to be imported by classifier.py and dataset_builder.py,
not run directly (though a small test block is included at the bottom).
"""

import librosa
import numpy as np
import os

# ---- Config (should match record_audio.py) ----
SAMPLE_RATE = 22050
N_MELS = 64          # number of mel frequency bands
FIXED_DURATION = 2.0  # seconds - all clips will be padded/trimmed to this length


def load_audio(filepath, sample_rate=SAMPLE_RATE, fixed_duration=FIXED_DURATION):
    """
    Loads a .wav file and standardizes its length so all samples
    have the same shape, regardless of how long the original recording was.

    Returns: 1D numpy array of audio samples.
    """
    audio, sr = librosa.load(filepath, sr=sample_rate)

    target_length = int(fixed_duration * sample_rate)

    if len(audio) > target_length:
        # Trim from the center to keep the most relevant part of the clip
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    else:
        # Pad with zeros (silence) if the clip is shorter than target length
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio


def extract_mel_spectrogram(audio, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    """
    Extracts a mel-spectrogram from raw audio.
    Useful if a CNN classifier is used later.

    Returns: 2D numpy array (n_mels x time_frames), in decibel scale.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_statistical_features(audio, sample_rate=SAMPLE_RATE):
    """
    Extracts a fixed-length feature vector suitable for classical ML models
    (e.g., SVM). Combines several standard audio descriptors:
    - MFCCs (mean + std across time)
    - Spectral centroid (mean)
    - Zero-crossing rate (mean)
    - RMS energy (mean)

    Returns: 1D numpy array (fixed-length feature vector).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_mean = np.mean(spectral_centroid)

    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [spectral_centroid_mean, zcr_mean, rms_mean]
    ])

    return feature_vector


def extract_features_from_file(filepath, feature_type="statistical"):
    """
    Convenience function: loads a .wav file and extracts features in one call.

    feature_type: "statistical" (for SVM) or "melspec" (for CNN)
    """
    audio = load_audio(filepath)

    if feature_type == "statistical":
        return extract_statistical_features(audio)
    elif feature_type == "melspec":
        return extract_mel_spectrogram(audio)
    else:
        raise ValueError("feature_type must be 'statistical' or 'melspec'")


if __name__ == "__main__":
    # Quick manual test - point this to one of your recorded samples
    test_file = os.path.join("recordings", "tap_20260728_165552.wav")

    if os.path.exists(test_file):
        features = extract_features_from_file(test_file, feature_type="statistical")
        print(f"Statistical feature vector shape: {features.shape}")
        print(features)
    else:
        print(f"Test file not found: {test_file}")
        print("Record a sample first using record_audio.py, then update the path above.")