"""
live_demo.py
-------------
Real-time demo: continuously listens to the microphone and classifies
each short audio window as "tap", "voice", or "noise" - live, as it
happens. When a survivor signal ("tap" or "voice") is detected, it
also speaks an audible alert out loud using voice_alert.py.

Designed for demonstrating the audio module to an audience (e.g., a
professor) without needing to record and replay files.

Usage:
    python live_demo.py

Press Ctrl+C to stop.
"""

import sounddevice as sd
import numpy as np
import time

from feature_extraction import extract_statistical_features
from classifier import load_model, LABEL_NAMES
from voice_alert import announce_detection

SAMPLE_RATE = 22050
CHUNK_DURATION = 2.0  # seconds per prediction window
SILENCE_THRESHOLD = 0.01  # RMS energy below this is treated as "quiet" (skip)

LABEL_DISPLAY = {
    "tap": "TAP DETECTED - possible survivor signal!",
    "voice": "VOICE DETECTED - possible survivor signal!",
    "noise": "... just background noise ...",
}


def classify_chunk(audio, model, scaler):
    """
    Extracts features from a raw audio chunk and returns (label, confidence).
    """
    features = extract_statistical_features(audio, sample_rate=SAMPLE_RATE)
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction]

    label = LABEL_NAMES[prediction]
    return label, confidence


def main():
    print("=" * 55)
    print("LIVE AUDIO MONITOR - Module 1 (Survivor Signal Detection)")
    print("=" * 55)
    print(f"Listening in {CHUNK_DURATION}-second windows... (Ctrl+C to stop)\n")

    model, scaler = load_model()

    try:
        while True:
            audio = sd.rec(
                int(CHUNK_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            rms_energy = np.sqrt(np.mean(audio ** 2))
            if rms_energy < SILENCE_THRESHOLD:
                print(f"[{time.strftime('%H:%M:%S')}] (quiet - skipping)")
                continue

            label, confidence = classify_chunk(audio, model, scaler)
            display_message = LABEL_DISPLAY.get(label, label)

            print(f"[{time.strftime('%H:%M:%S')}] {display_message} "
                  f"(confidence: {confidence:.2f})")

            if label in ("tap", "voice"):
                announce_detection(label)

    except KeyboardInterrupt:
        print("\n\nStopped listening. Demo ended.")


if __name__ == "__main__":
    main()