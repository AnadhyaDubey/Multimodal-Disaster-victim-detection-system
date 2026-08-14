"""
record_audio.py
----------------
Interactive microphone recording script for Module 1 (Audio Fusion sub-module).

Usage: Press Enter to start recording.
       Press Enter again to stop recording and save the .wav file.

Use this to record "tap", "voice", and "noise" samples,
with different filenames/labels (e.g., tap_001.wav, voice_001.wav, noise_001.wav).
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime

# ---- Config ----
SAMPLE_RATE = 22050      # matches librosa's default sample rate
CHANNELS = 1              # mono recording is sufficient for this use case
OUTPUT_DIR = "recordings"  # will be created inside this script's folder

VALID_LABELS = ("tap", "voice", "noise")


def record_until_enter():
    """
    Records audio until Enter is pressed again.
    Returns: numpy array of recorded audio (float32)
    """
    print("\nRecording started... Press Enter to stop.")

    recorded_frames = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recorded_frames.append(indata.copy())

    # Stream stays open until explicitly closed; recording continues meanwhile
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback
    )

    with stream:
        input()  # waits here until Enter is pressed (recording continues)

    print("Recording stopped.")

    if len(recorded_frames) == 0:
        return None

    audio_data = np.concatenate(recorded_frames, axis=0)
    return audio_data


def save_recording(audio_data, label):
    """
    Saves the recorded audio as a .wav file,
    with the label + timestamp in the filename (to avoid overwriting).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    sf.write(filepath, audio_data, SAMPLE_RATE)
    print(f"Saved: {filepath}")
    return filepath


def main():
    print("=" * 50)
    print("Audio Recording Tool - Module 1 (Audio Fusion)")
    print("=" * 50)
    print("Enter a label before each recording:")
    print("  'tap'   -> tapping/knocking sound sample")
    print("  'voice' -> moaning/calling-out sample")
    print("  'noise' -> normal/background noise sample")
    print("(Type 'q' instead of a label to quit)\n")

    while True:
        label = input("Enter label (tap / voice / noise / q to quit): ").strip().lower()

        if label == "q":
            print("Goodbye! Recording session ended.")
            break

        if label not in VALID_LABELS:
            print("Please enter only 'tap', 'voice', 'noise', or 'q'. Try again.\n")
            continue

        input("Ready? Press Enter to start recording...")
        audio_data = record_until_enter()

        if audio_data is None or len(audio_data) == 0:
            print("Nothing was recorded, please try again.\n")
            continue

        save_recording(audio_data, label)
        print()  # blank line for spacing


if __name__ == "__main__":
    main()