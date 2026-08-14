"""
split_recording.py
-------------------
Records ONE long continuous clip and automatically splits it into
multiple fixed-length samples, saved with the correct label prefix.

This is a faster alternative to record_audio.py when you need many
samples quickly - e.g., record yourself tapping 20 times in a row for
60 seconds, then this script cuts that into ~20 separate 2-second clips.

Usage:
    python split_recording.py

You will be asked for:
    - a label ("tap", "voice", or "noise")
    - how many seconds to record continuously

Output: multiple .wav files in recordings/, named like:
    tap_20260802_110500_00.wav
    tap_20260802_110500_01.wav
    ...
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime

SAMPLE_RATE = 22050
CHANNELS = 1
OUTPUT_DIR = "recordings"
CHUNK_DURATION = 2.0  # seconds per sample, matches feature_extraction.py

VALID_LABELS = ("tap", "voice", "noise")


def record_long_clip(duration_seconds):
    """
    Records continuous audio for a fixed duration (in seconds).
    Returns: 1D numpy array of the full recording.
    """
    print(f"\nRecording for {duration_seconds} seconds... start now!")
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()  # blocks until recording finishes
    print("Recording finished.")
    return audio.flatten()


def split_into_chunks(audio, chunk_duration=CHUNK_DURATION, sample_rate=SAMPLE_RATE):
    """
    Splits a long audio array into fixed-length, non-overlapping chunks.
    Any leftover audio shorter than one chunk is discarded.

    Returns: list of 1D numpy arrays.
    """
    chunk_size = int(chunk_duration * sample_rate)
    n_chunks = len(audio) // chunk_size

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(audio[start:end])

    return chunks


def save_chunks(chunks, label, sample_rate=SAMPLE_RATE, output_dir=OUTPUT_DIR):
    """
    Saves a list of audio chunks as individual .wav files.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_paths = []
    for i, chunk in enumerate(chunks):
        filename = f"{label}_{timestamp}_{i:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, chunk, sample_rate)
        saved_paths.append(filepath)

    return saved_paths


def main():
    print("=" * 50)
    print("Batch Audio Recorder - Module 1 (Audio Fusion)")
    print("=" * 50)
    print("Record one long clip, and it will be auto-split into")
    print(f"{CHUNK_DURATION}-second samples.\n")

    label = input("Enter label (tap / voice / noise): ").strip().lower()
    while label not in VALID_LABELS:
        label = input("Please enter 'tap', 'voice', or 'noise': ").strip().lower()

    duration_input = input(
        "How many seconds to record continuously? (e.g., 60): "
    ).strip()
    duration_seconds = float(duration_input) if duration_input else 60.0

    input("Press Enter when you're ready to start recording...")

    audio = record_long_clip(duration_seconds)
    chunks = split_into_chunks(audio)

    if len(chunks) == 0:
        print("Recording was too short to produce any chunks. Try again with a longer duration.")
        return

    saved_paths = save_chunks(chunks, label)

    print(f"\nSaved {len(saved_paths)} chunks:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()