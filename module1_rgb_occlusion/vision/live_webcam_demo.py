"""
live_webcam_demo.py

Live inference demo using your ACTUAL trained model (not the pretrained
baseline). Reads from either your laptop's built-in webcam OR a phone
camera streamed over WiFi (e.g. via the "IP Webcam" Android app), so you
can hold the phone right up to a real rubble/debris prop instead of being
stuck at your laptop's fixed camera position.

Source options:
  --source 0                                  laptop webcam (default)
  --source "http://192.168.1.42:8080/video"   phone stream over WiFi

Usage (live display window):
    python live_webcam_demo.py --weights path/to/stage_75pct/best.pt --source 0

Usage (phone stream, also save a video file to submit):
    python live_webcam_demo.py --weights path/to/best.pt \
        --source "http://192.168.1.42:8080/video" \
        --save_video demo_output.mp4

Usage (headless test / no display window, e.g. remote machine):
    python live_webcam_demo.py --weights best.pt --source 0 \
        --headless --max_frames 100 --save_video test_out.mp4

Press 'q' in the display window to quit (display mode only).
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def resolve_source(source_str: str):
    """--source can be an int (webcam index) or a URL string (phone stream)."""
    try:
        return int(source_str)
    except ValueError:
        return source_str  # URL string, e.g. IP Webcam stream address


def run_live_demo(weights: str, source_str: str, conf_thresh: float = 0.25,
                   save_video: str = None, headless: bool = False,
                   max_frames: int = None):
    model = YOLO(weights)
    source = resolve_source(source_str)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_str}. "
                            f"If using a phone stream, confirm the URL works "
                            f"in a browser first, and that both devices are "
                            f"on the same WiFi network.")

    # Network streams often don't report fps/dimensions reliably -- fall back
    # to sane defaults so VideoWriter doesn't silently produce a broken file.
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))

    frame_count = 0
    print(f"Source opened: {source_str} ({width}x{height} @ ~{fps:.1f}fps)")
    print("Press 'q' in the display window to quit." if not headless else
          f"Headless mode -- running until max_frames={max_frames} or stream ends.")

    try:
        while True: