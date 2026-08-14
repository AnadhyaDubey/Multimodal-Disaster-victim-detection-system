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
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or frame read failed.")
                break

            results = model(frame, conf=conf_thresh, verbose=False)[0]
            annotated = results.plot()

            confidences = results.boxes.conf.tolist() if len(results.boxes) > 0 else []
            best_conf = max(confidences) if confidences else 0.0
            cv2.putText(annotated, f"best conf: {best_conf:.2f}  detections: {len(confidences)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)

            if not headless:
                cv2.imshow("Live occlusion detection (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                print(f"Reached max_frames={max_frames}, stopping.")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"Done. Processed {frame_count} frames.")
    if save_video:
        print(f"Saved video: {save_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to your trained model, e.g. stage_75pct/weights/best.pt")
    parser.add_argument("--source", type=str, default="0",
                         help="0 for laptop webcam, or a phone stream URL "
                              "like http://192.168.1.42:8080/video")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--save_video", type=str, default=None,
                         help="Optional path to save the annotated stream as a video file")
    parser.add_argument("--headless", action="store_true",
                         help="No display window -- for testing or remote machines")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Stop after N frames -- mainly for testing")
    args = parser.parse_args()

    run_live_demo(args.weights, args.source, args.conf_thresh,
                   args.save_video, args.headless, args.max_frames)