"""
Module 1 — RGB Occlusion Detection + Audio Distress page.
Image: upload a file OR capture live via camera -> run YOLO occlusion detection.
Audio: upload a clip OR record live via microphone -> run distress classification.
Both results are stored in st.session_state so the Triage page can use them.
"""

import streamlit as st
from PIL import Image, ImageDraw
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    apply_theme, rescue_banner, load_yolo_model, run_occlusion_detection,
    run_audio_distress_detection,
)

st.set_page_config(page_title="RGB Occlusion Detection", page_icon="📷", layout="wide")
apply_theme()

rescue_banner(
    "📷 Module 1 — RGB Occlusion Detection",
    "Upload or capture a scene, and optionally an audio clip, to detect victims."
)

model = load_yolo_model()
if model is None:
    st.warning(
        "⚠️ Running in **placeholder mode** — weights file (best.pt) not found yet. "
        "Detections below are simulated so the pipeline can be built/tested end-to-end."
    )

# ---------------------------------------------------------------------------
# IMAGE INPUT — upload or live camera
# ---------------------------------------------------------------------------
st.markdown("#### 1. Scene image")
image_mode = st.radio(
    "How do you want to provide the image?",
    ["Upload a file", "Use camera"],
    horizontal=True,
    key="image_mode",
)

image = None
if image_mode == "Upload a file":
    uploaded_file = st.file_uploader("Upload a scene image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
else:
    camera_file = st.camera_input("Capture a scene with your camera")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

col1, col2 = st.columns([1.2, 1])

if image is not None:
    with col1:
        st.markdown("##### Scene")
        result = run_occlusion_detection(image, model)

        display_img = image.copy()
        draw = ImageDraw.Draw(display_img)
        if result.get("bbox"):
            draw.rectangle(result["bbox"], outline="#ff6b35", width=4)
        st.image(display_img, use_container_width=True)

    with col2:
        st.markdown("##### Detection Results")
        if result.get("no_detection"):
            st.warning("No person detected in this image.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Detection Confidence", f"{result['confidence'] * 100:.1f}%")
            c2.metric("Occlusion (est.)", f"{result['occlusion_pct']:.1f}%")

            if result.get("is_placeholder"):
                st.caption("🔧 Simulated result — model not loaded in this environment.")
            else:
                st.caption("✅ Real detection from best.pt (class: person, imgsz=640).")

            st.session_state["module1_result"] = {
                "confidence": result["confidence"],
                "occlusion_pct": result["occlusion_pct"],
            }
else:
    with col1:
        st.info("👆 Upload an image or capture one with your camera to run detection.")

st.markdown("---")

# ---------------------------------------------------------------------------
# AUDIO INPUT — upload or live microphone
# ---------------------------------------------------------------------------
st.markdown("#### 2. Audio clip (optional — distress signal)")
audio_mode = st.radio(
    "How do you want to provide audio?",
    ["None", "Upload a file", "Record with microphone"],
    horizontal=True,
    key="audio_mode",
)

audio_bytes = None
if audio_mode == "Upload a file":
    audio_file = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.audio(audio_file)
elif audio_mode == "Record with microphone":
    mic_recording = st.audio_input("Record ambient sound / call for help")
    if mic_recording is not None:
        audio_bytes = mic_recording.read()
        st.audio(mic_recording)

if audio_bytes is not None:
    audio_result = run_audio_distress_detection(audio_bytes)
    a1, a2, a3 = st.columns(3)
    a1.metric("Predicted Class", audio_result["class"].capitalize())
    a2.metric("Confidence", f"{audio_result['confidence'] * 100:.1f}%")
    a3.metric("Distress Score", f"{audio_result['distress_score']:.2f}")

    if audio_result.get("is_placeholder"):
        st.caption("🔧 Simulated result — replace with the trained audio classifier once ready.")

    # Merge into module1_result so Triage page can auto-fill the distress slider
    existing = st.session_state.get("module1_result", {})
    existing["audio_distress"] = audio_result["distress_score"]
    st.session_state["module1_result"] = existing

if st.session_state.get("module1_result"):
    st.success("✅ Result(s) saved — go to **Triage & Prioritization** to see them ranked.")