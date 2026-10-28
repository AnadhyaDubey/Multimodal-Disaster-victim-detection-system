"""
Shared utilities for the Multimodal Disaster Victim Detection dashboard.
- Custom HTML/CSS theme injection (Disaster/Rescue: dark navy + orange/red alerts)
- Model loading + inference placeholders (swap with real YOLO .pt once weights arrive)
- Fusion / triage scoring placeholder (Module 4 logic lives here until fusion.py is final)
"""

import streamlit as st
import random


# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

THEME = {
    "bg_primary": "#0a1929",      # deep navy background
    "bg_secondary": "#132f4c",    # card / panel navy
    "bg_tertiary": "#1c3d5a",     # hover / input background
    "accent_orange": "#ff6b35",   # primary alert accent
    "accent_red": "#e63946",      # critical / high-urgency accent
    "accent_amber": "#ffb703",    # medium-urgency accent
    "accent_green": "#2ecc71",    # low-urgency / safe accent
    "text_primary": "#f5f7fa",
    "text_secondary": "#a9c0d6",
    "border": "#26445e",
}


def apply_theme():
    """Inject the Disaster/Rescue custom theme (dark navy + orange/red) via HTML/CSS."""
    t = THEME
    st.markdown(
        f"""
        <style>
        /* ---------- App background & base text ---------- */
        .stApp {{
            background-color: {t['bg_primary']};
            color: {t['text_primary']};
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background-color: {t['bg_secondary']};
            border-right: 1px solid {t['border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {t['text_primary']} !important;
        }}

        /* ---------- Headings ---------- */
        h1, h2, h3 {{
            color: {t['text_primary']} !important;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: 0.3px;
        }}
        h1 {{
            border-bottom: 3px solid {t['accent_orange']};
            padding-bottom: 10px;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {{
            background-color: {t['accent_orange']};
            color: #0a1929;
            font-weight: 700;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            transition: 0.2s ease-in-out;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {t['accent_red']};
            color: white;
            transform: translateY(-1px);
        }}

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {t['bg_secondary']};
            border: 2px dashed {t['accent_orange']};
            border-radius: 10px;
        }}

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"] {{
            background-color: {t['bg_secondary']};
            border: 1px solid {t['border']};
            border-left: 5px solid {t['accent_orange']};
            border-radius: 8px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricValue"] {{
            color: {t['accent_orange']} !important;
        }}

        /* ---------- Dataframes / tables ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {t['border']};
            border-radius: 8px;
        }}

        /* ---------- Custom banner used on pages ---------- */
        .rescue-banner {{
            background: linear-gradient(90deg, {t['bg_secondary']} 0%, {t['bg_primary']} 100%);
            border-left: 6px solid {t['accent_red']};
            padding: 14px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .rescue-banner h3 {{
            margin: 0;
            color: {t['text_primary']} !important;
        }}
        .rescue-banner p {{
            margin: 4px 0 0 0;
            color: {t['text_secondary']};
            font-size: 0.9rem;
        }}

        /* ---------- Urgency badges ---------- */
        .badge-critical {{
            background-color: {t['accent_red']};
            color: white; padding: 3px 10px; border-radius: 12px;
            font-weight: 700; font-size: 0.8rem;
        }}
        .badge-high {{
            background-color: {t['accent_orange']};
            color: #0a1929; padding: 3px 10px; border-radius: 12px;
            font-weight: 700; font-size: 0.8rem;
        }}
        .badge-medium {{
            background-color: {t['accent_amber']};
            color: #0a1929; padding: 3px 10px; border-radius: 12px;
            font-weight: 700; font-size: 0.8rem;
        }}
        .badge-low {{
            background-color: {t['accent_green']};
            color: #0a1929; padding: 3px 10px; border-radius: 12px;
            font-weight: 700; font-size: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def rescue_banner(title: str, subtitle: str):
    """Render a themed banner header at the top of a page."""
    st.markdown(
        f"""
        <div class="rescue-banner">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def urgency_badge(level: str) -> str:
    """Return HTML for a colored urgency badge: critical / high / medium / low."""
    level = level.lower()
    css_class = {
        "critical": "badge-critical",
        "high": "badge-high",
        "medium": "badge-medium",
        "low": "badge-low",
    }.get(level, "badge-medium")
    return f'<span class="{css_class}">{level.upper()}</span>'


# ---------------------------------------------------------------------------
# MODEL / INFERENCE PLACEHOLDERS
# Swap these once best.pt (weights) is shared by your teammate.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "weights/best.pt"  # single-class ("person") YOLO detector, imgsz=640
CLASS_NAMES = {0: "person"}
MODEL_IMG_SIZE = 640


@st.cache_resource
def load_yolo_model(weights_path: str = WEIGHTS_PATH):
    """
    Load the YOLO occlusion-detection model.
    Returns None (placeholder mode) if weights file isn't found yet.
    """
    try:
        from ultralytics import YOLO
        import os
        if os.path.exists(weights_path):
            return YOLO(weights_path)
        return None
    except ImportError:
        return None


def run_occlusion_detection(image, model=None):
    """
    Run occlusion detection on a PIL/np image using the real YOLO model
    (single class: 'person'). Falls back to a simulated result if the
    model isn't loaded yet (e.g. weights missing in this environment).

    Returns a dict: {confidence, occlusion_pct, bbox, is_placeholder}
    occlusion_pct here is an estimate derived from detection confidence
    as a stand-in until a dedicated occlusion-% output is wired in from
    the model/training pipeline.
    """
    if model is not None:
        results = model.predict(image, imgsz=MODEL_IMG_SIZE, verbose=False)
        r = results[0]

        if len(r.boxes) == 0:
            return {
                "confidence": 0.0,
                "occlusion_pct": 0.0,
                "bbox": None,
                "is_placeholder": False,
                "no_detection": True,
            }

        # Take the highest-confidence detected person box
        best_idx = int(r.boxes.conf.argmax())
        conf = float(r.boxes.conf[best_idx])
        xyxy = [round(float(v), 1) for v in r.boxes.xyxy[best_idx].tolist()]

        # Rough occlusion estimate: lower detection confidence tends to
        # correlate with higher occlusion for a model trained this way.
        # Replace with the real stratified-occlusion metric once available.
        occlusion_pct = round((1 - conf) * 100, 1)

        return {
            "confidence": round(conf, 2),
            "occlusion_pct": occlusion_pct,
            "bbox": xyxy,
            "is_placeholder": False,
        }

    # ---- Placeholder / demo output (model not loaded) ----
    dummy_confidence = round(random.uniform(0.55, 0.97), 2)
    dummy_occlusion_pct = round(random.uniform(10, 80), 1)
    return {
        "confidence": dummy_confidence,
        "occlusion_pct": dummy_occlusion_pct,
        "bbox": [50, 40, 220, 260],
        "is_placeholder": True,
    }


# ---------------------------------------------------------------------------
# AUDIO DISTRESS DETECTION PLACEHOLDER (Module 1 — audio branch)
# Swap once the real classifier from module1_rgb_occlusion/audio/ is wired in.
# Classes match the trained classifier: noise, voice, tap.
# ---------------------------------------------------------------------------

AUDIO_CLASSES = ["noise", "voice", "tap"]
# Rough distress weighting per class — voice/tap (a call for help, knocking/
# banging) imply higher urgency than ambient noise.
AUDIO_DISTRESS_WEIGHT = {"noise": 0.20, "voice": 0.75, "tap": 0.60}


def run_audio_distress_detection(audio_bytes=None):
    """
    Classify a short audio clip into noise / voice / tap and derive a
    0-1 distress score from it. Returns None if no audio was provided.
    Currently a placeholder — replace with the real trained audio
    classifier once it's wired in from module1_rgb_occlusion/audio/.
    """
    if audio_bytes is None:
        return None

    predicted_class = random.choice(AUDIO_CLASSES)
    confidence = round(random.uniform(0.75, 0.96), 2)
    distress_score = AUDIO_DISTRESS_WEIGHT[predicted_class]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "distress_score": distress_score,
        "is_placeholder": True,
    }


# ---------------------------------------------------------------------------
# MODULE 2 (TRIAGE) — FUSION / SCORING PLACEHOLDER
# ---------------------------------------------------------------------------

def compute_priority_score(occlusion_confidence: float, occlusion_pct: float,
                            audio_distress: float = None) -> dict:
    """
    Combine Module 1 detection output (+ optional audio distress score)
    into a single rescue-priority score and urgency label.

    This is the placeholder for fusion.py's real logic. Replace the scoring
    formula below once the actual fusion methodology is finalized.
    """
    audio_component = audio_distress if audio_distress is not None else 0.5

    # Simple weighted placeholder formula:
    # higher occlusion % + higher distress + higher detection confidence => higher priority
    priority_score = (
        0.4 * (occlusion_pct / 100) +
        0.35 * audio_component +
        0.25 * occlusion_confidence
    )
    priority_score = round(priority_score * 100, 1)  # scale to 0-100

    if priority_score >= 75:
        urgency = "critical"
    elif priority_score >= 55:
        urgency = "high"
    elif priority_score >= 35:
        urgency = "medium"
    else:
        urgency = "low"

    return {"priority_score": priority_score, "urgency": urgency}