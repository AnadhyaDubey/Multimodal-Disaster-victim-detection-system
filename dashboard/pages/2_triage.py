"""
Module 2 — Triage & Prioritization page.
Consumes Module 1's occlusion detection output (+ optional audio distress input)
and produces a rescue-priority ranking. This is where the real fusion logic
will eventually plug in.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_theme, rescue_banner, compute_priority_score, urgency_badge

st.set_page_config(page_title="Triage & Prioritization", page_icon="🩺", layout="wide")
apply_theme()

rescue_banner(
    "🩺 Module 2 — Triage & Prioritization",
    "Fuses detection signals into a single rescue-priority ranking."
)

m1_result = st.session_state.get("module1_result")

if not m1_result or "confidence" not in m1_result:
    st.warning(
        "No image detection found yet. Go to **RGB Occlusion Detection**, "
        "upload or capture an image (audio alone isn't enough — confidence "
        "and occlusion % come from the image), then come back here."
    )
    st.stop()

st.markdown("#### Incoming signal from Module 1")
c1, c2 = st.columns(2)
c1.metric("Occlusion Confidence", f"{m1_result['confidence'] * 100:.1f}%")
c2.metric("Occlusion %", f"{m1_result['occlusion_pct']:.1f}%")

st.markdown("---")

auto_distress = m1_result.get("audio_distress")
if auto_distress is not None:
    st.markdown("#### Audio distress score (auto-filled from Module 1's mic/audio input)")
    st.caption("🎙️ This came from the audio clip recorded/uploaded on the RGB Occlusion page — drag to override.")
else:
    st.markdown("#### Optional: add an audio distress score (0 = silent, 1 = high distress)")
    st.caption("No audio was provided on the RGB Occlusion page — set a value manually or leave at default.")

audio_score = st.slider(
    "Audio distress score",
    0.0, 1.0,
    value=auto_distress if auto_distress is not None else 0.5,
    step=0.05,
)

result = compute_priority_score(
    occlusion_confidence=m1_result["confidence"],
    occlusion_pct=m1_result["occlusion_pct"],
    audio_distress=audio_score,
)

st.markdown("---")
st.markdown("#### Rescue Priority Verdict")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Priority Score", f"{result['priority_score']} / 100")
with col2:
    st.markdown(
        f"**Urgency Level:** {urgency_badge(result['urgency'])}",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("#### Session Log (this run)")
log_df = pd.DataFrame([{
    "Occlusion Confidence": f"{m1_result['confidence']*100:.1f}%",
    "Occlusion %": f"{m1_result['occlusion_pct']:.1f}%",
    "Audio Distress": audio_score,
    "Priority Score": result["priority_score"],
    "Urgency": result["urgency"].upper(),
}])
st.dataframe(log_df, use_container_width=True)

# ---------------------------------------------------------------------------
# ADD TO SITREP LOG — collects victims across multiple runs so the
# Situation Report page can generate one LLM summary for all of them.
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### 📋 Add this victim to the Situation Report log")
st.caption(
    "The SITREP generator needs a couple of fields your detection pipeline "
    "doesn't produce on its own — fill these in from what you observed."
)

if "victims_log" not in st.session_state:
    st.session_state["victims_log"] = []

sc1, sc2 = st.columns(2)
with sc1:
    time_buried = st.number_input(
        "Estimated time buried/trapped (minutes)", min_value=0, value=30, step=5
    )
with sc2:
    accessibility = st.selectbox(
        "Accessibility", ["easy", "moderate", "difficult"], index=1
    )

st.caption(
    "📍 Add this victim's location so the **Route Optimization** page can plan "
    "a rescue route. Defaults to a point near campus — adjust to simulate "
    "different sites, or enter real coordinates if you have GPS data."
)
lc1, lc2 = st.columns(2)
with lc1:
    victim_lat = st.number_input(
        "Latitude", value=26.2183 + (len(st.session_state["victims_log"]) * 0.004),
        format="%.5f", key="victim_lat_input"
    )
with lc2:
    victim_lon = st.number_input(
        "Longitude", value=78.1828 + (len(st.session_state["victims_log"]) * 0.004),
        format="%.5f", key="victim_lon_input"
    )

if st.button("➕ Add to SITREP log"):
    victim_id = f"V{len(st.session_state['victims_log']) + 1}"
    st.session_state["victims_log"].append({
        "victim_id": victim_id,
        "final_confidence": round(result["priority_score"] / 100, 2),
        "time_buried_minutes": int(time_buried),
        "accessibility": accessibility,
        "urgency": result["urgency"],
        "lat": victim_lat,
        "lon": victim_lon,
    })
    st.success(f"✅ {victim_id} added to the log.")

if st.session_state["victims_log"]:
    st.markdown("##### Current SITREP log")
    st.dataframe(pd.DataFrame(st.session_state["victims_log"]), use_container_width=True)
    if st.button("🗑️ Clear log"):
        st.session_state["victims_log"] = []
        st.rerun()
    st.info(
        "Go to **Situation Report** to generate a SITREP, or **Route Optimization** "
        "to plan the fastest priority-weighted rescue route for this log."
    )