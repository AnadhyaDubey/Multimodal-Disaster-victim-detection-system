"""
Situation Report (SITREP) page.
Takes the victims collected on the Triage page and generates an LLM-written
SITREP for field rescue teams, using module2_triage/sitrep/generate_sitrep_gemini.py (free Gemini API).
"""

import streamlit as st
import pandas as pd
import sys
import os

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.append(DASHBOARD_DIR)  # for utils.py
sys.path.append(REPO_ROOT)      # for module2_triage/...
from utils import apply_theme, rescue_banner

st.set_page_config(page_title="Situation Report", page_icon="📝", layout="wide")
apply_theme()

rescue_banner(
    "📝 Situation Report (SITREP)",
    "Generates an LLM-written summary for field rescue teams from the triage log."
)

victims_log = st.session_state.get("victims_log", [])

if not victims_log:
    st.warning(
        "No victims in the SITREP log yet. Go to **Triage & Prioritization**, "
        "run a detection, and click **Add to SITREP log** for each victim."
    )
    st.stop()

# Rank victims by priority (highest final_confidence = rank 1)
ranked_victims = sorted(victims_log, key=lambda v: v["final_confidence"], reverse=True)
for i, v in enumerate(ranked_victims, start=1):
    v["rule_based_rank"] = i

st.markdown("#### Victims in this report")
st.dataframe(pd.DataFrame(ranked_victims), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# API key — use env var if already set, otherwise let the user paste one
# for this session only (never written to disk).
# ---------------------------------------------------------------------------
def _saved_key():
    """Key from .streamlit/secrets.toml (local) or the Secrets box (Streamlit Cloud)."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


api_key = (
    _saved_key()
    or os.environ.get("GEMINI_API_KEY")
    or st.session_state.get("gemini_api_key")
)

if not api_key:
    st.info("No saved Gemini key found. Save it once in .streamlit/secrets.toml, or paste it below and press Enter. Free key: aistudio.google.com/apikey")
    entered_key = st.text_input("Paste your free Gemini API key for this session", type="password")
    if entered_key:
        st.session_state["gemini_api_key"] = entered_key
        api_key = entered_key

st.markdown("---")

if st.button("🧠 Generate SITREP", disabled=not api_key):
    try:
        # imported lazily so the page still loads even if `google-genai` isn't installed
        from module2_triage.sitrep.generate_sitrep_gemini import generate_sitrep
        with st.spinner("Generating situation report..."):
            report = generate_sitrep(ranked_victims, rank_key="rule_based_rank", api_key=api_key)
        st.markdown("#### Report")
        st.markdown(
            f'<div style="background:#132f4c;border-left:5px solid #ff6b35;'
            f'padding:16px 20px;border-radius:8px;white-space:pre-wrap;">{report}</div>',
            unsafe_allow_html=True,
        )
    except ImportError:
        st.error("The `google-genai` package isn't installed. Run: `pip install google-genai`")
    except EnvironmentError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"SITREP generation failed: {e}")

if not api_key:
    st.caption("Enter an API key above to enable the Generate button.")