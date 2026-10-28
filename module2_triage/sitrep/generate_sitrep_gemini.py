"""
generate_sitrep_gemini.py
-------------------------
Same job as generate_sitrep.py (turn the ranked victim list into a short
situation report for field teams), but uses Google's FREE Gemini API
instead of Anthropic. Get a free key at https://aistudio.google.com/apikey

Setup:  pip install google-genai
"""

import os

# Free-tier model names change over time. If you get a "model not found" or
# quota error, open https://aistudio.google.com and use a model it lists.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a rescue coordination assistant. Given structured data about "
    "detected disaster victims (from a multimodal detection system combining "
    "camera and audio signals), write a short, clear situation report for "
    "field rescue teams. Be concise, prioritize actionable information, and "
    "flag any low-confidence detections that should be verified before "
    "committing resources. Do not invent details not present in the data."
)


def _format_victims(ranked_victims, rank_key):
    lines = []
    for v in sorted(ranked_victims, key=lambda x: x[rank_key]):
        lines.append(
            f"- {v['victim_id']}: rank={v[rank_key]}, "
            f"final_confidence={v['final_confidence']}, "
            f"time_buried={v['time_buried_minutes']}min, "
            f"accessibility={v['accessibility']}"
        )
    return "\n".join(lines)


def generate_sitrep(ranked_victims, rank_key="predicted_rank", api_key=None):
    """Returns the SITREP text. api_key falls back to GEMINI_API_KEY."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    prompt = (
        f"Detected victims (ranked by rescue priority):\n\n"
        f"{_format_victims(ranked_victims, rank_key)}\n\n"
        f"Write a SITREP for the field team."
    )
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text