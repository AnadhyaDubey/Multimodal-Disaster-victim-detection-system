"""
generate_sitrep_groq.py
-----------------------
Turns the ranked victim list into a short situation report (SITREP) for
field rescue teams, using the FREE Groq API (no credit card needed).
Get a key at https://console.groq.com/keys

Setup:  pip install groq
"""

import os
import time

# Groq retires model names now and then. If you get a "model not found /
# decommissioned" error, the code below picks another available chat model
# automatically. You can also force one: export GROQ_MODEL="model-name"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

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


def _pick_fallback_model(client):
    """Choose any available chat model if the default name was retired."""
    ids = [m.id for m in client.models.list().data]
    skip = ("whisper", "guard", "tts", "playai", "embed", "distil")
    chat = [i for i in ids if not any(s in i.lower() for s in skip)]
    for i in chat:
        if "llama" in i.lower() and "versatile" in i.lower():
            return i
    return chat[0] if chat else None


def generate_sitrep(ranked_victims, rank_key="predicted_rank", api_key=None):
    """Returns the SITREP text. api_key falls back to GROQ_API_KEY."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    from groq import Groq

    client = Groq(api_key=key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Detected victims (ranked by rescue priority):\n\n"
            f"{_format_victims(ranked_victims, rank_key)}\n\n"
            f"Write a SITREP for the field team."
        )},
    ]

    model = MODEL
    tried_fallback = False
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=500
            )
            return resp.choices[0].message.content
        except Exception as e:
            text = str(e).lower()
            gone = ("model" in text) and any(
                w in text for w in ("not found", "decommissioned", "does not exist", "deprecated")
            )
            busy = any(w in text for w in ("503", "429", "rate limit", "overloaded", "unavailable"))
            if gone and not tried_fallback:
                tried_fallback = True
                new_model = _pick_fallback_model(client)
                if not new_model:
                    raise
                model = new_model
                continue
            if busy and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise