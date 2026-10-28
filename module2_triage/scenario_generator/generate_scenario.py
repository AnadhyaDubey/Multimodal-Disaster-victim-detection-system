"""
generate_scenario.py
---------------------
Generates synthetic multi-victim disaster scenarios for Module 4 (Triage).

Each victim has features derived from Module 1's outputs (visual + audio
fusion), plus a couple of situational factors used for triage ranking:

    - visual_confidence   : 0-1, from the vision model (person detection)
    - audio_confidence     : 0-1, from the audio classifier (tap/voice signal)
    - final_confidence     : fused score (same formula as fusion.py)
    - time_buried_minutes  : how long the victim has been trapped
    - accessibility        : 0-1, how easy it is for rescuers to reach them
                              (1 = easy, 0 = very difficult)

A "ground_truth_priority" is also generated for evaluation purposes later
(used to check ranking quality via Kendall's tau / NDCG in eval/).

Usage:
    from generate_scenario import generate_scenario

    victims = generate_scenario(n_victims=5, seed=42)
"""

import numpy as np

VISUAL_WEIGHT = 0.7
AUDIO_WEIGHT = 0.3


def generate_scenario(n_victims=5, seed=None):
    """
    Generates a synthetic disaster scenario with multiple victims.

    Returns: list of dicts, one per victim, each with:
        victim_id, visual_confidence, audio_confidence, final_confidence,
        time_buried_minutes, accessibility, ground_truth_priority
    """
    rng = np.random.default_rng(seed)

    victims = []
    for i in range(n_victims):
        visual_confidence = round(float(rng.uniform(0.2, 0.98)), 2)
        audio_confidence = round(float(rng.uniform(0.0, 0.95)), 2)
        final_confidence = round(
            VISUAL_WEIGHT * visual_confidence + AUDIO_WEIGHT * audio_confidence, 3
        )

        time_buried_minutes = int(rng.integers(5, 240))
        accessibility = round(float(rng.uniform(0.1, 1.0)), 2)

        victims.append({
            "victim_id": f"V{i + 1:02d}",
            "visual_confidence": visual_confidence,
            "audio_confidence": audio_confidence,
            "final_confidence": final_confidence,
            "time_buried_minutes": time_buried_minutes,
            "accessibility": accessibility,
        })

    # Ground-truth priority: higher confidence + longer time buried + easier
    # access = higher priority to rescue first. This combined score is used
    # only to generate a plausible "true" ranking for evaluation purposes -
    # the actual ranking model should learn to approximate this from the
    # individual features.
    for v in victims:
        v["_priority_score"] = (
            0.5 * v["final_confidence"]
            + 0.3 * (v["time_buried_minutes"] / 240)
            + 0.2 * v["accessibility"]
        )

    victims_sorted = sorted(victims, key=lambda v: v["_priority_score"], reverse=True)
    for rank, v in enumerate(victims_sorted, start=1):
        v["ground_truth_priority"] = rank
        del v["_priority_score"]

    return victims


if __name__ == "__main__":
    scenario = generate_scenario(n_victims=5, seed=42)

    print("=" * 70)
    print("Generated Scenario")
    print("=" * 70)
    for v in sorted(scenario, key=lambda x: x["ground_truth_priority"]):
        print(
            f"{v['victim_id']}  priority={v['ground_truth_priority']}  "
            f"final_conf={v['final_confidence']}  "
            f"time_buried={v['time_buried_minutes']}min  "
            f"accessibility={v['accessibility']}"
        )