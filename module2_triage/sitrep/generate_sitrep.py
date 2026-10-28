"""
generate_sitrep.py
--------------------
Generates a human-readable Situation Report (SITREP) from a ranked list
of triage victims, using the Claude API. Converts raw ranking data into
a short, actionable summary a rescue coordinator could actually use.

Setup (one-time):
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"

Usage:
    from generate_sitrep import generate_sitrep

    ranked_victims = rank_victims(victims)  # from rule_based_triage.py
                                             # or pairwise_ranker.py
    report = generate_sitrep(ranked_victims)
    print(report)
"""

import os
import anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a rescue coordination assistant. Given structured data about "
    "detected disaster victims (from a multimodal detection system combining "
    "camera and audio signals), write a short, clear situation report for "
    "field rescue teams. Be concise, prioritize actionable information, and "
    "flag any low-confidence detections that should be verified before "
    "committing resources. Do not invent details not present in the data."
)


def _format_victims_for_prompt(ranked_victims, rank_key="predicted_rank"):
    """
    Converts the victim list into a compact text block for the LLM prompt.
    """
    lines = []
    for v in sorted(ranked_victims, key=lambda x: x[rank_key]):
        lines.append(
            f"- {v['victim_id']}: rank={v[rank_key]}, "
            f"final_confidence={v['final_confidence']}, "
            f"time_buried={v['time_buried_minutes']}min, "
            f"accessibility={v['accessibility']}"
        )
    return "\n".join(lines)


def generate_sitrep(ranked_victims, rank_key="predicted_rank"):
    """
    Generates a SITREP for a ranked list of victims.

    ranked_victims: list of victim dicts, each containing at least
        victim_id, final_confidence, time_buried_minutes, accessibility,
        and a rank field (rank_key).
    rank_key: which rank field to sort/report by - "predicted_rank"
        (pairwise ranker output) or "rule_based_rank" (baseline output).

    Returns: the generated report as a string.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Run: export ANTHROPIC_API_KEY='your-key-here'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    victim_summary = _format_victims_for_prompt(ranked_victims, rank_key)
    user_prompt = (
        f"Detected victims (ranked by rescue priority):\n\n{victim_summary}\n\n"
        f"Write a SITREP for the field team."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scenario_generator"))
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ranking"))
    from generate_scenario import generate_scenario  # noqa: E402
    from pairwise_ranker import PairwiseRanker  # noqa: E402

    training_scenarios = [generate_scenario(n_victims=5, seed=s) for s in range(20)]
    ranker = PairwiseRanker()
    ranker.train(training_scenarios)

    test_victims = generate_scenario(n_victims=5, seed=42)
    ranked = ranker.rank_victims(test_victims)

    print("Generating SITREP...\n")
    report = generate_sitrep(ranked, rank_key="predicted_rank")

    print("=" * 60)
    print("SITUATION REPORT")
    print("=" * 60)
    print(report)