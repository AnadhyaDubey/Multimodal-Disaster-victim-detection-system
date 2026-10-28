"""
rule_based_triage.py
---------------------
Simple rule-based triage baseline (inspired by the START protocol) for
Module 4. Ranks victims by priority using straightforward, interpretable
rules - no machine learning. This serves as the baseline to compare the
learned ranking model (pairwise_ranker.py) against later.

Rules used:
    1. Confidence gate: victims with very low final_confidence are treated
       as likely false detections and pushed to the bottom (not a real
       survivor signal).
    2. Among confirmed detections, prioritize by a weighted combination of
       final_confidence, time_buried, and accessibility - same idea as the
       ground truth score in generate_scenario.py, but presented as an
       explicit, auditable rule rather than "black box" ranking.

Usage:
    from rule_based_triage import rank_victims

    ranked = rank_victims(victims)
"""

CONFIDENCE_GATE = 0.4  # below this, treat as likely false positive

# Same weighting as the ground-truth formula in generate_scenario.py -
# kept identical on purpose so the baseline is a fair, explainable
# approximation of it (not an arbitrary rule set).
CONFIDENCE_WEIGHT = 0.5
TIME_WEIGHT = 0.3
ACCESSIBILITY_WEIGHT = 0.2
MAX_TIME_BURIED = 240  # minutes, used to normalize time_buried


def compute_rule_based_score(victim):
    """
    Computes a single priority score for one victim using explicit rules.
    Higher score = higher rescue priority.
    """
    if victim["final_confidence"] < CONFIDENCE_GATE:
        return 0.0  # likely false detection - lowest priority

    normalized_time = min(victim["time_buried_minutes"] / MAX_TIME_BURIED, 1.0)

    score = (
        CONFIDENCE_WEIGHT * victim["final_confidence"]
        + TIME_WEIGHT * normalized_time
        + ACCESSIBILITY_WEIGHT * victim["accessibility"]
    )
    return round(score, 3)


def rank_victims(victims):
    """
    Ranks a list of victim dicts (from generate_scenario.py) by rule-based
    priority score, highest first.

    Returns: the same list of dicts, each with an added "rule_based_score"
    and "rule_based_rank" field, sorted by rank.
    """
    for v in victims:
        v["rule_based_score"] = compute_rule_based_score(v)

    ranked = sorted(victims, key=lambda v: v["rule_based_score"], reverse=True)

    for rank, v in enumerate(ranked, start=1):
        v["rule_based_rank"] = rank

    return ranked


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scenario_generator"))
    from generate_scenario import generate_scenario  # noqa: E402

    victims = generate_scenario(n_victims=5, seed=42)
    ranked = rank_victims(victims)

    print("=" * 70)
    print("Rule-Based Triage Ranking")
    print("=" * 70)
    for v in ranked:
        print(
            f"{v['victim_id']}  rule_rank={v['rule_based_rank']}  "
            f"rule_score={v['rule_based_score']}  "
            f"ground_truth_rank={v['ground_truth_priority']}"
        )