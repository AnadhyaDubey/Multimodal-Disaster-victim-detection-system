"""
eval_ranking.py
-----------------
Evaluates ranking quality for Module 4's triage models (rule-based
baseline and the learned pairwise ranker), using two standard ranking
metrics:

    - Kendall's tau : measures rank correlation between predicted and
                       ground-truth ordering (-1 to 1, higher is better).
    - NDCG          : measures how well the top-ranked items match the
                       ground truth, weighting top-of-list errors more
                       heavily (0 to 1, higher is better).

Runs both models on multiple fresh test scenarios and reports the
average score across all of them, for a fair comparison.

Usage:
    python eval_ranking.py
"""

import sys
import os

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scenario_generator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "baseline"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ranking"))

from generate_scenario import generate_scenario  # noqa: E402
from rule_based_triage import rank_victims as rule_based_rank  # noqa: E402
from pairwise_ranker import PairwiseRanker  # noqa: E402

N_TRAIN_SCENARIOS = 20
N_TEST_SCENARIOS = 15
N_VICTIMS_PER_SCENARIO = 5


def evaluate_ranking(victims, predicted_rank_key):
    """
    Computes Kendall's tau and NDCG for one scenario, comparing the
    model's predicted rank against ground_truth_priority.

    Lower rank number = higher priority for both, so we convert ranks to
    "relevance scores" (higher = more urgent) for NDCG, since NDCG expects
    higher values to mean higher relevance.
    """
    n = len(victims)
    ground_truth_ranks = [v["ground_truth_priority"] for v in victims]
    predicted_ranks = [v[predicted_rank_key] for v in victims]

    tau, _ = kendalltau(ground_truth_ranks, predicted_ranks)

    # Convert ranks (1 = most urgent) into relevance scores (higher = more urgent)
    true_relevance = np.array([[n - r + 1 for r in ground_truth_ranks]])
    predicted_relevance = np.array([[n - r + 1 for r in predicted_ranks]])
    ndcg = ndcg_score(true_relevance, predicted_relevance)

    return tau, ndcg


def main():
    print("=" * 70)
    print("Module 4 - Triage Ranking Evaluation")
    print("=" * 70)

    # Train the pairwise ranker once, on scenarios separate from the test set.
    training_scenarios = [
        generate_scenario(n_victims=N_VICTIMS_PER_SCENARIO, seed=s)
        for s in range(N_TRAIN_SCENARIOS)
    ]
    ranker = PairwiseRanker()
    ranker.train(training_scenarios)
    print()

    rule_taus, rule_ndcgs = [], []
    pairwise_taus, pairwise_ndcgs = [], []

    for s in range(N_TRAIN_SCENARIOS, N_TRAIN_SCENARIOS + N_TEST_SCENARIOS):
        victims = generate_scenario(n_victims=N_VICTIMS_PER_SCENARIO, seed=s)

        # Rule-based baseline (operates on a fresh copy of the same victims)
        rule_victims = [dict(v) for v in victims]
        rule_based_rank(rule_victims)
        tau, ndcg = evaluate_ranking(rule_victims, "rule_based_rank")
        rule_taus.append(tau)
        rule_ndcgs.append(ndcg)

        # Pairwise ranker (fresh copy again, so both start from identical data)
        pairwise_victims = [dict(v) for v in victims]
        ranker.rank_victims(pairwise_victims)
        tau, ndcg = evaluate_ranking(pairwise_victims, "predicted_rank")
        pairwise_taus.append(tau)
        pairwise_ndcgs.append(ndcg)

    print("-" * 70)
    print(f"{'Model':<20} {'Avg Kendall Tau':<20} {'Avg NDCG':<15}")
    print("-" * 70)
    print(f"{'Rule-Based':<20} {np.mean(rule_taus):<20.3f} {np.mean(rule_ndcgs):<15.3f}")
    print(f"{'Pairwise Ranker':<20} {np.mean(pairwise_taus):<20.3f} {np.mean(pairwise_ndcgs):<15.3f}")
    print("-" * 70)
    print(f"\nEvaluated over {N_TEST_SCENARIOS} held-out test scenarios "
          f"({N_VICTIMS_PER_SCENARIO} victims each).")


if __name__ == "__main__":
    main()