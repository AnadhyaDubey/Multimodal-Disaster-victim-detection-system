"""
pairwise_ranker.py
--------------------
Learned ranking model for Module 4 triage, using the pairwise ranking
approach: instead of predicting an absolute priority score, the model
learns to answer "given two victims, who should be rescued first?" and
this is then used to rank an entire list of victims.

Why pairwise: with small scenario datasets, learning relative comparisons
generalizes better than trying to regress an exact priority score.

Workflow:
    1. train_ranker.py generates many scenarios and builds pairwise
       training examples (feature difference -> which victim wins).
    2. This module trains a Logistic Regression classifier on those pairs.
    3. To rank a new list of victims, every pair is compared using the
       trained model, and victims are ranked by total "wins".

Usage:
    from pairwise_ranker import PairwiseRanker

    ranker = PairwiseRanker()
    ranker.train(scenarios)          # list of victim-lists (each with ground_truth_priority)
    ranker.save("ranker_model.joblib")

    ranker = PairwiseRanker.load("ranker_model.joblib")
    ranked = ranker.rank_victims(victims)
"""

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

FEATURES = ["final_confidence", "time_buried_minutes", "accessibility"]


def _feature_vector(victim):
    return np.array([victim[f] for f in FEATURES], dtype=float)


def build_pairwise_dataset(scenarios):
    """
    Converts a list of scenarios (each a list of victim dicts with
    ground_truth_priority) into pairwise training examples.

    For every pair (A, B) within a scenario:
        X = feature_vector(A) - feature_vector(B)
        y = 1 if A should be rescued before B (lower priority number),
            0 otherwise

    Both (A, B) and (B, A) orderings are included so the model learns a
    symmetric decision boundary.

    Returns: (X, y) as numpy arrays.
    """
    X, y = [], []

    for victims in scenarios:
        n = len(victims)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = victims[i], victims[j]
                diff = _feature_vector(a) - _feature_vector(b)
                label = 1 if a["ground_truth_priority"] < b["ground_truth_priority"] else 0
                X.append(diff)
                y.append(label)

    return np.array(X), np.array(y)


class PairwiseRanker:
    """
    Wraps a Logistic Regression model trained on pairwise feature
    differences, plus the ranking logic built on top of it.
    """

    def __init__(self):
        self.model = LogisticRegression()
        self._is_trained = False

    def train(self, scenarios):
        """
        Trains the pairwise model on a list of scenarios.
        Each scenario is a list of victim dicts with ground_truth_priority
        (as produced by generate_scenario.py).
        """
        X, y = build_pairwise_dataset(scenarios)
        self.model.fit(X, y)
        self._is_trained = True

        train_accuracy = self.model.score(X, y)
        print(f"Pairwise model trained on {len(X)} pairs, "
              f"train accuracy: {train_accuracy:.2f}")

    def _pairwise_win_probability(self, victim_a, victim_b):
        """Returns P(victim_a should be rescued before victim_b)."""
        diff = (_feature_vector(victim_a) - _feature_vector(victim_b)).reshape(1, -1)
        return self.model.predict_proba(diff)[0][1]

    def rank_victims(self, victims):
        """
        Ranks a list of victims by comparing every pair and summing up
        how many "wins" (predicted higher priority) each victim gets.
        Victims with more wins are ranked higher.

        Returns: the same list of dicts, each with an added
        "predicted_score" and "predicted_rank" field, sorted by rank.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() or load() first.")

        n = len(victims)
        win_scores = [0.0] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                win_scores[i] += self._pairwise_win_probability(victims[i], victims[j])

        for i, v in enumerate(victims):
            v["predicted_score"] = round(win_scores[i] / (n - 1), 3) if n > 1 else 0.0

        ranked = sorted(victims, key=lambda v: v["predicted_score"], reverse=True)
        for rank, v in enumerate(ranked, start=1):
            v["predicted_rank"] = rank

        return ranked

    def save(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"Ranker model saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        ranker = cls()
        ranker.model = joblib.load(filepath)
        ranker._is_trained = True
        return ranker


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scenario_generator"))
    from generate_scenario import generate_scenario  # noqa: E402

    # Quick smoke test: train on a handful of synthetic scenarios, then
    # rank a fresh one and compare to ground truth.
    training_scenarios = [generate_scenario(n_victims=5, seed=s) for s in range(20)]

    ranker = PairwiseRanker()
    ranker.train(training_scenarios)

    test_victims = generate_scenario(n_victims=5, seed=999)
    ranked = ranker.rank_victims(test_victims)

    print("\n" + "=" * 70)
    print("Pairwise Ranker - Test Result")
    print("=" * 70)
    for v in ranked:
        print(
            f"{v['victim_id']}  predicted_rank={v['predicted_rank']}  "
            f"ground_truth_rank={v['ground_truth_priority']}"
        )