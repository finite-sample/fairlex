"""Demonstration of fairlex on a small synthetic survey.

This example illustrates how to construct a membership matrix and target
totals, call the weight‐fair leximin calibration, and inspect the resulting
weights and diagnostics.
"""

import numpy as np

from fairlex import evaluate_solution, leximin_weight_fair


def main() -> None:
    # Suppose we survey five people and want to calibrate on sex and age.
    # Each margin is represented by two rows: the indicator for the
    # first category and the second category. We also include a total row.
    A = np.array([
        # sex: female
        [1, 0, 1, 0, 1],
        # sex: male
        [0, 1, 0, 1, 0],
        # age: young (<=40)
        [1, 1, 0, 0, 1],
        # age: old (>40)
        [0, 0, 1, 1, 0],
        # total
        [1, 1, 1, 1, 1],
    ], dtype=float)
    # Base weights (e.g. equal weights in a simple random sample)
    w0 = np.ones(5)
    # Target totals for the population (feasible with max weight 2.0 per person)
    # Note: with 5 people and max weight 2.0, total achievable is 10
    target = np.array([6, 4, 6, 4, 10], dtype=float)
    # Perform calibration with modest bounds
    result = leximin_weight_fair(A, target, w0, min_ratio=0.5, max_ratio=2.0)
    print("Calibrated weights:", result.w)
    # Evaluate solution quality
    metrics = evaluate_solution(A, target, result.w, base_weights=w0)
    print("Diagnostics:\n", metrics)


if __name__ == "__main__":
    main()
