"""Tests for fairlex metrics module."""

import numpy as np
import pytest

from fairlex.metrics import design_effect, effective_sample_size, evaluate_solution

# Effective sample size


def test_equal_weights():
    assert np.isclose(effective_sample_size(np.array([1, 1, 1, 1])), 4.0)


def test_zero_weights():
    assert np.isnan(effective_sample_size(np.array([0, 0, 0])))


def test_mixed_weights():
    # (2+1+1)^2 / (4+1+1) = 16/6
    assert np.isclose(effective_sample_size(np.array([2, 1, 1])), 16.0 / 6.0)


def test_single_weight():
    assert np.isclose(effective_sample_size(np.array([5])), 1.0)


# Design effect


def test_equal_weights_deff():
    assert np.isclose(design_effect(np.array([1, 1, 1, 1])), 1.0)


def test_zero_weights_deff():
    assert np.isnan(design_effect(np.array([0, 0, 0])))


def test_variable_weights_deff():
    # n = 3, ESS = 16/6, so deff = 3 / (16/6) = 9/8
    assert np.isclose(design_effect(np.array([2, 1, 1])), 9.0 / 8.0)


# Solution evaluation


def test_perfect_solution():
    A = np.array([[1, 0], [0, 1], [1, 1]])
    b = np.array([2, 3, 5])
    w = np.array([2, 3])

    result = evaluate_solution(A, b, w)

    assert np.isclose(result["resid_max_abs"], 0.0)
    assert np.isclose(result["resid_max_rel"], 0.0)
    assert np.isclose(result["resid_median"], 0.0)
    assert "total_error" not in result
    assert result["ESS"] > 0
    assert result["deff"] > 0


def test_relative_residual_skips_zero_targets():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([4, 0])
    w = np.array([3, 1])

    result = evaluate_solution(A, b, w)

    assert np.isclose(result["resid_max_abs"], 1.0)
    assert np.isclose(result["resid_max_rel"], 0.25)


def test_with_base_weights():
    A = np.array([[1, 1]])
    b = np.array([5])
    w = np.array([2, 3])

    result = evaluate_solution(A, b, w, base_weights=np.array([1, 1]))

    # |2-1|/1 = 1, |3-1|/1 = 2
    assert np.isclose(result["max_rel_dev"], 2.0)
    assert np.isclose(result["median_rel_dev"], 1.5)


def test_zero_base_weights_are_counted_not_divided():
    A = np.array([[1, 1]])
    b = np.array([5])

    moved = evaluate_solution(A, b, np.array([2, 3]), base_weights=np.array([0, 1]))
    kept = evaluate_solution(A, b, np.array([0, 3]), base_weights=np.array([0, 1]))

    assert np.isclose(moved["max_rel_dev"], 2.0)
    assert moved["n_zero_base_moved"] == 1
    assert np.isclose(kept["max_rel_dev"], 2.0)
    assert kept["n_zero_base_moved"] == 0


def test_quantile_keys_follow_request():
    """Audit probe P6: quantiles=(0.9,) used to be reported as weight_p99."""
    A = np.ones((1, 100))
    b = np.array([5050.0])
    w = np.arange(1, 101, dtype=float)

    result = evaluate_solution(A, b, w, quantiles=(0.9, 0.975, 0.1))

    assert np.isclose(result["weight_p90"], np.quantile(w, 0.9))
    assert np.isclose(result["weight_p97.5"], np.quantile(w, 0.975))
    assert np.isclose(result["weight_p10"], np.quantile(w, 0.1))
    assert "weight_p99" not in result
    assert "weight_p95" not in result


def test_residual_calculations():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    w = np.array([1.5, 2.5])

    result = evaluate_solution(A, b, w)

    assert np.isclose(result["resid_max_abs"], 0.5)
    assert np.isclose(result["resid_median"], 0.5)
    assert np.isclose(result["resid_p95"], 0.5)
    assert np.isclose(result["resid_max_rel"], 0.5)


def test_weight_statistics():
    A = np.array([[1, 1, 1, 1]])
    b = np.array([10])
    w = np.array([1, 2, 3, 4])

    result = evaluate_solution(A, b, w)

    assert np.isclose(result["weight_min"], 1.0)
    assert np.isclose(result["weight_max"], 4.0)
    assert np.isclose(result["weight_p50"], 2.5)
    assert np.isclose(result["weight_p99"], np.quantile(w, 0.99))
    # ESS = 100/30; deff = 4 / ESS = 1.2
    assert np.isclose(result["ESS"], 100.0 / 30.0)
    assert np.isclose(result["deff"], 1.2)


@pytest.mark.parametrize(
    ("w", "match"),
    [([np.nan, 1.0], "finite"), ([-1.0, 1.0], "non-negative")],
)
def test_invalid_weights_raise(w, match):
    with pytest.raises(ValueError, match=match):
        evaluate_solution(np.array([[1, 1]]), np.array([1.0]), np.array(w))


def test_invalid_quantile_raises():
    with pytest.raises(ValueError, match="quantiles"):
        evaluate_solution(
            np.array([[1, 1]]), np.array([1.0]), np.ones(2), quantiles=(1.5,)
        )
