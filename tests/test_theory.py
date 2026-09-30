"""Numerical checks of the claims on the theory page (docs/theory.rst).

Each check reaches the claimed quantity by a route that shares no code with
the calibration solver: the worst-case bias is found by its own linear
programme, capped post-stratification is computed in closed form, and so on.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from fairlex.calibration import leximin_weights
from fairlex.metrics import design_effect

ACC = 1e-5


def _worst_case_bias(r, s):
    """Max of r @ beta over the ball sum_j s_j |beta_j| <= 1, by LP.

    beta = u - v with u, v >= 0; maximise r @ (u - v) s.t. s @ (u + v) <= 1.
    """
    m = len(r)
    res = linprog(
        -np.r_[r, -r],
        A_ub=np.r_[s, s][None, :],
        b_ub=[1.0],
        bounds=[(0, None)] * (2 * m),
    )
    assert res.success
    return -res.fun


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 300))
    cols = []
    for k in rng.integers(2, 5, size=int(rng.integers(2, 4))):
        cat = rng.integers(k, size=n)
        cols += [cat == j for j in range(k)]
    A = np.vstack([np.array(cols, float), np.ones(n)])
    w0 = rng.uniform(0.5, 2.0, n)
    b = A @ w0 * rng.uniform(0.6, 1.6, A.shape[0])
    return A, b, w0


@pytest.mark.parametrize("seed", range(20))
def test_epsilon_is_worst_case_bias(seed):
    """Proposition 1: epsilon equals the worst-case bias over the s-ball."""
    A, b, w0 = _random_problem(seed)
    s = np.random.default_rng(seed + 1000).uniform(0.5, 3.0, len(b)) * np.abs(b)

    result = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0, s=s)

    assert np.isclose(_worst_case_bias(result.residuals, s), result.epsilon)


def test_relative_scale_ignores_units_of_each_margin():
    """Relative mode: multiplying one margin's row and target by c changes nothing."""
    A, b, w0 = _random_problem(3)
    rows_rescaled, b2 = A.copy(), b.copy()
    rows_rescaled[1] *= 1000.0
    b2[1] *= 1000.0

    r1 = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0)
    r2 = leximin_weights(rows_rescaled, b2, w0, min_ratio=0.5, max_ratio=2.0)

    assert np.allclose(np.abs(r1.residuals) / b, np.abs(r2.residuals) / b2, atol=ACC)


def test_absolute_scale_depends_on_units_of_each_margin():
    """The contrast: in absolute mode the same rescaling changes the misses."""
    A, b, w0 = _random_problem(3)
    rows_rescaled, b2 = A.copy(), b.copy()
    rows_rescaled[1] *= 1000.0
    b2[1] *= 1000.0

    r1 = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0, s=np.ones(len(b)))
    r2 = leximin_weights(
        rows_rescaled, b2, w0, min_ratio=0.5, max_ratio=2.0, s=np.ones(len(b))
    )

    assert not np.allclose(
        np.abs(r1.residuals) / b, np.abs(r2.residuals) / b2, atol=1e-3
    )


def test_single_variable_is_capped_post_stratification():
    """One categorical variable: each category's ratio is clip(b_k / W_k)."""
    rng = np.random.default_rng(7)
    n, k = 500, 4
    cat = rng.integers(k, size=n)
    A = np.array([cat == j for j in range(k)], float)
    w0 = rng.uniform(0.5, 2.0, n)
    mass = A @ w0
    b = mass * np.array([0.3, 0.9, 1.4, 3.0])
    lo, hi = 0.5, 2.0

    result = leximin_weights(A, b, w0, min_ratio=lo, max_ratio=hi)

    expected_ratio = np.clip(b / mass, lo, hi)[cat]
    assert np.allclose(result.w / w0, expected_ratio, atol=ACC)


@pytest.mark.parametrize("seed", range(10))
def test_weight_stage_bounds_design_effect(seed):
    """Ratios in [1 - t, 1 + t] imply deff <= deff0 * ((1 + t) / (1 - t))^2."""
    A, b, w0 = _random_problem(seed)

    result = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0)

    t = result.t
    assert t is not None
    if t < 1:
        bound = design_effect(w0) * ((1 + t) / (1 - t)) ** 2
        assert design_effect(result.w) <= bound * (1 + ACC)
