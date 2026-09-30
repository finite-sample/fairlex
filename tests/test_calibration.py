"""Tests for fairlex calibration algorithms."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, linprog

from fairlex import calibration
from fairlex.calibration import leximin_weights

# Levels are held with a relative tolerance of 3e-7 per round (above HiGHS's
# 1e-7 feasibility tolerance), so results are exact to about 1e-5 of the
# target scale rather than to machine precision.
ACC = 1e-5

# Input validation (what a user can get wrong through fairlex.calibrate)


@pytest.mark.parametrize(
    ("b", "w0", "kwargs", "match"),
    [
        ([np.inf], [1.0, 1.0], {}, "targets"),
        ([1.0], [1.0, np.nan], {}, "base weights"),
        ([1.0], [-1.0, 1.0], {}, "base weights"),
        ([1.0], [1.0, 1.0], {"min_ratio": 2, "max_ratio": 1}, "bounds"),
        ([1.0], [1.0, 1.0], {"min_ratio": -0.5}, "bounds"),
        ([1.0], [1.0, 1.0], {"max_ratio": np.inf}, "bounds"),
        ([0.0], [1.0, 1.0], {}, "scale"),
        ([1.0], [1.0, 1.0], {"s": [0.0]}, "scale"),
        ([1.0], [1.0, 1.0], {"s": [np.nan]}, "scale"),
        ([1.0], [1.0, 1.0], {"slack": -1.0}, "slack"),
    ],
)
def test_invalid_inputs_raise(b, w0, kwargs, match):
    with pytest.raises(ValueError, match=match):
        leximin_weights(np.array([[1.0, 1.0]]), np.array(b), np.array(w0), **kwargs)


def test_zero_target_allowed_with_its_own_scale():
    result = leximin_weights(
        np.array([[1.0, 1.0]]), np.array([0.0]), np.ones(2), s=np.ones(1)
    )
    assert result.status == 0


# Simple cases


def test_single_variable_single_constraint():
    result = leximin_weights(
        np.array([[1.0]]), np.array([2.0]), np.array([1.0]), min_ratio=0.5
    )

    assert result.status == 0
    assert np.isclose(result.w[0], 2.0, atol=ACC)
    assert result.epsilon < ACC
    assert np.isclose(result.t, 1.0, atol=ACC)


def test_exactly_feasible_problem():
    A = np.array([[1, 0], [0, 1], [1, 1]])
    b = np.array([3, 2, 5])

    result = leximin_weights(A, b, np.ones(2))

    assert result.status == 0
    assert np.allclose(result.w, [3, 2], atol=ACC)
    assert np.allclose(result.residuals, 0.0, atol=10 * ACC)


def test_conflicting_targets_absolute():
    A = np.array([[1, 0], [1, 0]])
    b = np.array([3, 4])

    result = leximin_weights(A, b, np.ones(2), s=np.ones(len(b)))

    assert np.isclose(result.w[0], 3.5)
    assert np.isclose(result.epsilon, 0.5)


def test_conflicting_targets_relative():
    """Relative scale equalises percentage misses: |x-3|/3 = |x-4|/4 at 24/7."""
    A = np.array([[1, 0], [1, 0]])
    b = np.array([3, 4])

    result = leximin_weights(A, b, np.ones(2))

    assert np.isclose(result.w[0], 24 / 7)
    assert np.isclose(result.epsilon, 1 / 7)


def test_empty_problem():
    result = leximin_weights(np.zeros((0, 0)), np.array([]), np.array([]))
    assert result.status == 0
    assert len(result.w) == 0


def test_no_margins_keeps_base_weights():
    w0 = np.array([1.0, 2.0])
    result = leximin_weights(np.zeros((0, 2)), np.array([]), w0)
    assert np.allclose(result.w, w0)
    assert np.isclose(result.t, 0.0)


def test_zero_base_weights():
    A = np.array([[1, 1]])
    b = np.array([2])
    w0 = np.array([0, 1])

    for fn in (leximin_weights,):
        result = fn(A, b, w0, min_ratio=0.5, max_ratio=2.0)
        assert result.status == 0
        assert np.isclose(result.w[0], 0)
        assert np.isclose(result.w[1], 2)


# Leximin correctness


def test_second_worst_margin_is_minimised():
    """Audit probe P1: min-max alone left margin 3 at 4.5 when 3 is reachable.

    Margin 1 needs unit 0 at 10 but it is capped at 2 (miss 8, unavoidable).
    Margin 2 is then at best 6. Margin 3 needs unit 1 at 5, capped at 2, so
    the best achievable miss is 3.
    """
    A = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.array([10, 10, 5], dtype=float)

    result = leximin_weights(
        A, b, np.ones(3), min_ratio=0.5, max_ratio=2.0, s=np.ones(len(b))
    )

    assert np.allclose(np.abs(result.residuals), [8, 6, 3])
    assert np.isclose(result.epsilon, 8)


def _ordered_outcome_leximin(a_scaled, c, lb, ub, extra=None):
    """Leximin vector of |a_scaled w - c| via Ogryczak's ordered-outcome method.

    Independent of fairlex's saturation loop: it lexicographically minimises
    the cumulative sums of the k largest absolute residuals, k = 1..m, using
    top_k(e) = min_u k*u + sum_j max(0, e_j - u). The successive differences
    of those optimal sums are the leximin vector in descending order.
    ``extra = (rows, rhs)`` adds constraints ``rows @ w <= rhs`` on ``w``.
    """
    m, n = a_scaled.shape
    thetas = []
    for k in range(1, m + 1):
        # Variables: w (n), e (m), then (u_q, v_q (m)) for q = 1..k.
        nv = n + m + k * (1 + m)
        rows, rhs = [], []
        for j in range(m):
            for sign in (1.0, -1.0):
                row = np.zeros(nv)
                row[:n] = sign * a_scaled[j]
                row[n + j] = -1.0
                rows.append(row)
                rhs.append(sign * c[j])
        for q in range(1, k + 1):
            u = n + m + (q - 1) * (1 + m)
            for j in range(m):
                row = np.zeros(nv)
                row[n + j] = 1.0
                row[u] = -1.0
                row[u + 1 + j] = -1.0
                rows.append(row)
                rhs.append(0.0)
            if q < k:
                row = np.zeros(nv)
                row[u] = q
                row[u + 1 : u + 1 + m] = 1.0
                rows.append(row)
                # Room above the solver's feasibility tolerance: older HiGHS
                # (SciPy 1.12) reports tighter pins as infeasible.
                rhs.append(thetas[q - 1] * (1 + 1e-6) + 1e-8)
        if extra is not None:
            for row_w, r in zip(*extra, strict=True):
                row = np.zeros(nv)
                row[:n] = row_w
                rows.append(row)
                rhs.append(r)
        obj = np.zeros(nv)
        u = n + m + (k - 1) * (1 + m)
        obj[u] = k
        obj[u + 1 : u + 1 + m] = 1.0
        bounds = (
            list(zip(lb, ub, strict=True))
            + [(0, None)] * m
            + [(None, None), *[(0, None)] * m] * k
        )
        # Interior point, unlike the engine's dual simplex, for a second
        # solver path. Presolve is off because SciPy 1.12's HiGHS presolve
        # wrongly declares some of these programmes infeasible.
        res = linprog(
            obj,
            A_ub=np.array(rows),
            b_ub=rhs,
            bounds=bounds,
            method="highs-ipm",
            options={"presolve": False},
        )
        assert res.success
        thetas.append(res.fun)
    return np.diff(np.concatenate([[0.0], thetas]))


def _levels(A, b, w0, s, lo, hi):
    """Leximin miss levels from the saturation stage alone (before weights)."""
    cells = calibration._cells(A, b, w0, s, lo, hi)
    levels, _, _ = calibration._leximin(cells.coef, cells.target, cells.bounds)
    return levels


def check_against_oracle(A, b, w0, s, lo, hi):
    """Levels match the oracle; final misses never exceed their levels.

    The saturation stage computes the leximin miss vector; the oracle must
    agree with it. The weight stage then only promises to keep each margin at
    or below its level (plus the ~1e-6 tolerance), and may leave lower-ranked
    margins a little below theirs, so the final misses are checked against
    that promise rather than compared with the oracle.
    """
    levels = _levels(A, b, w0, s, lo, hi)
    expected = _ordered_outcome_leximin(A / s[:, None], b / s, w0 * lo, w0 * hi)
    scale = 1 + np.abs(b / s).max()
    assert np.allclose(np.sort(levels)[::-1], expected, atol=ACC * scale)

    result = leximin_weights(A, b, w0, s, min_ratio=lo, max_ratio=hi)
    final = np.abs(result.residuals) / s
    assert np.all(final <= levels + ACC * scale)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("scale", ["relative", "absolute"])
def test_matches_independent_leximin(seed, scale):
    """Leximin levels match the ordered-outcome oracle under two scales."""
    rng = np.random.default_rng(seed)
    m, n = rng.integers(2, 7), rng.integers(3, 12)
    A = (rng.random((m, n)) < 0.5).astype(float)
    A[:, rng.integers(n)] = 1.0
    w0 = rng.uniform(0.5, 3.0, n)
    b = A @ w0 * rng.uniform(0.4, 2.5, m)
    s = np.abs(b) if scale == "relative" else np.ones(m)

    check_against_oracle(A, b, w0, s, 0.5, 2.0)


def test_relative_scale_protects_small_groups():
    """Inconsistent targets: a 30-unit group, its complement and their total.

    The targets imply 60 + 970 = 1030 but the total says 1000, so 30 units
    of miss must go somewhere. Absolute scale splits them evenly (10 each),
    which is a 17% miss for the small group; relative scale equalises the
    percentage misses at 30 / 2030.
    """
    n, n_g = 1000, 30
    group = np.r_[np.ones(n_g), np.zeros(n - n_g)]
    A = np.vstack([np.ones(n), group, 1 - group])
    b = np.array([1000.0, 60.0, 970.0])
    w0 = np.r_[np.full(n_g, 2.0), np.ones(n - n_g)]

    absolute = leximin_weights(A, b, w0, s=np.ones(len(b)))
    relative = leximin_weights(A, b, w0)

    assert np.allclose(np.abs(absolute.residuals), 10.0, atol=ACC * b.max())
    assert np.abs(absolute.residuals[1]) / 60 > 0.16
    assert np.allclose(np.abs(relative.residuals) / b, 30 / 2030, atol=ACC)


def test_scale_array_equal_to_targets_matches_relative():
    A, b, w0 = _conflicting_margins()

    relative = leximin_weights(A, b, w0)
    explicit = leximin_weights(A, b, w0, s=np.abs(b))

    assert np.allclose(relative.residuals, explicit.residuals, atol=ACC)
    assert np.isclose(relative.epsilon, explicit.epsilon, atol=ACC)


def test_scale_array_sets_margin_priority():
    """Doubling s_j tolerates twice the miss on margin j at the same priority.

    The misses must sum to 30 across the three margins. With s = |b| they
    are equal percentages; doubling the group's s lets it take twice the
    percentage miss of the others.
    """
    A, b, w0 = _conflicting_margins()
    s = np.abs(b) * np.array([1.0, 2.0, 1.0])

    result = leximin_weights(A, b, w0, s=s)

    scaled = np.abs(result.residuals) / s
    # The weight stage may leave a margin a hair under its level (~1e-5).
    assert np.allclose(scaled, scaled[0], atol=2 * ACC)
    assert np.isclose(np.abs(result.residuals).sum(), 30.0, atol=1e-3)
    assert np.isclose(
        np.abs(result.residuals[1]) / 60,
        2 * np.abs(result.residuals[0]) / 1000,
        atol=4 * ACC,
    )


def _conflicting_margins():
    """Group of 30 (target 60) + complement (target 970) vs total 1000."""
    n, n_g = 1000, 30
    group = np.r_[np.ones(n_g), np.zeros(n - n_g)]
    A = np.vstack([np.ones(n), group, 1 - group])
    w0 = np.r_[np.full(n_g, 2.0), np.ones(n - n_g)]
    return A, np.array([1000.0, 60.0, 970.0]), w0


# Weight-fair stage


def test_weight_fair_basic():
    A = np.array([[1, 0], [0, 1], [1, 1]])
    b = np.array([3, 2, 5])

    result = leximin_weights(A, b, np.ones(2))

    assert result.status == 0
    assert np.allclose(result.w, [3, 2], atol=ACC)
    assert result.epsilon < ACC
    assert np.isclose(result.t, 2.0, atol=ACC)


def test_weight_fair_keeps_leximin_residual_profile():
    A = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.array([10, 10, 5], dtype=float)

    result = leximin_weights(
        A, b, np.ones(3), min_ratio=0.5, max_ratio=2.0, s=np.ones(len(b))
    )

    assert np.allclose(np.abs(result.residuals), [8, 6, 3], atol=1e-6)


def test_epsilon_reports_achieved_residual_with_slack():
    """Audit probe P2: epsilon was the stage-1 value (0) while the miss was 0.5."""
    A = np.array([[1, 1, 0], [0, 1, 1]], dtype=float)
    b = np.array([3.0, 3.0])

    result = leximin_weights(A, b, np.ones(3), slack=0.5, s=np.ones(len(b)))

    achieved = np.max(np.abs(A @ result.w - b))
    assert achieved > 0.1
    assert np.isclose(result.epsilon, achieved)
    assert np.allclose(result.residuals, A @ result.w - b)
    assert np.isclose(result.t, np.max(np.abs(result.w - 1.0)))


def test_weight_fair_spreads_changes_evenly():
    """Group A needs +50%, group B +5%, group C nothing.

    Min-max alone moved every unit by 50% (deff 1.235); min-max then
    min-sum moved 10% of group B by 50% and left the rest (deff 1.013).
    Leximin over changes moves all of A by 50%, all of B by 5%, none of C.
    """
    sizes = (20, 400, 580)
    grp = np.repeat([0, 1, 2], sizes)
    A = np.vstack([grp == 0, grp == 1, grp == 2]).astype(float)
    b = np.array([20 * 1.5, 400 * 1.05, 580.0])
    w0 = np.ones(grp.size)

    result = leximin_weights(A, b, w0, min_ratio=0.3, max_ratio=3.0)

    change = np.abs(result.w - w0)
    assert result.epsilon < ACC
    assert np.allclose(change[grp == 0], 0.5, atol=ACC)
    assert np.allclose(change[grp == 1], 0.05, atol=ACC)
    assert np.allclose(change[grp == 2], 0.0, atol=ACC)
    assert np.isclose(result.t, 0.5, atol=ACC)


def test_weight_fair_unequal_base_weights_share_ratio():
    """Units in one cell get the same ratio w / w0 whatever their w0."""
    A = np.array([[1.0, 1.0, 1.0]])
    w0 = np.array([1.0, 2.0, 3.0])

    result = leximin_weights(A, np.array([7.2]), w0)

    assert np.allclose(result.w / w0, 1.2)


def test_weight_fair_scales_to_large_n():
    """The old dense weight stage needed ~40 GB here."""
    rng = np.random.default_rng(2)
    n, m = 50_000, 5
    A = np.vstack([np.ones(n), (rng.random((m - 1, n)) < 0.3).astype(float)])
    w0 = rng.uniform(0.5, 2.0, n)
    b = A @ w0 * np.array([1.0, 1.05, 0.95, 1.1, 0.9])

    result = leximin_weights(A, b, w0, min_ratio=0.3, max_ratio=3.0)

    assert result.status == 0
    assert result.epsilon < 1e-6


def test_residual_stage_failure_propagates(monkeypatch):
    def fail(*_args, **_kwargs):
        return OptimizeResult(success=False, status=4, message="numerical trouble")

    monkeypatch.setattr(calibration, "_solve_lp", fail)
    A, b, w0 = np.array([[1.0, 1.0]]), np.array([3.0]), np.ones(2)

    for fn in (leximin_weights,):
        result = fn(A, b, w0)
        assert result.status == 4
        assert result.message == "numerical trouble"
        assert np.all(np.isnan(result.w))
        assert np.isnan(result.epsilon)
        assert result.t is None


def test_weight_stage_failure_propagates(monkeypatch):
    real = calibration._solve_lp
    calls = {"n": 0}

    def fail_after_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            return OptimizeResult(success=False, status=2, message="infeasible")
        return real(*args, **kwargs)

    monkeypatch.setattr(calibration, "_solve_lp", fail_after_first)
    result = leximin_weights(np.array([[1.0, 1.0]]), np.array([3.0]), np.ones(2))

    assert result.status == 2
    assert np.all(np.isnan(result.w))
    assert result.t is None


# Numerical range


def test_very_small_weights():
    result = leximin_weights(
        np.array([[1, 1]]), np.array([3e-10]), np.array([1e-10, 1e-10])
    )
    assert result.status == 0
    assert np.isclose(result.w.sum(), 3e-10, rtol=1e-6)


def test_large_weights():
    result = leximin_weights(
        np.array([[1, 1]]), np.array([3e10]), np.array([1e10, 1e10])
    )
    assert result.status == 0
    assert np.isclose(result.w.sum(), 3e10, rtol=1e-6)
