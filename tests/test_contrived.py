"""Contrived problems whose leximin answer is known in closed form."""

import numpy as np
import pytest

from fairlex.calibration import leximin_weights

ACC = 1e-5


@pytest.mark.parametrize("seed", range(10))
def test_two_groups_and_a_conflicting_total(seed):
    """Groups of targets t1, t2 and a total T != t1 + t2, any scale s.

    Leximin equalises the scaled misses: the groups fall short by delta * s_g
    and the total overshoots by delta * s_T, with
    delta = (t1 + t2 - T) / (s1 + s2 + sT). Caps are loose enough not to bind.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = rng.integers(5, 40, size=2)
    group = np.r_[np.ones(n1), np.zeros(n2)]
    A = np.vstack([group, 1 - group, np.ones(n1 + n2)])
    w0 = rng.uniform(0.5, 2.0, n1 + n2)
    t1, t2 = A[:2] @ w0 * rng.uniform(0.9, 1.1, 2)
    total = (t1 + t2) * rng.uniform(0.9, 0.98)
    b = np.array([t1, t2, total])
    s = rng.uniform(0.5, 2.0, 3) * b

    result = leximin_weights(A, b, w0, min_ratio=0.1, max_ratio=10.0, s=s)

    delta = (t1 + t2 - total) / s.sum()
    expected = np.array([-delta * s[0], -delta * s[1], delta * s[2]])
    assert np.allclose(result.residuals, expected, atol=ACC * b.max())


@pytest.mark.parametrize(("p", "q"), [(3.0, 4.0), (10.0, 30.0), (1.0, 1.5)])
def test_one_unit_two_contradictory_targets(p, q):
    """Relative scale gives the harmonic mean; absolute scale the midpoint."""
    A = np.array([[1.0], [1.0]])
    b = np.array([p, q])
    w0 = np.array([(p + q) / 2])

    relative = leximin_weights(A, b, w0, min_ratio=0.01, max_ratio=100.0)
    absolute = leximin_weights(
        A, b, w0, min_ratio=0.01, max_ratio=100.0, s=np.ones(len(b))
    )

    assert np.isclose(relative.w[0], 2 * p * q / (p + q), rtol=ACC)
    assert np.isclose(absolute.w[0], (p + q) / 2, rtol=ACC)


def test_unreachable_target_pins_group_at_its_cap():
    """A group needing 5x its base total, capped at 2x, sits exactly at 2x."""
    group = np.r_[np.ones(10), np.zeros(30)]
    A = np.vstack([group, 1 - group])
    w0 = np.ones(40)
    b = np.array([50.0, 30.0])

    result = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0)

    assert np.allclose(result.w[:10], 2.0, atol=ACC)
    assert np.allclose(result.w[10:], 1.0, atol=ACC)
    assert np.isclose(result.residuals[0], 20.0 - 50.0, atol=1e-3)


def test_weight_stage_leaves_an_unconstrained_cell_alone():
    """Four sex-by-age cells; women need +20%, young people stay at 1.

    Holding the misses at zero forces women-young and women-old to +20%
    (to share the women's increase at the smallest maximum) and men-young to
    -20% (to keep the young total). Men-old is in no margin, so min-max alone
    leaves it anywhere in [0.8, 1.2]; leximin must leave it at 1.
    """
    # Columns: women-young, women-old, men-young, men-old; three units each.
    cell = np.repeat([0, 1, 2, 3], 3)
    women = np.isin(cell, [0, 1]).astype(float)
    young = np.isin(cell, [0, 2]).astype(float)
    A = np.vstack([women, young])
    w0 = np.ones(12)
    b = np.array([6 * 1.2, 6.0])

    result = leximin_weights(A, b, w0, min_ratio=0.5, max_ratio=2.0)

    ratio = result.w / w0
    assert result.epsilon < ACC
    assert np.allclose(ratio[cell == 0], 1.2, atol=ACC)
    assert np.allclose(ratio[cell == 1], 1.2, atol=ACC)
    assert np.allclose(ratio[cell == 2], 0.8, atol=ACC)
    assert np.allclose(ratio[cell == 3], 1.0, atol=ACC)


def test_nested_consistent_margins_are_hit_exactly():
    """Group inside total, consistent and reachable: both hit, others equal."""
    rng = np.random.default_rng(3)
    n = 60
    group = (rng.random(n) < 0.3).astype(float)
    A = np.vstack([group, np.ones(n)])
    w0 = np.ones(n)
    b = np.array([group.sum() * 1.4, n * 1.1])

    result = leximin_weights(A, b, w0)

    assert result.epsilon < ACC
    inside, outside = result.w[group == 1], result.w[group == 0]
    assert np.allclose(inside, inside[0], atol=ACC)
    assert np.allclose(outside, outside[0], atol=ACC)
