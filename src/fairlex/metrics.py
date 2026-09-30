"""Metric and summary utilities for fairlex.

This module contains helper functions to assess the quality of calibrated
weights. The primary entry point, :func:`evaluate_solution`, returns a
dictionary of commonly used diagnostics:

* ``resid_max_abs`` - the maximum absolute residual across all margins.
* ``resid_max_rel`` - the maximum of ``|residual_j| / |b_j|`` over margins
  with a nonzero target (``NaN`` if every target is zero).
* ``resid_p95`` - the 95th percentile of absolute residuals.
* ``resid_median`` - the median absolute residual.
* ``ESS`` - the Kish effective sample size of the weights.
* ``deff`` - design effect due to the weights (n / ESS).
* ``weight_min`` and ``weight_max`` - the smallest and largest weight.
* ``weight_p{100q}`` - one key per requested quantile ``q``, e.g.
  ``weight_p99`` for 0.99 and ``weight_p97.5`` for 0.975.

Additional convenience functions are provided for computing the effective
sample size and design effect alone.

"""

import numpy as np

__all__ = [
    "design_effect",
    "effective_sample_size",
    "evaluate_solution",
]


def effective_sample_size(weights: np.ndarray) -> float:
    r"""Compute the Kish effective sample size.

    The Kish effective sample size is defined as

    .. math::

        \mathrm{ESS} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.

    Args:
        weights: Array of weights.

    Returns:
        Effective sample size. Returns ``np.nan`` if the denominator is zero.

    """
    w = np.asarray(weights, dtype=float)
    numerator = np.sum(w)
    denom = np.sum(w * w)
    if denom == 0:
        return np.nan
    return float((numerator * numerator) / denom)


def design_effect(weights: np.ndarray) -> float:
    """Compute the design effect due to weighting.

    The design effect is given by ``n / ESS`` where ``n`` is the number of
    observations. It quantifies the inflation in variance attributable to
    unequal weights.

    Args:
        weights: Array of weights.

    Returns:
        Design effect. Returns ``np.nan`` if the effective sample size is
        undefined.

    """
    w = np.asarray(weights, dtype=float)
    ess = effective_sample_size(w)
    if np.isnan(ess) or ess == 0:
        return np.nan
    return float(len(w) / ess)


def _compute_residual_metrics(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
) -> dict[str, float]:
    """Compute residual-based metrics.

    Args:
        A: Membership matrix of shape ``(m, n)``.
        b: Target totals of shape ``(m,)``.
        w: Calibrated weights of shape ``(n,)``.

    Returns:
        Dictionary containing residual metrics.

    """
    resid = A @ w - b
    abs_resid = np.abs(resid)
    nonzero = b != 0
    return {
        "resid_max_abs": float(np.max(abs_resid)),
        "resid_max_rel": float(np.max(abs_resid[nonzero] / np.abs(b[nonzero])))
        if nonzero.any()
        else float("nan"),
        "resid_p95": float(np.percentile(abs_resid, 95)),
        "resid_median": float(np.median(abs_resid)),
    }


def _compute_weight_metrics(
    w: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[str, float]:
    """Compute weight distribution metrics.

    Args:
        w: Calibrated weights of shape ``(n,)``.
        quantiles: Quantiles to compute on the weight distribution.

    Returns:
        Dictionary containing weight distribution metrics.

    Raises:
        ValueError: If two different quantiles round to the same
            ``weight_p*`` key. A repeated quantile just yields its one key.

    """
    # Keys use ``:g`` (six significant digits) so common quantiles read as
    # weight_p99 or weight_p97.5; exact float formatting would print 0.07 as
    # weight_p7.000000000000001. The price is that very close quantiles can
    # round to one key, which is refused rather than silently overwritten.
    keys = {q: f"weight_p{100 * q:g}" for q in quantiles}
    if len(set(keys.values())) < len(keys):
        msg = f"quantiles {quantiles} round to the same key; separate them further"
        raise ValueError(msg)
    result = {"weight_min": float(np.min(w)), "weight_max": float(np.max(w))}
    for q, value in zip(quantiles, np.quantile(w, quantiles), strict=True):
        result[keys[q]] = float(value)
    result["ESS"] = float(effective_sample_size(w))
    result["deff"] = float(design_effect(w))
    return result


def _compute_relative_deviations(
    w: np.ndarray,
    base_weights: np.ndarray,
) -> dict[str, float]:
    """Compute relative deviation metrics.

    Args:
        w: Calibrated weights of shape ``(n,)``.
        base_weights: Original weights the deviations are measured against.

    Returns:
        Dictionary containing relative deviation metrics over units with
        positive base weight, plus ``n_zero_base_moved``: the number of units
        with zero base weight whose weight is no longer zero, for which a
        relative change is undefined.

    """
    bw = np.asarray(base_weights, dtype=float)
    positive = bw > 0
    rel_dev = np.abs(w[positive] - bw[positive]) / bw[positive]
    empty = not positive.any()
    return {
        "max_rel_dev": float("nan") if empty else float(np.max(rel_dev)),
        "p95_rel_dev": float("nan") if empty else float(np.percentile(rel_dev, 95)),
        "median_rel_dev": float("nan") if empty else float(np.median(rel_dev)),
        "n_zero_base_moved": float(np.sum(w[~positive] != 0)),
    }


def evaluate_solution(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.99, 0.95, 0.5),
    base_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute summary diagnostics for a calibration solution.

    Args:
        A: Membership matrix of shape ``(m, n)`` used in the calibration.
        b: Target totals of shape ``(m,)``.
        w: Calibrated weights of shape ``(n,)``.
        quantiles: Quantiles of the weight distribution to report, each under
            the key ``weight_p{100q}``. Defaults to ``(0.99, 0.95, 0.5)``.
        base_weights: Original/base weights. If provided, relative deviations
            will be computed and returned under the keys ``max_rel_dev``,
            ``p95_rel_dev`` and ``median_rel_dev`` (over units with positive
            base weight), with ``n_zero_base_moved`` counting zero-base units
            that received weight.

    Returns:
        A dictionary containing residual and weight diagnostics. See module
        docstring for the key descriptions.

    Raises:
        ValueError: If ``w`` has non-finite or negative entries (e.g. the
            ``NaN`` weights of a failed solve), a quantile is outside
            ``[0, 1]``, or two different quantiles round to the same key.

    """
    w = np.asarray(w, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.all(np.isfinite(w)):
        msg = "w must contain only finite values; did the calibration fail?"
        raise ValueError(msg)
    if np.any(w < 0):
        msg = "w must be non-negative"
        raise ValueError(msg)
    if not all(0 <= q <= 1 for q in quantiles):
        msg = f"quantiles must lie in [0, 1], got {quantiles}"
        raise ValueError(msg)

    result = _compute_residual_metrics(A, b, w)
    result.update(_compute_weight_metrics(w, quantiles))

    if base_weights is not None:
        result.update(_compute_relative_deviations(w, base_weights))

    return result
