"""Metric and summary utilities for fairlex.

This module contains helper functions to assess the quality of calibrated
weights. The primary entry point, :func:`evaluate_solution`, returns a
dictionary of commonly used diagnostics:

* ``resid_max_abs`` – the maximum absolute residual across all margins.
* ``resid_p95`` – the 95th percentile of absolute residuals.
* ``resid_median`` – the median absolute residual.
* ``ESS`` – the Kish effective sample size of the weights.
* ``deff`` – design effect due to the weights (n / ESS).
* ``weight_max`` – maximum weight.
* ``weight_p99`` – 99th percentile of weights.
* ``weight_p95`` – 95th percentile of weights.
* ``weight_median`` – median weight.
* ``weight_min`` – minimum weight.
* ``total_error`` – the difference between the weighted total and the target
  total in the last margin (assumed to be the sum of membership for the
  entire population).

Additional convenience functions are provided for computing the effective
sample size and design effect alone.

"""

from __future__ import annotations

import numpy as np

__all__ = [
    "effective_sample_size",
    "design_effect",
    "evaluate_solution",
]


def effective_sample_size(weights: np.ndarray) -> float:
    r"""Compute the Kish effective sample size.

    The Kish effective sample size is defined as

    .. math::

        \mathrm{ESS} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.

    Parameters
    ----------
    weights : ndarray
        Array of weights.

    Returns
    -------
    float
        Effective sample size. Returns ``np.nan`` if the denominator is zero.
    """
    w = np.asarray(weights, dtype=float)
    numer = np.sum(w)
    denom = np.sum(w * w)
    if denom == 0:
        return np.nan
    return float((numer * numer) / denom)


def design_effect(weights: np.ndarray) -> float:
    """Compute the design effect due to weighting.

    The design effect is given by ``n / ESS`` where ``n`` is the number of
    observations. It quantifies the inflation in variance attributable to
    unequal weights.

    Parameters
    ----------
    weights : ndarray
        Array of weights.

    Returns
    -------
    float
        Design effect. Returns ``np.nan`` if the effective sample size is
        undefined.
    """
    w = np.asarray(weights, dtype=float)
    ess = effective_sample_size(w)
    if np.isnan(ess) or ess == 0:
        return np.nan
    return float(len(w) / ess)


def evaluate_solution(
    A: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.99, 0.95, 0.5),
    base_weights: np.ndarray | None = None,
) -> dict:
    """Compute summary diagnostics for a calibration solution.

    Parameters
    ----------
    A : ndarray
        Membership matrix of shape ``(m, n)`` used in the calibration.
    b : ndarray
        Target totals of shape ``(m,)``.
    w : ndarray
        Calibrated weights of shape ``(n,)``.
    quantiles : tuple of float, optional
        Quantiles to compute on the weight distribution. Defaults to
        ``(0.99, 0.95, 0.5)``, corresponding to the 99th percentile, 95th
        percentile and median.
    base_weights : ndarray, optional
        Original/base weights. If provided, relative deviations will be
        computed and returned under the keys ``max_rel_dev``, ``p95_rel_dev``,
        and ``median_rel_dev``.

    Returns
    -------
    dict
        A dictionary containing residual and weight diagnostics. See module
        docstring for the key descriptions.
    """
    w = np.asarray(w, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    # Residuals
    resid = A @ w - b
    abs_resid = np.abs(resid)
    resid_max_abs = float(np.max(abs_resid))
    resid_p95 = float(np.percentile(abs_resid, 95))
    resid_median = float(np.median(abs_resid))
    # Weight summary
    q_vals = np.quantile(w, quantiles + (0.0, 1.0))  # compute tails for min and max
    # quantiles returned in ascending order; append min and max at the end
    # Extract weight quantiles by position
    # Example: for quantiles (0.99, 0.95, 0.5) we get q_vals[0]=0th?? Actually quantiles appended in order: (0.99, 0.95, 0.5, 0.0, 1.0)
    # We'll map them explicitly.
    # Build a dict for clarity
    q_map = {q: v for q, v in zip(quantiles + (0.0, 1.0), q_vals)}
    weight_max = float(q_map[1.0])
    weight_min = float(q_map[0.0])
    # Sort the requested quantiles for deterministic mapping
    sorted_q = sorted(quantiles, reverse=True)
    # weight_p99 corresponds to highest quantile (e.g., 0.99) if present
    weight_p99 = float(q_map.get(0.99, q_map[sorted_q[0]]))
    # weight_p95 corresponds to 0.95 or next available
    weight_p95 = float(q_map.get(0.95, q_map[sorted_q[min(1, len(sorted_q) - 1)]]))
    # weight_median corresponds to 0.5 or median quantile in list
    weight_median = float(q_map.get(0.5, np.median(w)))
    ess = effective_sample_size(w)
    deff = design_effect(w)
    total_error = float((A[-1] @ w) - b[-1]) if A.shape[0] > 0 else float('nan')
    result = {
        "resid_max_abs": resid_max_abs,
        "resid_p95": resid_p95,
        "resid_median": resid_median,
        "ESS": float(ess),
        "deff": float(deff),
        "weight_max": weight_max,
        "weight_p99": weight_p99,
        "weight_p95": weight_p95,
        "weight_median": weight_median,
        "weight_min": weight_min,
        "total_error": total_error,
    }
    if base_weights is not None:
        bw = np.asarray(base_weights, dtype=float)
        rel_dev = np.abs(w - bw) / np.where(bw == 0, 1.0, np.abs(bw))
        result.update(
            max_rel_dev=float(np.max(rel_dev)),
            p95_rel_dev=float(np.percentile(rel_dev, 95)),
            median_rel_dev=float(np.median(rel_dev)),
        )
    return result