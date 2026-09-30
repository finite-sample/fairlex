"""Weight diagnostics used in :class:`fairlex.CalibrationReport`."""

import numpy as np

__all__ = ["design_effect", "effective_sample_size"]


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
