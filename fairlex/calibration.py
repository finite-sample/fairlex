"""Core calibration routines for fairlex.

This module contains implementations of leximin-style calibration for survey
weights. Two variants are provided:

* ``leximin_residual`` minimises the worst absolute deviation between the
  calibrated and target margins (a ``min-max`` problem). It is akin to
  solving a Chebyshev approximation on the residuals. While this drives
  margin errors down, it can lead to large deviations from the original
  weights if the margin targets are difficult to meet within bounds.

* ``leximin_weight_fair`` first performs the residual leximin step and then
  minimises the largest relative change from the base weights, subject to
  keeping the residuals at the optimum level (plus a small optional slack).
  This spreads the adjustments more evenly across units and yields a more
  stable set of weights.

Both functions accept a membership matrix ``A`` of shape ``(m, n)`` where
``m`` is the number of margins and ``n`` is the number of units. Each row
should correspond to a margin (e.g., a demographic group), and the entries
indicate whether a unit belongs to that margin (1.0 or 0.0, or continuous
weights for soft membership). The target totals ``b`` must be of length
``m``. The base weights ``w0`` must be of length ``n``.

Weight bounds are specified as multiplicative factors relative to ``w0``.
For example, ``min_ratio=0.5`` and ``max_ratio=2.0`` constrains each
calibrated weight to lie between half and twice its original value.

The underlying optimisation problems are solved via ``scipy.optimize.linprog``
using the HiGHS solvers. If SciPy is unavailable, attempting to call these
functions will raise an informative ``ImportError``.

"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]

# Constants
EXPECTED_MATRIX_DIMENSIONS = 2


@dataclass
class CalibrationResult:
    """Structured result from a calibration call.

    Attributes
    ----------
    w : ndarray
        Calibrated weights of shape ``(n,)``.
    epsilon : float
        The worst absolute residual achieved in the residual minimisation
        problem. Only meaningful for ``leximin_residual``.
    t : Optional[float]
        The worst relative weight change achieved in the weight fairness
        problem. ``None`` if only the residual stage is performed.
    status : int
        Status code from the linear programme (0 indicates success).
    message : str
        Solver termination message for diagnostics.

    """

    w: np.ndarray
    epsilon: float
    t: float | None
    status: int
    message: str


def _validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce input arrays to ensure they have compatible shapes.

    Parameters
    ----------
    A : array-like
        Membership matrix of shape ``(m, n)``.
    b : array-like
        Target totals of shape ``(m,)``.
    w0 : array-like
        Base weights of shape ``(n,)``.

    Returns
    -------
    (A, b, w0) : tuple of ndarrays
        Validated and dtype-coerced versions of the inputs.

    Raises
    ------
    ValueError
        If shapes are incompatible.

    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    if A.ndim != EXPECTED_MATRIX_DIMENSIONS:
        msg = f"A must be two-dimensional, got shape {A.shape}"
        raise ValueError(msg)
    m, n = A.shape
    if b.shape != (m,):
        msg = f"b must be of shape {(m,)}, got {b.shape}"
        raise ValueError(msg)
    if w0.shape != (n,):
        msg = f"w0 must be of shape {(n,)}, got {w0.shape}"
        raise ValueError(msg)
    return A, b, w0


def _solve_lp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
) -> "scipy.optimize.OptimizeResult":  # type: ignore[name-defined]  # noqa: F821
    """Solve a linear programming problem using SciPy HiGHS.

    This helper centralises the call to ``scipy.optimize.linprog`` and
    provides a clear error message if SciPy is not installed.

    Parameters
    ----------
    c : ndarray
        Objective coefficients.
    A_ub : ndarray
        Inequality constraint matrix.
    b_ub : ndarray
        Inequality constraint right hand side.
    bounds : sequence of (float, float)
        Variable bounds.

    Returns
    -------
    res : OptimizeResult
        Result from the solver.

    """
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    return res


def leximin_residual(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
) -> CalibrationResult:
    r"""Compute weights by minimising the worst absolute margin residual.

    This function solves

    .. math::

        \min_{w, \epsilon}\;\epsilon
        \quad\text{such that}\quad
        -\epsilon \le A\,w - b \le \epsilon,
        \quad \text{and}\quad
        w_i \in [w_{0,i}\,\text{min\_ratio},\, w_{0,i}\,\text{max\_ratio}].

    Parameters
    ----------
    A : ndarray
        Membership matrix of shape ``(m, n)``.
    b : ndarray
        Target totals of shape ``(m,)``.
    w0 : ndarray
        Base weights of shape ``(n,)``.
    min_ratio : float, optional
        Lower bound on weights relative to ``w0``. Defaults to ``0.1``.
    max_ratio : float, optional
        Upper bound on weights relative to ``w0``. Defaults to ``10.0``.

    Returns
    -------
    CalibrationResult
        Structured result containing the weights, the optimum ``epsilon``, and
        solver diagnostics.

    Notes
    -----
    If the problem is infeasible (e.g., because the bounds preclude any
    solution), the returned status will be nonzero and the weights may not be
    meaningful. Check ``status`` and ``message`` on the result.

    """
    A, b, w0 = _validate_inputs(A, b, w0)
    m, n = A.shape
    # Objective: minimise epsilon
    c = np.zeros(n + 1)
    c[-1] = 1.0
    # Variable bounds: w between w0*min_ratio and w0*max_ratio; epsilon >= 0
    bounds = [(w0[i] * min_ratio, w0[i] * max_ratio) for i in range(n)] + [(0, None)]
    # Construct inequality constraints
    # We will build 2*m inequalities: A_j w - epsilon <= b_j and -A_j w - epsilon <= -b_j
    A_ub = np.zeros((2 * m, n + 1))
    b_ub = np.zeros(2 * m)
    for j in range(m):
        # A_j w - epsilon <= b_j
        A_ub[2 * j, :n] = A[j]
        A_ub[2 * j, -1] = -1.0
        b_ub[2 * j] = b[j]
        # -A_j w - epsilon <= -b_j
        A_ub[2 * j + 1, :n] = -A[j]
        A_ub[2 * j + 1, -1] = -1.0
        b_ub[2 * j + 1] = -b[j]
    res = _solve_lp(c, A_ub, b_ub, bounds)
    if not res.success:
        return CalibrationResult(
            w=np.full_like(w0, np.nan),
            epsilon=np.nan,
            t=None,
            status=res.status,
            message=res.message,
        )
    x = res.x
    w = x[:n]
    epsilon = x[-1]
    return CalibrationResult(
        w=w,
        epsilon=epsilon,
        t=None,
        status=res.status,
        message=res.message,
    )


def _setup_weight_fair_constraints(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    epsilon_opt: float,
    *,
    min_ratio: float,
    max_ratio: float,
    slack: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float | None, float | None]]]:
    """Set up constraints for the weight-fair stage of calibration.

    Returns
    -------
    A_ub : ndarray
        Inequality constraint matrix.
    b_ub : ndarray
        Inequality constraint right hand side.
    bounds : list
        Variable bounds.

    """
    m, n = A.shape

    # Variables: w (n) and t (1)
    # Bounds: w within [w0*min_ratio, w0*max_ratio], t >= 0
    bounds = [(w0[i] * min_ratio, w0[i] * max_ratio) for i in range(n)] + [(0, None)]

    # Build inequality constraints
    # Residual constraints: +/- (A_j w - b_j) <= epsilon_opt + slack
    # We'll build 2*m inequalities of the form A_j w + 0*t <= b_j + epsilon_opt + slack
    # and -A_j w + 0*t <= -b_j + epsilon_opt + slack
    total_constraints = 2 * m + 2 * n  # residual constraints + weight change bounds
    A_ub = np.zeros((total_constraints, n + 1))
    b_ub = np.zeros(total_constraints)

    # Residual constraints
    for j in range(m):
        # A_j w <= b_j + epsilon_opt + slack
        A_ub[2 * j, :n] = A[j]
        A_ub[2 * j, -1] = 0.0
        b_ub[2 * j] = b[j] + epsilon_opt + slack
        # -A_j w <= -b_j + epsilon_opt + slack
        A_ub[2 * j + 1, :n] = -A[j]
        A_ub[2 * j + 1, -1] = 0.0
        b_ub[2 * j + 1] = -b[j] + epsilon_opt + slack

    # Weight change bounds: for each i, w_i - w0_i <= t * w0_i and -(w_i - w0_i) <= t * w0_i
    offset = 2 * m
    for i in range(n):
        # w_i - w0_i - t * w0_i <= 0  -> 1*w_i - w0_i* t <= w0_i
        row = np.zeros(n + 1)
        row[i] = 1.0
        row[-1] = -w0[i]
        A_ub[offset + 2 * i] = row
        b_ub[offset + 2 * i] = w0[i]
        # -w_i + w0_i - t * w0_i <= 0  -> -1*w_i - w0_i* t <= -w0_i
        row = np.zeros(n + 1)
        row[i] = -1.0
        row[-1] = -w0[i]
        A_ub[offset + 2 * i + 1] = row
        b_ub[offset + 2 * i + 1] = -w0[i]

    return A_ub, b_ub, bounds


def leximin_weight_fair(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
    slack: float = 0.0,
    return_stages: bool = False,
) -> CalibrationResult | tuple[CalibrationResult, CalibrationResult]:
    r"""Compute weights via residual leximin followed by weight-fair refinement.

    This function first solves the residual minimisation problem as in
    :func:`leximin_residual`, yielding an optimum ``epsilon``. It then fixes
    residuals to remain within ``epsilon + slack`` and minimises the worst
    relative change from the base weights. The optimisation problem for the
    second stage is:

    .. math::

        \min_{w, t}\;t
        \quad\text{such that}\quad
        |A\,w - b| \le \epsilon^\* + \text{slack},
        \quad
        |w_i - w_{0,i}| \le t\,w_{0,i},
        \quad
        w_i \in [w_{0,i}\,\text{min\_ratio},\, w_{0,i}\,\text{max\_ratio}].

    Parameters
    ----------
    A, b, w0 : see :func:`leximin_residual`.
    min_ratio, max_ratio : float, optional
        Weight bounds relative to the base weights. Defaults are ``0.1`` and
        ``10.0`` respectively.
    slack : float, optional
        Additional slack added to the optimum residual when constraining
        residuals in the second stage. Allows the algorithm to trade a small
        increase in margin error for improved weight stability. Defaults to
        ``0.0``.
    return_stages : bool, optional
        If ``True``, return the intermediate result of the residual stage as
        well as the final result.

    Returns
    -------
    CalibrationResult or tuple
        If ``return_stages`` is ``False`` (default), a single
        :class:`CalibrationResult` containing the final weights and both the
        residual and weight fairness optima. If ``return_stages`` is
        ``True``, a tuple ``(stage1_result, stage2_result)``.

    """
    stage1 = leximin_residual(A, b, w0, min_ratio=min_ratio, max_ratio=max_ratio)
    # If the residual stage failed, propagate the failure
    if stage1.status != 0 or np.isnan(stage1.epsilon):
        if return_stages:
            return stage1, stage1
        return stage1

    # Set up the second stage: minimise t subject to residual constraints and weight change bounds
    A, b, w0 = _validate_inputs(A, b, w0)
    n = A.shape[1]

    # Variables: w (n) and t (1)
    # Objective: minimise t
    c = np.zeros(n + 1)
    c[-1] = 1.0

    # Set up constraints using helper function
    A_ub, b_ub, bounds = _setup_weight_fair_constraints(
        A,
        b,
        w0,
        stage1.epsilon,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        slack=slack,
    )

    res = _solve_lp(c, A_ub, b_ub, bounds)
    if not res.success:
        stage2 = CalibrationResult(
            w=np.full_like(w0, np.nan),
            epsilon=stage1.epsilon,
            t=np.nan,
            status=res.status,
            message=res.message,
        )
        if return_stages:
            return stage1, stage2
        return stage2

    x = res.x
    w = x[:n]
    t_opt = x[-1]
    stage2 = CalibrationResult(
        w=w,
        epsilon=stage1.epsilon,
        t=t_opt,
        status=res.status,
        message=res.message,
    )
    if return_stages:
        return stage1, stage2
    return stage2
