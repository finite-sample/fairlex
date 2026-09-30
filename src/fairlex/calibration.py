r"""Core calibration routines for fairlex.

Both routines take a membership matrix ``A`` of shape ``(m, n)`` (one row per
margin, one column per unit; entries are 0/1 or soft memberships), target
totals ``b`` of length ``m`` and base weights ``w0`` of length ``n``. Each
calibrated weight is kept within ``[min_ratio * w0_i, max_ratio * w0_i]``.

Margin misses are compared on a common scale before they are ranked. With
``scale="relative"`` (the default) margin ``j`` contributes
:math:`|A_j w - b_j| / |b_j|`, so a miss of 10 on a group of 30 counts for
more than a miss of 10 on the population total. With ``scale="absolute"``
raw misses are compared, and an array of per-margin scales sets priorities
directly: margin ``j``'s miss is divided by ``s_j``.

* :func:`leximin_residual` finds the leximin-optimal vector of scaled misses:
  the largest miss is as small as possible, then the second largest, and so
  on.
* :func:`leximin_weight_fair` keeps every margin at its leximin level (plus
  optional ``slack``) and, among those weights, makes the vector of relative
  weight changes :math:`|w_i - w_{0,i}| / w_{0,i}` leximin-optimal as well.

Implementation notes. Every problem is solved in ratio space,
:math:`g_i = w_i / w_{0,i}`, which keeps the linear programmes well scaled
whatever the magnitude of the weights. Units with identical columns of ``A``
("cells") are interchangeable: giving them a common ratio never worsens
either leximin vector (averaging two ratios never raises the larger of their
changes), so the programmes are solved over distinct cells rather than
units. With 0/1 memberships on a handful of margins that is at most a few
dozen variables regardless of ``n``. Soft memberships rarely repeat, so they
get no such reduction. Levels are held to within a small tolerance of the
solver's optimum, so misses and changes are leximin-optimal to about 1e-5 of
the target scale, not to machine precision.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

EXPECTED_MATRIX_DIMENSIONS = 2

# HiGHS works to a primal feasibility tolerance of 1e-7, so a solution it
# returns can sit that far outside its constraints. Every "hold this at its
# level" bound gets this much room (relative to the size of the quantity) on
# top of the level the previous solution actually achieved. On 300 simulated
# surveys the weight stage failed 30 times at 1e-8 and never at 1e-7 or above;
# 3e-7 keeps a margin at a cost of misses up to ~1e-5 of a target above the
# exact leximin level.
_TOL = 3e-7
# Duals smaller than this are treated as zero when deciding which objectives
# are saturated.
_DUAL_TOL = 1e-9

Scale = Literal["relative", "absolute"] | np.ndarray | Sequence[float]


@dataclass
class CalibrationResult:
    """Structured result from a calibration call.

    Attributes:
        w: Calibrated weights of shape ``(n,)``. ``NaN`` if the solve failed.
        residuals: Raw margin residuals ``A @ w - b`` of shape ``(m,)``.
        epsilon: Largest scaled absolute residual achieved by ``w``
            (a fraction of the target under ``scale="relative"``).
        t: Largest relative weight change ``|w_i - w0_i| / w0_i`` achieved,
            over units with positive base weight. ``None`` for
            :func:`leximin_residual` and for failed solves.
        status: Status code of the last linear programme (0 is success).
        message: Solver termination message for diagnostics.

    """

    w: np.ndarray
    residuals: np.ndarray
    epsilon: float
    t: float | None
    status: int
    message: str


@dataclass
class _Cells:
    """The calibration problem collapsed onto distinct membership cells."""

    units: np.ndarray  # indices of units with positive base weight
    cell_of_unit: np.ndarray  # cell index of each unit in ``units``
    coef: sparse.csr_array  # (m, n_cells): scaled margin total per unit of ratio
    target: np.ndarray  # (m,): scaled targets b / s
    bounds: np.ndarray  # (n_cells, 2): ratio bounds


def _validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    min_ratio: float,
    max_ratio: float,
    scale: Scale,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and derive the per-margin residual scale.

    Args:
        A: Membership matrix of shape ``(m, n)``.
        b: Target totals of shape ``(m,)``.
        w0: Base weights of shape ``(n,)``.
        min_ratio: Lower bound on weights relative to ``w0``.
        max_ratio: Upper bound on weights relative to ``w0``.
        scale: ``"relative"``, ``"absolute"`` or one positive number per
            margin.

    Returns:
        ``(A, b, w0, s)`` as float arrays, where ``s`` holds the divisor for
        each margin's residual.

    Raises:
        ValueError: If shapes are incompatible, values are non-finite, base
            weights are negative, the ratios are not ``0 <= min <= max``,
            ``scale`` is unknown or not one positive number per margin, or a
            relative scale meets a zero target.

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
    for name, arr in (("A", A), ("b", b), ("w0", w0)):
        if not np.all(np.isfinite(arr)):
            msg = f"{name} must contain only finite values"
            raise ValueError(msg)
    if np.any(w0 < 0):
        msg = "w0 must be non-negative"
        raise ValueError(msg)
    if not (np.isfinite(max_ratio) and 0 <= min_ratio <= max_ratio):
        msg = (
            "ratios must satisfy 0 <= min_ratio <= max_ratio < inf, got "
            f"min_ratio={min_ratio}, max_ratio={max_ratio}"
        )
        raise ValueError(msg)
    return A, b, w0, _resolve_scale(scale, b)


def _resolve_scale(scale: Scale, b: np.ndarray) -> np.ndarray:
    """Turn the ``scale`` argument into one positive divisor per margin.

    Args:
        scale: ``"relative"``, ``"absolute"`` or one positive number per
            margin.
        b: Target totals of shape ``(m,)``.

    Returns:
        The per-margin divisors ``s``.

    Raises:
        ValueError: If ``scale`` is unknown or not one positive finite number
            per margin, or a relative scale meets a zero target.

    """
    m = len(b)
    if not isinstance(scale, str):
        s = np.asarray(scale, dtype=float)
        if s.shape != (m,) or not np.all(np.isfinite(s)) or np.any(s <= 0):
            msg = (
                f"scale as an array must hold {m} finite positive numbers, one "
                f"per margin; got {scale!r}"
            )
            raise ValueError(msg)
        return s
    if scale == "relative":
        if np.any(b == 0):
            msg = (
                "scale='relative' divides each residual by |b_j|, but some "
                "targets are zero; use scale='absolute' or a scale array"
            )
            raise ValueError(msg)
        return np.abs(b)
    if scale == "absolute":
        return np.ones(m)
    msg = f"scale must be 'relative', 'absolute' or an array, got {scale!r}"
    raise ValueError(msg)


def _cells(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    s: np.ndarray,
    min_ratio: float,
    max_ratio: float,
) -> _Cells:
    units = np.flatnonzero(w0 > 0)
    m = A.shape[0]
    if len(units):
        patterns, cell_of_unit = np.unique(A[:, units].T, axis=0, return_inverse=True)
        cell_of_unit = cell_of_unit.ravel()
    else:
        patterns, cell_of_unit = np.zeros((0, m)), np.zeros(0, dtype=int)
    mass = np.bincount(cell_of_unit, weights=w0[units], minlength=len(patterns))
    coef = sparse.csr_array(patterns.T * mass / s[:, None])
    bounds = np.tile([min_ratio, max_ratio], (len(patterns), 1)).astype(float)
    return _Cells(units, cell_of_unit, coef, b / s, bounds)


def _solve_lp(
    c: np.ndarray,
    A_ub: sparse.csr_array,
    b_ub: np.ndarray,
    A_eq: sparse.csr_array,
    b_eq: np.ndarray,
    bounds: np.ndarray,
) -> OptimizeResult:
    """Solve ``min c @ x`` subject to linear constraints and bounds with HiGHS.

    Args:
        c: Objective coefficients.
        A_ub: Inequality constraint matrix (``A_ub @ x <= b_ub``).
        b_ub: Inequality constraint right hand side.
        A_eq: Equality constraint matrix (``A_eq @ x == b_eq``).
        b_eq: Equality constraint right hand side.
        bounds: Array of shape ``(len(c), 2)`` with lower and upper bounds.

    Returns:
        The solver's ``OptimizeResult``.

    """
    return linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )


def _stack(blocks: list[list[sparse.csr_array | None]]) -> sparse.csr_array:
    return sparse.csr_array(sparse.block_array(blocks))


def _tol(level: np.ndarray, center: np.ndarray) -> np.ndarray:
    return _TOL * (1.0 + np.abs(level) + np.abs(center))


def _leximin(
    rows_obj: sparse.csr_array,
    center: np.ndarray,
    bounds: np.ndarray,
    base: tuple[sparse.csr_array, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, OptimizeResult | None]:
    """Leximin levels of ``|rows_obj x - center|`` by saturation.

    Each round minimises the largest deviation ``z`` over the objectives not
    yet fixed, holding fixed objectives at their levels. An objective whose
    ``|e_j| <= z`` rows carry a positive dual is at ``z*`` in every optimal
    solution of that round (complementary slackness), so it can be fixed
    there without loss. The duals on ``z`` sum to one, so each round fixes at
    least one objective and there are at most ``len(center)`` rounds. This is
    the dual variant of Ogryczak & Śliwiński (2006), "On Direct Methods for
    Lexicographic Min-Max Optimization". The one maintained Python package
    for leximin, ``cvxpy_leximin``, patches ``cvxpy.Problem`` globally, uses
    1% saturation tolerances and solves one sub-problem per objective per
    round, so it is not used here.

    Each deviation is an explicit variable ``e_j = rows_obj_j x - center_j``,
    so holding it at its level is a bound on ``e_j`` rather than a pair of
    opposing inequality rows. Bands narrower than the solver's feasibility
    tolerance written as row pairs made HiGHS report feasible problems as
    infeasible; as bounds they are handled reliably.

    Args:
        rows_obj: Objective rows, shape ``(p, n_cells)``.
        center: Targets for the objective rows, shape ``(p,)``.
        bounds: Variable bounds, shape ``(n_cells, 2)``.
        base: Constraints ``|rows x - center| <= radius`` that always hold,
            as ``(rows, center, radius)``.

    Returns:
        ``(levels, x, res)``: each objective's leximin level, the solution of
        the last round (``None`` if no programme was solved) and the last
        ``OptimizeResult`` (``None`` if no programme was solved).

    """
    p, n_cells = len(center), len(bounds)
    if base is None:
        base = (sparse.csr_array((0, n_cells)), np.zeros(0), np.zeros(0))
    base_rows, base_center, base_radius = base
    q = len(base_center)
    # Variables: x (n_cells), e (p), r (q), z. Equalities define e and r.
    A_eq = _stack(
        [
            [rows_obj, -sparse.eye_array(p), None],
            [base_rows, None, -sparse.eye_array(q)],
        ]
    )
    A_eq = _stack([[A_eq, sparse.csr_array((p + q, 1))]])
    b_eq = np.concatenate([center, base_center])
    objective = np.r_[np.zeros(n_cells + p + q), 1.0]
    levels = np.full(p, np.nan)
    free = np.ones(p, dtype=bool)
    x, res = None, None
    while free.any():
        free_idx = np.flatnonzero(free)
        k = len(free_idx)
        pick = sparse.csr_array(
            (np.ones(k), (np.arange(k), n_cells + free_idx)), shape=(k, len(objective))
        )
        z_col = sparse.csr_array(
            (np.ones(k), (np.arange(k), np.full(k, len(objective) - 1))),
            shape=(k, len(objective)),
        )
        A_ub = _stack([[pick - z_col], [-pick - z_col]])
        e_bound = np.where(free, np.inf, levels + _tol(levels, center))
        all_bounds = np.vstack(
            [
                bounds,
                np.column_stack([-e_bound, e_bound]),
                np.column_stack([-base_radius, base_radius]),
                [0.0, np.inf],
            ]
        )
        res = _solve_lp(objective, A_ub, np.zeros(2 * k), A_eq, b_eq, all_bounds)
        if not res.success:
            return levels, None, res
        x, z = res.x[:n_cells], float(res.x[-1])
        if z <= _TOL:
            saturated = np.ones(k, dtype=bool)
        else:
            marginals = res.ineqlin.marginals
            dual = -(marginals[:k] + marginals[k:])
            saturated = dual > _DUAL_TOL
            if not saturated.any():
                saturated[np.argmax(dual)] = True
        # Record levels from what x achieves, not from z: x may sit up to the
        # solver's tolerance outside its constraints, and a level below x's
        # own deviation could make the next round infeasible.
        achieved = np.abs(rows_obj @ x - center)
        fixed = ~free
        levels[fixed] = np.maximum(levels[fixed], achieved[fixed])
        newly = free_idx[saturated]
        levels[newly] = np.maximum(z, achieved[newly])
        free[newly] = False
    return levels, x, res


def _failure(n: int, m: int, res: OptimizeResult) -> CalibrationResult:
    return CalibrationResult(
        w=np.full(n, np.nan),
        residuals=np.full(m, np.nan),
        epsilon=np.nan,
        t=None,
        status=int(res.status),
        message=str(res.message),
    )


def _weights(
    w0: np.ndarray,
    cells: _Cells,
    g: np.ndarray | None,
    min_ratio: float,
    max_ratio: float,
) -> np.ndarray:
    w = np.zeros_like(w0)
    if g is None:  # no margins: nothing pulls the weights away from w0
        w[cells.units] = w0[cells.units] * np.clip(1.0, min_ratio, max_ratio)
    else:
        w[cells.units] = w0[cells.units] * g[cells.cell_of_unit]
    return w


def _result(
    A: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    w: np.ndarray,
    t: float | None,
    res: OptimizeResult | None,
) -> CalibrationResult:
    residuals = A @ w - b
    scaled = np.abs(residuals) / s
    return CalibrationResult(
        w=w,
        residuals=residuals,
        epsilon=float(scaled.max()) if len(scaled) else 0.0,
        t=t,
        status=0 if res is None else int(res.status),
        message="No linear programme needed." if res is None else str(res.message),
    )


def leximin_residual(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
    scale: Scale = "relative",
) -> CalibrationResult:
    r"""Compute weights whose scaled margin misses are leximin-optimal.

    With :math:`r_j(w) = |A_j w - b_j| / s_j` (``s_j = |b_j|`` for
    ``scale="relative"``, 1 for ``"absolute"``), the returned weights make the
    vector of :math:`r_j`, sorted from largest to smallest, lexicographically
    minimal: the largest miss is as small as possible, then the second
    largest given that, and so on, subject to
    :math:`w_{0,i}\,\text{min\_ratio} \le w_i \le w_{0,i}\,\text{max\_ratio}`.

    The miss levels are unique, but the weights achieving them usually are
    not; :func:`leximin_weight_fair` picks the ones that change ``w0`` least.

    Args:
        A: Membership matrix of shape ``(m, n)``.
        b: Target totals of shape ``(m,)``.
        w0: Non-negative base weights of shape ``(n,)``.
        min_ratio: Lower bound on weights relative to ``w0``.
        max_ratio: Upper bound on weights relative to ``w0``.
        scale: How misses are compared across margins: ``"relative"`` divides
            each by ``|b_j|``, ``"absolute"`` uses raw misses, and an array
            ``s`` of one positive number per margin divides margin ``j``'s
            miss by ``s_j``. A larger ``s_j`` means that margin matters less:
            ``epsilon`` is then the worst-case bias over outcomes whose
            loadings satisfy ``sum_j s_j |beta_j| <= 1`` (see the theory page).

    Returns:
        Weights, raw residuals and the largest scaled miss ``epsilon``. If a
        solve fails, ``status`` is nonzero and the arrays are ``NaN``.

    """
    A, b, w0, s = _validate_inputs(
        A, b, w0, min_ratio=min_ratio, max_ratio=max_ratio, scale=scale
    )
    cells = _cells(A, b, w0, s, min_ratio, max_ratio)
    _, g, res = _leximin(cells.coef, cells.target, cells.bounds)
    if res is not None and not res.success:
        return _failure(len(w0), len(b), res)
    return _result(A, b, s, _weights(w0, cells, g, min_ratio, max_ratio), None, res)


def leximin_weight_fair(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
    scale: Scale = "relative",
    slack: float = 0.0,
) -> CalibrationResult:
    r"""Compute leximin-calibrated weights whose changes are leximin-fair too.

    First the leximin miss level :math:`\ell_j` of every margin is found as in
    :func:`leximin_residual`. Then, over weights with
    :math:`|A_j w - b_j| / s_j \le \ell_j + \text{slack}` for every margin and
    within the ratio bounds, the relative changes
    :math:`|w_i - w_{0,i}| / w_{0,i}` are made leximin-optimal: the largest
    change is as small as possible, then the next largest, and so on. The
    adjustment a margin needs is therefore spread evenly over the units that
    can supply it, and units no margin needs to move stay at ``w0``. Units
    with ``w0_i = 0`` stay at zero and are excluded from :math:`t`.

    Args:
        A: Membership matrix of shape ``(m, n)``.
        b: Target totals of shape ``(m,)``.
        w0: Non-negative base weights of shape ``(n,)``.
        min_ratio: Lower bound on weights relative to ``w0``.
        max_ratio: Upper bound on weights relative to ``w0``.
        scale: How misses are compared across margins: ``"relative"`` divides
            each by ``|b_j|``, ``"absolute"`` uses raw misses, and an array
            ``s`` of one positive number per margin divides margin ``j``'s
            miss by ``s_j``. A larger ``s_j`` means that margin matters less:
            ``epsilon`` is then the worst-case bias over outcomes whose
            loadings satisfy ``sum_j s_j |beta_j| <= 1`` (see the theory page).
        slack: Extra scaled miss each margin may take on top of its leximin
            level, in the units of ``scale`` (a fraction of the target under
            ``"relative"``), to buy smaller weight changes.

    Returns:
        Weights, raw residuals, the achieved largest scaled miss ``epsilon``
        and the achieved largest relative weight change ``t``. If a solve
        fails, ``status`` is nonzero, the arrays are ``NaN`` and ``t`` is
        ``None``.

    Raises:
        ValueError: If ``slack`` is negative or non-finite, or the inputs are
            invalid (see :func:`leximin_residual`).

    """
    if not (np.isfinite(slack) and slack >= 0):
        msg = f"slack must be a finite non-negative number, got {slack}"
        raise ValueError(msg)
    A, b, w0, s = _validate_inputs(
        A, b, w0, min_ratio=min_ratio, max_ratio=max_ratio, scale=scale
    )
    m, n = A.shape
    cells = _cells(A, b, w0, s, min_ratio, max_ratio)
    levels, g, res = _leximin(cells.coef, cells.target, cells.bounds)
    if res is not None and not res.success:
        return _failure(n, m, res)

    n_cells = len(cells.bounds)
    if n_cells:
        radius = levels + slack
        margins = (cells.coef, cells.target, radius + _tol(radius, cells.target))
        _, g, res = _leximin(
            sparse.csr_array(sparse.eye_array(n_cells)),
            np.ones(n_cells),
            cells.bounds,
            margins,
        )
        if res is not None and not res.success:
            return _failure(n, m, res)
    w = _weights(w0, cells, g, min_ratio, max_ratio)
    units = cells.units
    t = float(np.max(np.abs(w[units] / w0[units] - 1.0))) if len(units) else 0.0
    return _result(A, b, s, w, t, res)
