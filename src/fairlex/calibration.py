r"""The leximin calibration engine behind :func:`fairlex.calibrate`.

:func:`leximin_weights` takes a membership matrix ``A`` of shape ``(m, n)``
(one row per margin, one column per respondent), target totals ``b``, base
weights ``w0`` and a positive scale ``s`` per margin. It finds weights within
``[min_ratio * w0_i, max_ratio * w0_i]`` whose scaled misses
:math:`|A_j w - b_j| / s_j` are leximin-optimal (the largest as small as
possible, then the next largest, and so on) and, among those, whose relative
changes :math:`|w_i - w_{0,i}| / w_{0,i}` are leximin-optimal as well.

Every problem is solved in ratio space, :math:`g_i = w_i / w_{0,i}`, which
keeps the linear programmes well scaled whatever the size of the weights.
Respondents with identical columns of ``A`` ("cells") are interchangeable:
giving them a common ratio never worsens either leximin vector (averaging two
ratios never raises the larger of their changes), so the programmes are
solved over distinct cells rather than respondents. Levels are held to within
a small tolerance of the solver's optimum, so misses and changes are
leximin-optimal to about 1e-5 of the target scale, not to machine precision.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, linprog

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


@dataclass
class CalibrationResult:
    """What :func:`leximin_weights` returns.

    Attributes:
        w: Calibrated weights of shape ``(n,)``. ``NaN`` if the solve failed.
        residuals: Raw margin residuals ``A @ w - b`` of shape ``(m,)``.
        epsilon: Largest scaled absolute residual achieved by ``w``.
        t: Largest relative weight change ``|w_i - w0_i| / w0_i`` achieved,
            over units with positive base weight. ``None`` if the solve
            failed.
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
    b: np.ndarray,
    w0: np.ndarray,
    s: np.ndarray,
    *,
    min_ratio: float,
    max_ratio: float,
    slack: float,
) -> None:
    """Check the inputs a user can get wrong through :func:`fairlex.calibrate`.

    Args:
        b: Target totals.
        w0: Base weights.
        s: Per-margin scales.
        min_ratio: Lower bound on weights relative to ``w0``.
        max_ratio: Upper bound on weights relative to ``w0``.
        slack: Extra scaled miss allowed on top of the leximin levels.

    Raises:
        ValueError: If targets or base weights are non-finite, base weights
            are negative, the bounds are not ``0 <= lower <= upper < inf``,
            scales are not positive and finite, or ``slack`` is negative.

    """
    if not np.all(np.isfinite(b)):
        msg = "targets must be finite numbers"
        raise ValueError(msg)
    if not np.all(np.isfinite(w0)) or np.any(w0 < 0):
        msg = "base weights must be finite and non-negative"
        raise ValueError(msg)
    if not (np.isfinite(max_ratio) and 0 <= min_ratio <= max_ratio):
        msg = (
            "bounds must satisfy 0 <= lower <= upper < inf, "
            f"got ({min_ratio}, {max_ratio})"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        msg = "every margin needs a positive, finite scale; is a target zero?"
        raise ValueError(msg)
    if not (np.isfinite(slack) and slack >= 0):
        msg = f"slack must be a finite non-negative number, got {slack}"
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


def leximin_weights(
    A: np.ndarray,
    b: np.ndarray,
    w0: np.ndarray,
    s: np.ndarray | None = None,
    *,
    min_ratio: float = 0.1,
    max_ratio: float = 10.0,
    slack: float = 0.0,
) -> CalibrationResult:
    r"""Leximin-calibrated weights whose changes are leximin-fair too.

    First the leximin miss level :math:`\ell_j` of every margin is found:
    the largest scaled miss :math:`|A_j w - b_j| / s_j` as small as the bounds
    allow, then the next largest, and so on. Then, over weights with
    :math:`|A_j w - b_j| / s_j \le \ell_j + \text{slack}` for every margin
    and within the ratio bounds, the relative changes
    :math:`|w_i - w_{0,i}| / w_{0,i}` are made leximin-optimal in the same
    way. The adjustment a margin needs is spread evenly over the respondents
    who can supply it, and respondents no margin needs to move stay at
    ``w0``. Respondents with ``w0_i = 0`` stay at zero.

    Args:
        A: Membership matrix of shape ``(m, n)``.
        b: Target totals of shape ``(m,)``.
        w0: Non-negative base weights of shape ``(n,)``.
        s: Positive scale per margin; margin ``j``'s miss is divided by
            ``s_j``. Defaults to ``|b|``, so misses compare as fractions of
            their targets.
        min_ratio: Lower bound on weights relative to ``w0``.
        max_ratio: Upper bound on weights relative to ``w0``.
        slack: Extra scaled miss each margin may take on top of its leximin
            level, to buy smaller weight changes.

    Returns:
        Weights, raw residuals, the achieved largest scaled miss ``epsilon``
        and the achieved largest relative weight change ``t``. If a solve
        fails, ``status`` is nonzero, the arrays are ``NaN`` and ``t`` is
        ``None``.

    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    w0 = np.asarray(w0, dtype=float)
    s = np.abs(b) if s is None else np.asarray(s, dtype=float)
    _validate_inputs(b, w0, s, min_ratio=min_ratio, max_ratio=max_ratio, slack=slack)
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
