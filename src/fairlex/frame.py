"""Data-frame front end: targets as dictionaries, results as a report.

The target format follows the survey weighting packages people already use
(svy, weightipy, R icarus): ``{variable: {level: target}}``, either as totals
or, with ``shares=True``, as proportions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .calibration import leximin_weights
from .metrics import design_effect, effective_sample_size

# A margin counts as conflicting when its scaled miss is within this factor of
# the worst one: the solver holds levels to about 1e-6 of the target scale.
_BINDING_RTOL = 1e-4
_MIN_BINDING = 1e-6
# Shares may be rounded when published; within this of 1 they count as summing
# to 1. Also the relative tolerance for variables agreeing on a population size.
_SHARE_TOL = 1e-3
# A unit counts as at its bound when its ratio is within this fraction of the
# bound. The weight stage may move a capped unit by ~1e-4 while holding the
# margins to their tolerance (1.24984 against a 1.25 cap in the docs example),
# so the threshold sits well above that and well below any real gap.
_AT_BOUND_RTOL = 1e-3

Targets = Mapping[str, Mapping[object, float]]


@dataclass
class CalibrationReport:
    """What :func:`calibrate` returns.

    Attributes:
        weights: Calibrated weights, indexed like the input data frame.
        margins: One row per margin: ``variable``, ``level``, ``target``,
            ``before`` and ``after`` (weighted totals), ``miss`` and
            ``miss_pct`` (miss as a percentage of the target).
        binding: The margins at the worst scaled miss, when that miss is
            above zero, as ``(variable, level, reason)``. ``reason`` is
            ``"weight bounds"`` when every member already sits at the bound
            that would help, so the target is out of reach on its own, and
            ``"conflicting targets"`` otherwise: meeting it better would push
            another margin further off.
        epsilon: Largest scaled miss: the largest relative miss times its
            variable's importance.
        t: Largest relative weight change.
        ess: Kish effective sample size of the calibrated weights.
        deff: Kish design effect of the calibrated weights.
        status: Solver status (0 is success).
        message: Solver message.

    """

    weights: pd.Series
    margins: pd.DataFrame
    binding: list[tuple[str, object, str]]
    epsilon: float
    t: float | None
    ess: float
    deff: float
    status: int
    message: str

    def __str__(self) -> str:
        """Plain-text summary: headline numbers, binding margins and the table."""
        worst = self.margins["miss_pct"].abs().max() if len(self.margins) else 0.0
        lines = [
            f"largest miss: {worst:.2f}% of target   "
            f"largest weight change: {self.t if self.t is None else f'{self.t:.1%}'}   "
            f"ESS: {self.ess:,.0f}   design effect: {self.deff:.2f}",
        ]
        if self.binding:
            names = ", ".join(f"{v}={lvl} ({why})" for v, lvl, why in self.binding)
            lines.append(f"worst-missed targets: {names}")
        table = self.margins.copy()
        for col in ("target", "before", "after", "miss"):
            table[col] = (table[col].round(1) + 0.0).map("{:,.1f}".format)
        table["miss_pct"] = (self.margins["miss_pct"].round(2) + 0.0).map(
            "{:+.2f}%".format
        )
        lines.append(table.to_string(index=False))
        return "\n".join(lines)


def _margin_rows(
    data: pd.DataFrame, targets: Targets
) -> tuple[list[tuple[str, object]], np.ndarray, np.ndarray]:
    """Membership rows and target values, with the checks users need.

    Args:
        data: Respondent-level data.
        targets: ``{variable: {level: value}}``.

    Returns:
        The ``(variable, level)`` labels, membership matrix and raw targets.

    Raises:
        ValueError: If a variable is missing or has missing values, a level
            in the data has no target, or a target level has no respondents.

    """
    labels, rows, values = [], [], []
    for var, levels in targets.items():
        if var not in data.columns:
            msg = f"targets name {var!r}, which is not a column of the data"
            raise ValueError(msg)
        col = data[var]
        if bool(col.isna().any()):
            msg = f"{var!r} has missing values; recode them to a level first"
            raise ValueError(msg)
        untargeted = sorted(map(str, set(col.unique()) - set(levels)))
        if untargeted:
            msg = f"{var!r} has levels with no target: {', '.join(untargeted)}"
            raise ValueError(msg)
        for level, value in levels.items():
            member = (col == level).to_numpy(dtype=float)
            if not member.any():
                msg = f"no respondents have {var}={level!r}, so it cannot be hit"
                raise ValueError(msg)
            if not float(value) >= 0:
                msg = (
                    "targets must be non-negative numbers, "
                    f"got {var}={level!r}: {value!r}"
                )
                raise ValueError(msg)
            labels.append((var, level))
            rows.append(member)
            values.append(float(value))
    return labels, np.array(rows), np.array(values)


def _variable_sums(b: np.ndarray, variables: list[str]) -> dict[str, float]:
    return {
        v: float(b[[x == v for x in variables]].sum()) for v in dict.fromkeys(variables)
    }


def _totals(
    b: np.ndarray, variables: list[str], *, shares: bool, total: float | None
) -> np.ndarray:
    """Turn shares into totals, after checking they are proportions.

    Args:
        b: Raw target values, one per margin.
        variables: The variable of each margin.
        shares: Whether the values are proportions.
        total: Population size, required with ``shares``.

    Returns:
        Target totals, one per margin.

    Raises:
        ValueError: If ``shares`` is set without ``total`` or a variable's
            shares do not sum to 1.

    """
    if not shares:
        return b
    if total is None:
        msg = "shares=True needs total= to turn proportions into totals"
        raise ValueError(msg)
    sums = _variable_sums(b, variables)
    off = {v: s for v, s in sums.items() if abs(s - 1) > _SHARE_TOL}
    if off:
        detail = ", ".join(f"{v} sums to {s:g}" for v, s in off.items())
        msg = f"with shares=True each variable's targets must sum to 1: {detail}"
        raise ValueError(msg)
    return b * total


def _starting_weights(
    data: pd.DataFrame,
    b: np.ndarray,
    variables: list[str],
    *,
    base_weight: str | None,
    total: float | None,
) -> np.ndarray:
    """Base weights: the named column, or ``total / n`` for everyone.

    Args:
        data: Respondent-level data.
        b: Target totals, one per margin.
        variables: The variable of each margin.
        base_weight: Column of base weights, if any.
        total: Population size, if given.

    Returns:
        One base weight per respondent.

    Raises:
        ValueError: If there is no base-weight column and no ``total``, and
            the variables' targets imply different population sizes.

    """
    if base_weight is not None:
        return data[base_weight].to_numpy(dtype=float)
    size = total
    if size is None:
        sums = list(_variable_sums(b, variables).values())
        if max(sums) - min(sums) > _SHARE_TOL * max(sums):
            msg = (
                "the variables' targets imply different population sizes "
                f"({min(sums):g} to {max(sums):g}); pass total= or base_weight="
            )
            raise ValueError(msg)
        size = sums[0]
    return np.full(len(data), size / len(data))


def calibrate(
    data: pd.DataFrame,
    targets: Targets,
    *,
    base_weight: str | None = None,
    total: float | None = None,
    shares: bool = False,
    importance: Mapping[str, float] | None = None,
    bounds: tuple[float, float] = (0.1, 10.0),
    slack: float = 0.0,
) -> CalibrationReport:
    """Calibrate survey weights to categorical targets from a data frame.

    Misses are compared relative to each target, and both the misses and the
    weight changes are made leximin-optimal (see
    :func:`fairlex.calibration.leximin_weights`).

    Args:
        data: One row per respondent.
        targets: ``{variable: {level: target}}``. Every level present in the
            data must have a target. With ``shares=False`` the values are
            population totals, and different variables may disagree about
            the overall size: that disagreement is what gets shared out.
            With ``shares=True`` each variable's values are proportions that
            sum to 1, multiplied by ``total``. Targets must be non-negative.
        base_weight: Column holding the base (design) weights. If omitted,
            every respondent starts at ``total / n``.
        total: Population size. Added as its own margin when given. Required
            with ``shares=True``, and when ``base_weight`` is omitted and the
            variables' targets imply different population sizes.
        shares: Treat each variable's targets as proportions. Use it when
            your sources agree on proportions but not on the population size;
            the disagreement about size then disappears before calibration.
        importance: ``{variable: factor}``. A factor of 2 makes misses on that
            variable's margins count twice as much. Defaults to 1.
        bounds: ``(lower, upper)`` bounds on each weight relative to its base
            weight, as in R ``survey``, icarus and svy.
        slack: Extra relative miss every margin may take on top of its
            leximin level, in exchange for smaller weight changes.

    Returns:
        A :class:`CalibrationReport`; ``print`` it for a summary.

    Raises:
        ValueError: If the targets and data do not match (see above), if
            ``total`` is needed but missing, or if ``importance`` names an
            unknown variable or a non-positive factor.

    """
    labels, A, b = _margin_rows(data, targets)
    variables = [v for v, _ in labels]

    b = _totals(b, variables, shares=shares, total=total)
    n = len(data)
    w0 = _starting_weights(data, b, variables, base_weight=base_weight, total=total)

    if total is not None:
        labels.append(("(total)", "all"))
        A = np.vstack([A, np.ones(n)])
        b = np.r_[b, total]

    factor = np.ones(len(b))
    for var, value in (importance or {}).items():
        if var not in targets or not value > 0:
            msg = (
                "importance needs known variables and positive factors, "
                f"got {var!r}: {value!r}"
            )
            raise ValueError(msg)
        factor[[v == var for v, _ in labels]] = value

    result = leximin_weights(
        A,
        b,
        w0,
        np.abs(b) / factor,
        min_ratio=bounds[0],
        max_ratio=bounds[1],
        slack=slack,
    )

    before = A @ w0
    after = A @ result.w
    margins = pd.DataFrame(
        {
            "variable": [v for v, _ in labels],
            "level": [lvl for _, lvl in labels],
            "target": b,
            "before": before,
            "after": after,
            "miss": after - b,
            "miss_pct": 100 * (after - b) / b,
        }
    )
    scaled = np.abs(result.residuals) * factor / np.abs(b)
    binding = []
    # A miss within ``slack`` was allowed on purpose, so it is not binding.
    if result.status == 0 and result.epsilon > slack + _MIN_BINDING:
        # Units with zero base weight (e.g. left out of a bootstrap draw) are
        # pinned at zero and have no ratio, so they are not members here.
        positive = w0 > 0
        ratio = np.divide(result.w, w0, out=np.ones_like(w0), where=positive)
        top = scaled >= result.epsilon * (1 - _BINDING_RTOL)
        for j in np.flatnonzero(top):
            members = (A[j] > 0) & positive
            helpful = bounds[1] if result.residuals[j] < 0 else bounds[0]
            at_bound = np.all(
                np.abs(ratio[members] - helpful) <= _AT_BOUND_RTOL * helpful
            )
            reason = "weight bounds" if at_bound else "conflicting targets"
            binding.append((*labels[j], reason))

    return CalibrationReport(
        weights=pd.Series(result.w, index=data.index, name="weight"),
        margins=margins,
        binding=binding,
        epsilon=result.epsilon,
        t=result.t,
        ess=effective_sample_size(result.w),
        deff=design_effect(result.w),
        status=result.status,
        message=result.message,
    )


def calibrate_replicates(
    data: pd.DataFrame,
    targets: Targets,
    replicate_weights: Sequence[str],
    **options: Any,
) -> pd.DataFrame:
    """Calibrate every set of replicate base weights to the same targets.

    This is how calibrated weights get standard errors: generate replicate
    base weights with your survey package (bootstrap, jackknife, BRR, e.g.
    svy's ``create_bs_wgts`` or R ``survey::as.svrepdesign``), calibrate each
    replicate exactly as the full sample was, and let the survey package
    combine the replicate estimates. Replicate weights of zero, as in a
    bootstrap draw that leaves a respondent out, stay zero.

    Args:
        data: One row per respondent.
        targets: Same as for :func:`calibrate`.
        replicate_weights: Columns of ``data`` holding replicate base weights.
        **options: Any other keyword argument of :func:`calibrate`, applied
            to every replicate. Pass the same ones as for the full sample.

    Returns:
        Calibrated replicate weights, one column per replicate, indexed like
        ``data``.

    Raises:
        ValueError: If ``options`` includes ``base_weight``, which the
            replicate columns replace.

    """
    if "base_weight" in options:
        msg = "replicate_weights replace base_weight; do not pass both"
        raise ValueError(msg)
    return pd.DataFrame(
        {
            col: calibrate(data, targets, base_weight=col, **options).weights
            for col in replicate_weights
        },
        index=data.index,
    )
