"""Do bootstrap standard errors for fairlex-calibrated estimates hold up?

Recipe under test: draw bootstrap replicate base weights with svy,
calibrate each replicate with ``fairlex.calibrate_replicates`` exactly as the
full sample, and take the spread of the replicate estimates as the standard
error of the weighted total.

Population: 20,000 people with age (4 levels), education (3) and region (4);
outcome y is additive in all three plus an age-by-education interaction the
margins miss, plus noise. Sample: Poisson sampling with inclusion
probabilities that favour older, more educated people (expected size 1,000),
then nonresponse that depends on age alone (response rates 35% to 80%).
Base weights are 1 / pi, so they undo the selection but not the nonresponse;
calibrating on age undoes the nonresponse, because it is uniform within age.

Regimes:

* ``exact targets, loose bounds``: targets are the true margins, bounds
  (0.2, 5). The estimator should be nearly unbiased, its standard errors
  calibrated and its 95% intervals should cover.
* ``exact targets, binding bounds``: bounds (0.7, 1.3), which stop the
  weights from fully correcting the nonresponse, so estimates are biased;
  standard errors should still match the sampling spread.
* ``disagreeing targets``: one fixed set of targets, each variable off by a
  common 5% plus 2% category noise; bounds (0.2, 5). Estimates are biased by
  the bad targets; standard errors should still match the sampling spread.

Two bootstrap schemes are compared in every regime. Rao-Wu resamples a fixed
number of units; the Poisson bootstrap lets the replicate sample size vary,
as Poisson sampling does. When weight bounds stop calibration from absorbing
the variation in realised sample size, only the Poisson bootstrap sees it.

Gates come from simcheck (bands set by the replicate count, not hand-picked):
``assert_se_calibrated`` in every regime, and ``assert_unbiased`` and
``assert_coverage`` in the first. Writes ``benchmarks/results/se_study.csv``.

Run: ``uv run --group bench python benchmarks/se_study.py``.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import simcheck
import svy

from fairlex import calibrate, calibrate_replicates

N_POP, EXPECTED_N = 20_000, 1_000
# 200 bootstrap replicates per sample and simcheck's deep tier of 400 Monte
# Carlo samples, so coverage is resolved to about +/- 3 points. Set SE_REPS
# and SE_BOOT to small numbers for a quick smoke run.
N_BOOT = int(os.environ.get("SE_BOOT", "200"))
RESPONSE_BY_AGE = np.array([0.35, 0.5, 0.65, 0.8])
MC_REPS = int(os.environ.get("SE_REPS", str(simcheck.DEEP_REPS)))
LEVELS = {"age": 4, "edu": 3, "region": 4}
SEED = 20261001
BOOT_KINDS = ("rao-wu", "poisson")
OUT = Path(__file__).parent / "results" / "se_study.csv"
SUMMARY = Path(__file__).parent / "results" / "se_study_summary.csv"


def make_population(rng):
    """Population data frame plus the outcome column ``y``."""
    z = rng.normal(size=N_POP)
    age = np.digitize(z + rng.normal(size=N_POP), [-1.0, 0.0, 1.0])
    edu = np.digitize(0.8 * z + rng.normal(size=N_POP), [-0.5, 0.7])
    region = rng.integers(4, size=N_POP)
    pop = pd.DataFrame({"age": age, "edu": edu, "region": region}).astype(str)
    y = (
        np.array([0.0, 1.0, 2.0, 3.0])[age]
        + np.array([0.0, 1.5, 3.0])[edu]
        + np.array([0.0, 0.5, -0.5, 1.0])[region]
        + 0.8 * (age * edu == 0)
        + rng.normal(size=N_POP)
    )
    pop["y"] = y
    return pop


def true_targets(pop):
    """Population counts of every category of every variable."""
    return {
        var: {lvl: float(c) for lvl, c in pop[var].value_counts().items()}
        for var in LEVELS
    }


def noisy_targets(targets, rng):
    """Each variable off by a common 5% plus 2% per category."""
    return {
        var: {
            lvl: v * (1 + 0.05 * common) * (1 + 0.02 * rng.normal())
            for lvl, v in levels.items()
        }
        for var, levels, common in (
            (var, levels, rng.choice([-1.0, 1.0])) for var, levels in targets.items()
        )
    }


def inclusion_probs(pop):
    """Poisson inclusion probabilities favouring older, more educated people."""
    score = np.exp(0.35 * pop["age"].astype(int) + 0.3 * pop["edu"].astype(int))
    return np.minimum(score * EXPECTED_N / score.sum(), 1.0)


def world():
    """Population, inclusion probabilities and the three regimes' settings.

    Deterministic, so every worker process rebuilds the same world.
    """
    rng = np.random.default_rng(SEED)
    pop = make_population(rng)
    pi = inclusion_probs(pop).to_numpy()
    exact = true_targets(pop)
    regimes = {
        "exact targets, loose bounds": (exact, (0.2, 5.0), True),
        "exact targets, binding bounds": (exact, (0.7, 1.3), False),
        "disagreeing targets": (noisy_targets(exact, rng), (0.2, 5.0), False),
    }
    return pop, pi, regimes


_WORLD = None


def one_sample(task):
    """One Monte Carlo sample: draw, calibrate, bootstrap the standard error."""
    global _WORLD  # noqa: PLW0603 - per-process cache of the deterministic world
    if _WORLD is None:
        _WORLD = world()
    pop, pi, regimes = _WORLD
    regime, kind, seed = task
    targets, bounds, _ = regimes[regime]
    rng = np.random.default_rng(seed)
    age = pop["age"].astype(int).to_numpy()
    keep = (rng.random(N_POP) < pi) & (rng.random(N_POP) < RESPONSE_BY_AGE[age])
    sample = pop[keep].reset_index(drop=True)
    sample["w0"] = 1 / pi[keep]
    sample["id"] = np.arange(len(sample))  # Poisson sampling: each unit a PSU
    boot = (
        svy.Sample(pl.from_pandas(sample), svy.Design(wgt="w0", psu="id"))
        .weighting.create_bs_wgts(n_reps=N_BOOT, kind=kind, rep_prefix="bs", rstate=rng)
        .data.to_pandas()
    )
    rep_cols = [c for c in boot.columns if c.startswith("bs")]
    options = {"total": float(N_POP), "bounds": bounds}
    full = calibrate(sample, targets, base_weight="w0", **options).weights
    reps = calibrate_replicates(boot, targets, rep_cols, **options)
    estimate = float(full @ sample["y"])
    rep_estimates = reps.to_numpy().T @ boot["y"].to_numpy()
    se = float(np.sqrt(np.mean((rep_estimates - estimate) ** 2)))
    return estimate, se


def gate(check, *args):
    """Run one simcheck gate and report pass or the reason it failed."""
    try:
        check(*args)
    except AssertionError as err:
        return f"fail: {err}"
    return "pass"


def summarize(table):
    """Compact table for the docs; gate outcomes as pass or fail."""
    out = pd.DataFrame(
        {
            "regime": table["regime"],
            "bootstrap": table["bootstrap"],
            "bias, % of truth": table["bias, % of truth"].round(1) + 0.0,
            "reported se / true spread": table["se ratio"].round(2),
            "95% coverage, %": (100 * table["95% coverage"]).round(1),
        }
    )
    for col in ("se calibrated", "unbiased", "coverage ok"):
        if col in table:
            out[f"{col} (simcheck)"] = (
                table[col].fillna("not tested").str.split(":").str[0]
            )
    return out


def main():
    """Run the three regimes and write one row each."""
    pop, _, regimes = world()
    truth = float(pop["y"].sum())
    # The same per-sample seeds simcheck.monte_carlo would use, so the runs
    # reproduce and extend; samples run in parallel across processes.
    seeds = np.random.SeedSequence(7).spawn(MC_REPS)
    rows = []
    with ProcessPoolExecutor() as pool:
        for (name, (_, _, expect_unbiased)), kind in (
            (item, kind) for item in regimes.items() for kind in BOOT_KINDS
        ):
            tasks = [(name, kind, s) for s in seeds]
            out = np.array(list(pool.map(one_sample, tasks)))
            estimates, ses = out[:, 0], out[:, 1]
            result = simcheck.MonteCarloResult(
                estimates=estimates,
                standard_errors=ses,
                covered=None,
                rejected=None,
                truth=truth,
                lowers=estimates - 1.96 * ses,
                uppers=estimates + 1.96 * ses,
            )
            row = {
                "regime": name,
                "bootstrap": kind,
                "reps": result.reps,
                "bias, % of truth": 100 * result.bias / truth,
                "bias / MC se": result.bias_t,
                "sampling sd": result.sampling_sd,
                "mean reported se": result.reported_se,
                "se ratio": result.se_ratio,
                "95% coverage": result.coverage,
                "se calibrated": gate(simcheck.assert_se_calibrated, result, name),
            }
            if expect_unbiased:
                row["unbiased"] = gate(simcheck.assert_unbiased, result, name)
                row["coverage ok"] = gate(simcheck.assert_coverage, result, 0.95, name)
            rows.append(row)
            print(f"done: {name}, {kind}", flush=True)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    summarize(table).to_csv(SUMMARY, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
