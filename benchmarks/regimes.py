"""Where leximin calibration helps: a synthetic regime study.

Question: when calibration targets disagree with each other and weights are
capped, how do leximin, the practitioner default (raking to shares with
trimming) and ridge calibration compare?

Population: 100,000 people with age (5 levels), education (4), race (5, two
groups under 7%) and region (4); age and education share a latent factor.
Sample: 2,000 drawn with inclusion probabilities that favour older, more
educated and non-Hispanic people; base weights are equal (N / n), so a cap on
weight ratios equals a cap on weights.

Targets: each variable's category totals come from a "different source":
the true totals times (1 + a variable-level error) times (1 + a per-category
error), both normal with sd sigma, plus the exact population count. Sources
therefore disagree once sigma > 0.

Measures, per sample and method:

* ``worst_miss_vs_targets``: max_j |A_j w - b_j| / b_j, the worst-case bias
  bound of the theory page. Among methods that respect the weight cap and
  calibrate to these targets, leximin must be lowest; anything else is a bug.
* ``small_group_error``: worst relative error of the estimated size of the two
  small race groups, against the TRUE population.
* ``outcome_rmse``: error of weighted totals for 100 random outcomes that load
  on every category plus an age-by-education interaction the margins miss,
  standardised by N * sd(y). RMSE over outcomes and samples.
* ``deff``: Kish design effect.
* ``cap_exceeded``: share of samples whose weights leave [w0 / cap, w0 * cap].

Each metric is averaged over the samples in a regime and reported with its
standard error across samples (``*_se``).

"fairlex on shares" first rescales each variable's targets to shares of the
population count (``to_shares``), as raking to shares does, then calibrates.
Its misses are still measured against the original targets.

Run: ``uv run --group bench python benchmarks/regimes.py``. Writes
``benchmarks/results/regimes.csv`` (one row per regime and method) and the
compact ``regimes_summary.csv`` shown in the docs.

Ridge calibration is hand-rolled because no maintained Python package
offers inexact (penalised) calibration: svy's ``calibrate`` is exact GREG and
its ``rake`` rejects disagreeing totals unless given shares.
"""

from pathlib import Path

import numpy as np
import polars as pl
import svy
from scipy.optimize import minimize

from fairlex import leximin_weight_fair
from fairlex.metrics import design_effect

N_POP, N_SAMPLE, REPS, N_OUTCOMES = 100_000, 2_000, 30, 100
SIGMAS, CAPS = (0.0, 0.03, 0.08), (2.0, 4.0)
RIDGE_LAMBDAS = (10.0, 1000.0)
LEVELS = {"age": 5, "edu": 4, "race": 5, "region": 4}
RACE_SHARES = (0.60, 0.13, 0.17, 0.06, 0.04)
SMALL_RACE = (3, 4)
HISPANIC = 2  # under-sampled race group
OUT = Path(__file__).parent / "results" / "regimes.csv"
SUMMARY = Path(__file__).parent / "results" / "regimes_summary.csv"


def make_population(rng):
    """Draw the synthetic population as a dict of category codes."""
    z = rng.normal(size=N_POP)
    age = np.digitize(z + rng.normal(size=N_POP), np.quantile(z, [0.2, 0.4, 0.6, 0.8]))
    edu = np.digitize(0.8 * z + rng.normal(size=N_POP), [-1.0, 0.0, 1.0])
    race = rng.choice(5, size=N_POP, p=RACE_SHARES)
    region = rng.choice(4, size=N_POP)
    return {"age": age, "edu": edu, "race": race, "region": region}


def membership(cats, idx=None):
    """Membership matrix: one row per category of every variable, plus total."""
    rows = []
    for var, k in LEVELS.items():
        col = cats[var] if idx is None else cats[var][idx]
        rows += [col == j for j in range(k)]
    n = len(rows[0])
    return np.vstack([np.array(rows, float), np.ones(n)])


def draw_sample(pop, rng):
    """Draw a sample that over-represents older, educated, non-Hispanic people."""
    logit = 0.2 * pop["age"] + 0.25 * pop["edu"] - 0.7 * (pop["race"] == HISPANIC)
    p = np.exp(logit)
    return rng.choice(N_POP, size=N_SAMPLE, replace=False, p=p / p.sum())


def make_targets(true_b, sigma, rng):
    """Perturb each variable's totals as if it came from its own source."""
    b, start = true_b.copy(), 0
    for k in LEVELS.values():
        block = slice(start, start + k)
        b[block] *= (1 + sigma * rng.normal()) * (1 + sigma / 2 * rng.normal(size=k))
        start += k
    return np.maximum(b, 1.0)


def make_outcomes(pop, rng):
    """Random outcomes: additive in every category plus an interaction."""
    y = np.zeros((N_OUTCOMES, N_POP))
    for var, k in LEVELS.items():
        y += rng.normal(size=(N_OUTCOMES, k))[:, pop[var]]
    inter = rng.normal(size=(N_OUTCOMES, 5, 4))[:, pop["age"], pop["edu"]]
    y += 0.5 * inter + rng.normal(size=(N_OUTCOMES, N_POP))
    return y


def to_shares(b):
    """Rescale each variable's category totals to sum to the population count.

    This is what raking to shares does implicitly: it keeps each source's
    proportions and discards its disagreement about the overall size.
    """
    out, start = b.copy(), 0
    for k in LEVELS.values():
        block = slice(start, start + k)
        out[block] *= N_POP / b[block].sum()
        start += k
    out[-1] = N_POP
    return out


def rake_trim(pop, idx, b, w0, cap):
    """Practitioner default: rake to shares (svy normalises), trim, repeat."""
    data = {var: pop[var][idx].astype(str) for var in LEVELS}
    df = pl.DataFrame({**data, "w": w0})
    shares, start = {}, 0
    for var, k in LEVELS.items():
        shares[var] = {str(j): float(b[start + j]) for j in range(k)}
        start += k
    sample = svy.Sample(df, svy.Design(wgt="w"))
    trim = svy.TrimConfig(upper=float(w0[0] * cap), lower=float(w0[0] / cap))
    out = sample.weighting.rake(
        shares=shares, trimming=trim, strict=False, wgt_name="rk"
    )
    return out.data["rk"].to_numpy()


def ridge(A, b, w0, cap, lam):
    """Minimise mean((g-1)^2) + lam * sum_j ((A_j w - b_j) / b_j)^2, g in bounds.

    Solved per membership cell (units in a cell share g at the optimum by
    convexity and symmetry), with L-BFGS-B on the box bounds.
    """
    patterns, cell = np.unique(A.T, axis=0, return_inverse=True)
    cell = cell.ravel()
    mass = np.bincount(cell, weights=w0)
    coef = patterns.T * mass / b[:, None]
    share = mass / mass.sum()

    def f(g):
        r = coef @ g - 1.0
        return share @ (g - 1) ** 2 + lam * r @ r, 2 * share * (
            g - 1
        ) + 2 * lam * coef.T @ r

    res = minimize(
        f,
        np.ones(len(mass)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(1 / cap, cap)] * len(mass),
    )
    return w0 * res.x[cell]


def evaluate(w, w0, A, b, truth, cap):
    """Score one set of weights; ``truth`` is (true_b, y_sample, y_total, y_sd)."""
    true_b, y_sample, y_total, y_sd = truth
    small_rows = [LEVELS["age"] + LEVELS["edu"] + k for k in SMALL_RACE]  # race rows
    est = A @ w
    ratio = w / w0
    err = (y_sample @ w - y_total) / (N_POP * y_sd)
    return {
        "worst_miss_vs_targets": float(np.max(np.abs(est - b) / b)),
        "small_group_error": float(
            np.max(np.abs(est[small_rows] - true_b[small_rows]) / true_b[small_rows])
        ),
        "outcome_sq_err": float(np.mean(err**2)),
        "deff": design_effect(w),
        "cap_exceeded": float(
            np.any(ratio > cap * (1 + 1e-6)) or np.any(ratio < (1 / cap) * (1 - 1e-6))
        ),
    }


def summarize(table):
    """Compact table for the docs: key metrics as percentages, mean (se)."""

    def fmt(col, scale=100.0, digits=1):
        return pl.format(
            "{} ({})",
            (pl.col(col) * scale).round(digits),
            (pl.col(f"{col}_se") * scale).round(digits),
        )

    return table.select(
        pl.format("{}%", (pl.col("sigma") * 100).round(0).cast(pl.Int64)).alias(
            "target noise"
        ),
        pl.format("{}x", pl.col("cap").cast(pl.Int64)).alias("cap"),
        pl.col("method"),
        fmt("worst_miss_vs_targets").alias("worst miss vs targets, %"),
        fmt("small_group_error").alias("small-group error vs truth, %"),
        fmt("outcome_rmse").alias("outcome RMSE, % of N sd"),
        fmt("deff", 1.0, 2).alias("design effect"),
        (pl.col("cap_exceeded") * 100).round(0).alias("cap exceeded, % of samples"),
    )


def main():
    """Run the grid and write one row per regime and method."""
    rng = np.random.default_rng(20260930)
    pop = make_population(rng)
    true_b = membership(pop).sum(axis=1)
    y = make_outcomes(pop, rng)
    y_total, y_sd = y.sum(axis=1), y.std(axis=1)

    rows = []
    for sigma in SIGMAS:
        for cap in CAPS:
            acc = {}
            for _ in range(REPS):
                idx = draw_sample(pop, rng)
                A = membership(pop, idx)
                w0 = np.full(N_SAMPLE, N_POP / N_SAMPLE)
                b = make_targets(true_b, sigma, rng)
                methods = {
                    "base weights": w0,
                    "fairlex (relative)": leximin_weight_fair(
                        A, b, w0, min_ratio=1 / cap, max_ratio=cap
                    ).w,
                    "fairlex on shares": leximin_weight_fair(
                        A, to_shares(b), w0, min_ratio=1 / cap, max_ratio=cap
                    ).w,
                    "svy rake + trim": rake_trim(pop, idx, b, w0, cap),
                }
                for lam in RIDGE_LAMBDAS:
                    methods[f"ridge (lambda={lam:g})"] = ridge(A, b, w0, cap, lam)
                for name, w in methods.items():
                    truth = (true_b, y[:, idx], y_total, y_sd)
                    m = evaluate(w, w0, A, b, truth, cap)
                    for key, val in m.items():
                        acc.setdefault(name, {}).setdefault(key, []).append(val)
            for name, metrics in acc.items():
                row = {"sigma": sigma, "cap": cap, "method": name}
                for key, per_sample in metrics.items():
                    vals = np.asarray(per_sample)
                    se = vals.std(ddof=1) / np.sqrt(len(vals))
                    if key == "outcome_sq_err":
                        # Delta method: se(sqrt(m)) = se(m) / (2 sqrt(m)).
                        rmse = float(np.sqrt(vals.mean()))
                        row["outcome_rmse"] = rmse
                        row["outcome_rmse_se"] = float(se / (2 * rmse))
                    else:
                        row[key] = float(vals.mean())
                        row[f"{key}_se"] = float(se)
                rows.append(row)
            print(f"done sigma={sigma} cap={cap}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows)
    table.write_csv(OUT)
    summarize(table).write_csv(SUMMARY)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(table)


if __name__ == "__main__":
    main()
