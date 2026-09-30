Standard errors
===============

fairlex returns weights, not variances. The way to get standard errors for
estimates built on those weights is the one survey statisticians use for any
calibration: replicate weights.

1. Generate replicate base weights with your survey package: a bootstrap,
   jackknife or BRR from svy (``weighting.create_bs_wgts``) or R ``survey``
   (``as.svrepdesign``).
2. Calibrate every replicate exactly as you calibrated the full sample:

   .. code-block:: python

      from fairlex import calibrate, calibrate_replicates

      full = calibrate(df, targets, base_weight="w0", total=N, bounds=(0.5, 2))
      reps = calibrate_replicates(df, targets, rep_cols, total=N, bounds=(0.5, 2))

3. Compute your estimate with ``full.weights`` and with each column of
   ``reps``, and combine the replicate estimates with the formula for your
   replicate type. For a bootstrap, that is the root mean square of the
   replicate estimates around the full-sample estimate.

Match the bootstrap to the design
---------------------------------

When the weight bounds bind, calibration can no longer absorb the chance
variation in how many people respond. The estimate then moves with the
realised sample size, and a replicate scheme that holds the sample size fixed
misses that variation. Under Poisson sampling, where the sample size is
random, use svy's ``kind="poisson"`` bootstrap rather than Rao-Wu. This is a
general rule for calibrated estimates, not something specific to fairlex; the
bounds are what expose it.

Evidence
--------

``benchmarks/se_study.py`` checks the recipe with simcheck. That library's
pass bands are set by the number of simulated samples, not chosen by hand.

* **Population.** 20,000 people. The outcome depends on age, education and
  region, plus an age-by-education interaction that the margins miss.
* **Sample.** Poisson sampling that favours older, more educated people, with
  an expected size of 1,000. Nonresponse then depends on age, with response
  rates from 35% to 80%. The base weights undo the selection but not the
  nonresponse.
* **Replication.** 400 simulated samples per regime, each with 200 bootstrap
  replicates. The estimand is the population total of the outcome.

With correct targets and loose bounds, the estimate is unbiased, its standard
error is right (1.05 to 1.06 times the true spread) and 95% intervals cover
96%. Both bootstrap schemes pass every gate.

With binding bounds (0.7x to 1.3x), the weights cannot undo the nonresponse,
so the estimate is 17% low. That bias is why no interval covers. The Rao-Wu
bootstrap reports a standard error only 0.36 times the true spread. The
estimate tracks the realised sample size almost exactly (correlation 0.98),
and Rao-Wu holds that size fixed. The Poisson bootstrap reports 1.02 times
the true spread.

With disagreeing targets, the estimate inherits the targets' 1.5% error.
Both schemes still measure its sampling spread correctly (1.05 to 1.06);
coverage falls to 84% because of the bias, not because of the standard error.

.. csv-table:: 400 simulated samples per regime, 200 bootstrap replicates each
   :file: ../benchmarks/results/se_study_summary.csv
   :header-rows: 1
