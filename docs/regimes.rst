Where it helps
==============

A simulation of what the guarantee on the :doc:`theory` page buys in practice,
and what it costs. The script is ``benchmarks/regimes.py``. The table below is
read from its output file, so it cannot drift from the code.

Findings
--------

**When sources disagree, fairlex keeps the worst miss about a third lower than
ridge calibration.** At 8% target noise its worst miss is 9.6% (2x cap) and
9.1% (4x cap), against 13.3% and 13.9% for ridge. With consistent targets the
two tie. This is the guarantee from the theory page, and it held in every
regime.

**Small groups come out closer to the truth when disagreement is large.** At
8% noise the worst error in the size of the two small groups is 6.5% and 8.5%
for fairlex against 8.0% and 10.6% for ridge. At 3% noise or less the
differences are under a percentage point.

**The price is average accuracy and variance.** For typical outcomes, ridge is
up to about 17% more accurate when targets are consistent or only mildly off
(outcome RMSE 1.5 against 1.8 at no noise and a 4x cap), and fairlex's design
effect is up to 0.1 higher.

**Raking with trimming looks most accurate but does not keep its caps.**
svy's rake-then-trim cycle ended outside the weight cap in 90% to 100% of
samples, because the final raking pass undoes the trimming.

**Normalising to shares first gets most of raking's accuracy and keeps the
caps.** Rescaling each variable's targets to shares of one population count
before calling fairlex matches raking on small-group error (5.8% vs 5.7% and
6.1% vs 6.1% at 8% noise). Its average error is within 0.2 to 0.5 points of
raking's, and it never breaks a cap. It gives up some worst-case miss against
the original targets, because it deliberately ignores each source's total.

The last finding depends on how the simulation makes sources disagree. Here
each source is off by a common factor for all its categories, plus smaller
category-level noise. Normalising to shares removes the common factor exactly.
When sources disagree about proportions rather than size, normalising will not
help. Calibrate to the original targets then.

Normalising is a few lines of NumPy. For each variable, rescale its category
targets to sum to the population count:

.. code-block:: python

   for rows in variable_rows:  # row indices of each variable's categories
       b[rows] *= population_count / b[rows].sum()

Design
------

* **Population.** 100,000 people with age (5 levels), education (4), race
  (5, two groups under 7%) and region (4). Age and education share a latent
  factor.
* **Sample.** 2,000 people, drawn with probabilities that favour older, more
  educated and non-Hispanic people. Base weights are equal, so a cap on weight
  ratios is a cap on weights.
* **Targets.** Each variable's category totals come from its own "source": the
  true totals times (1 + a variable-level error) times (1 + a category-level
  error), with standard deviations sigma and sigma / 2, plus the exact
  population count.
* **Methods.** Base weights; fairlex with relative scale; fairlex on
  share-normalised targets; svy raking to shares with trimming at the cap; and
  ridge calibration, which penalises squared relative misses at two strengths
  (lambda 10 and 1000).
* **Measures.** Worst relative miss against the given targets; worst error in
  the size of the two small race groups against the true population; RMSE of
  weighted totals for 100 random outcomes, each loading on every category plus
  an age-by-education interaction that no margin captures; Kish design effect;
  and the share of samples whose weights break the cap.
* **Replication.** 30 samples per regime. Cells show the mean, with the
  standard error across samples in parentheses. All methods see the same
  samples, so differences between methods are estimated more precisely than
  the standard errors suggest.

Results
-------

.. csv-table:: Mean (standard error) over 30 samples per regime
   :file: ../benchmarks/results/regimes_summary.csv
   :header-rows: 1
