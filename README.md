fairlex: leximin calibration
============================

[![PyPI version](https://img.shields.io/pypi/v/fairlex.svg)](https://pypi.org/project/fairlex/)
[![PyPI Downloads](https://static.pepy.tech/badge/fairlex)](https://pepy.tech/projects/fairlex)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fairlex)](https://pypi.org/project/fairlex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/fairlex/)

When your survey weights cannot hit every target, something has to give.
``fairlex`` decides what gives by a rule you can state and defend: make the worst
percentage miss across your margins as small as possible, then the next worst, and so
on, while keeping every weight within a fixed ratio of its base weight.

Targets often cannot all be hit. They come from different sources that disagree (census
age by sex, a voter file for party, last year's survey for region), and weight caps
limit how far any respondent can be pushed. Raking then stops wherever it stops, and
penalised calibration spreads the misses according to a tuning constant nobody reports.
Either way, a small group can quietly end up far off its target.

The name is about that choice. "Fair" means sharing unavoidable misses across margins,
not machine-learning fairness.

What you get
------------

A small group is not sacrificed to the big margins. In the example below, three targets
contradict each other by 30 people. Comparing raw counts puts almost the whole miss on a
group of 30, which ends up 16.7% short. fairlex's default spreads it as an equal 1.48%
on every margin.

You get one number that bounds the damage. ``epsilon``, the largest scaled miss, is the
most bias the misses can add to *any* weighted estimate, among outcomes whose dependence
on the margins is limited by the scale you chose. The
[documentation](https://finite-sample.github.io/fairlex/) proves this in one line and
says what it does not cover.

You set the priorities. Pass one scale per margin, and a margin with a smaller scale is
protected more. Giving the small group four times the priority cuts its miss from 1.48%
to 0.38%.

Weights move as little as they can. Among weights with the best misses,
``leximin_weight_fair`` makes the relative changes ``|w - w0| / w0`` leximin-optimal
too. Each group's adjustment is spread evenly over its members, units no margin needs to
move keep their base weight, and the design effect is bounded by the largest change.
Pass last wave's weights as ``w0`` and a new wave moves each respondent as little as the
new targets allow.

It is fast. Respondents with the same margin memberships share one ratio, so the linear
programmes grow with the number of distinct membership patterns, not respondents: 50,000
respondents on five 0/1 margins take about 0.1 s.

The documentation's "Where it helps" page puts numbers on this with a simulation of
disagreeing sources and capped weights. When sources disagree, fairlex's worst miss is
about a third lower than ridge calibration's, and small groups land closer to the
truth. The cost is somewhat higher average error when targets are nearly consistent.
Normalising each source to shares before calling fairlex keeps most of raking's
accuracy without breaking the weight caps, which raking with trimming broke in 90% to
100% of samples.

When not to use it
------------------

If your targets are consistent and reachable within your weight caps, every calibration
method hits them, and the choice among methods matters less. Standard raking in
[svy](https://github.com/samplics-org/svy) or [balance](https://github.com/facebookresearch/balance)
is fine. If you care about a single outcome and know how it depends on the margins, a
method tuned to that outcome can do better on it than a worst-case guarantee.

Installation
------------

``fairlex`` requires Python 3.12+ and depends on ``numpy>=1.26`` and
``scipy>=1.12``:

```bash
pip install fairlex
```

Usage
-----

Each row of the membership matrix ``A`` is a margin and each column a respondent
(1 if the respondent belongs to the margin, else 0). ``b`` holds the target totals and
``w0`` the base weights.

```python
import numpy as np
from fairlex import leximin_weight_fair

# 1,000 respondents: 30 in a small group, 970 outside it
group = np.r_[np.ones(30), np.zeros(970)]
A = np.vstack([np.ones(1000), group, 1 - group])
# Targets from different sources: 60 + 970 = 1030, but the total says 1000
b = np.array([1000.0, 60.0, 970.0])
w0 = np.r_[np.full(30, 2.0), np.ones(970)]

res = leximin_weight_fair(A, b, w0, min_ratio=0.5, max_ratio=2.0)
print(res.residuals / b)  # [ 0.0148 -0.0148 -0.0148]: an equal 1.48% miss each
print(res.epsilon, res.t)  # largest relative miss 1.48%, largest weight change 1.48%

# Raw counts instead of percentages: the small group absorbs the miss
leximin_weight_fair(A, b, w0, min_ratio=0.5, max_ratio=2.0, scale="absolute")
# relative misses: +1.00%, -16.67%, -1.03%

# Four times the priority for the small group: a smaller scale means more protection
s = np.abs(b)
s[1] /= 4
leximin_weight_fair(A, b, w0, min_ratio=0.5, max_ratio=2.0, scale=s)
# relative misses: +1.51%, -0.38%, -1.51%
```

``leximin_residual`` stops after the first stage and returns some weights achieving the
best misses. Use it when only the misses matter. ``leximin_weight_fair`` is the usual
choice.

``res.residuals`` holds the raw misses ``A @ w - b``, ``res.epsilon`` the largest scaled
miss and ``res.t`` the largest relative weight change. ``slack`` lets every margin miss by
a little more than its leximin level in exchange for smaller weight changes.

``evaluate_solution(A, b, res.w, base_weights=w0)`` reports the largest absolute and
relative miss, effective sample size, design effect, weight quantiles
(``weight_p99``, ``weight_p95``, ``weight_p50`` by default) and relative weight changes.

Development
-----------

```bash
git clone https://github.com/finite-sample/fairlex.git
cd fairlex
uv sync --all-groups
```

`make help` lists the development targets; `make ci` runs the same checks CI does.
