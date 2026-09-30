fairlex: leximin calibration
============================

[![PyPI version](https://img.shields.io/pypi/v/fairlex.svg)](https://pypi.org/project/fairlex/)
[![PyPI Downloads](https://static.pepy.tech/badge/fairlex)](https://pepy.tech/projects/fairlex)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fairlex)](https://pypi.org/project/fairlex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/fairlex/)


``fairlex`` implements risk-averse calibration of survey weights using leximin objectives.
Unlike standard calibration that either (a) hits all margins exactly (sometimes creating
spiky weights) or (b) accepts uneven misses, leximin prioritizes uniform guarantees: it
shrinks the worst margin miss first, then the next worst, and so on, with every weight
kept within a fixed ratio of its base value. Misses are compared as a percentage of each
target by default, so a small group's miss is not swamped by the population total.

Why use it?
-----------

When exact calibration is infeasible under weight caps.

1. When targets are noisy/inconsistent and you want bounded misses rather than fragile exact hits.
2. When you need fairness/stability—no margin (or subgroup) becomes the sacrificial lamb.
3. In rolling waves, to prevent whiplash by bounding the worst per-unit weight changes.

``fairlex`` is designed to be both easy to use and
flexible enough to support different calibration objectives. The two
principal calibration strategies are:

* **Residual leximin** (``leximin_residual``) – finds weights whose margin
  misses are leximin-optimal: the largest miss is as small as the weight
  bounds allow, then the second largest, and so on. Many weight vectors
  achieve those misses; this function returns one of them.
* **Weight‐fair leximin** (``leximin_weight_fair``) – keeps every margin at
  its leximin miss (plus an optional ``slack``) and, among those weights,
  makes the relative weight changes ``|w - w0| / w0`` leximin-optimal too.
  A group's adjustment is spread evenly over its members, and units no
  margin needs to move keep their base weight.

Both are solved as sequences of linear programmes (SciPy's HiGHS) over the
distinct membership patterns in ``A``, so with 0/1 margins the cost barely
depends on the number of respondents.

Installation
------------

``fairlex`` requires Python 3.12+ and depends on ``numpy>=1.26`` and
``scipy>=1.12``. Install it from PyPI:

```bash
pip install fairlex
```

For development, clone this repository and sync the environment with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/finite-sample/fairlex.git
cd fairlex
uv sync --all-groups
```

`make help` lists the available development targets; `make ci` runs the same
checks CI does.

Usage
-----

Construct a membership matrix ``A`` of shape ``(m, n)``, where each row
corresponds to a margin and each column to a survey unit. Each entry
represents whether the unit belongs to the margin (1.0 or 0.0 for simple
groups). Supply the target totals ``b``, the base weights ``w0`` and call
the desired calibration function:

```python
import numpy as np
from fairlex import leximin_weight_fair, evaluate_solution

# Example data: two margins (sex and age) plus total
A = np.array(
    [
        # sex: female
        [1, 0, 1, 0, 1],
        # sex: male
        [0, 1, 0, 1, 0],
        # age: young
        [1, 1, 0, 0, 1],
        # age: old
        [0, 0, 1, 1, 0],
        # total
        [1, 1, 1, 1, 1],
    ],
    dtype=float,
)
target = np.array([6, 4, 6, 4, 10], dtype=float)  # Feasible targets
w0 = np.array([1, 1, 1, 1, 1], dtype=float)

# Calibrate using weight‐fair leximin
res = leximin_weight_fair(A, target, w0, min_ratio=0.5, max_ratio=2.0)

# Inspect the weights and diagnostics
weights = res.w
metrics = evaluate_solution(A, target, weights, base_weights=w0)
print(metrics)
```

``res.residuals`` holds the raw misses ``A @ w - b``, ``res.epsilon`` the
largest miss as a fraction of its target (``scale="absolute"`` compares raw
misses instead) and ``res.t`` the largest relative weight change.

``evaluate_solution`` returns a dictionary with a variety of diagnostics,
including the largest absolute and relative residual, effective sample size
(ESS), design effect and the requested quantiles of the weight distribution
(``weight_p99``, ``weight_p95``, ``weight_p50`` by default). If you supply
the base weights via ``base_weights``, it also reports relative deviations
from the original weights, over units whose base weight is positive.
