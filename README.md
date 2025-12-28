fairlex: leximin calibration
============================

[![PyPI version](https://img.shields.io/pypi/v/fairlex.svg)](https://pypi.org/project/fairlex/)
[![PyPI Downloads](https://static.pepy.tech/badge/fairlex)](https://pepy.tech/projects/fairlex)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fairlex)](https://pypi.org/project/fairlex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


``fairlex`` implements risk-averse calibration of survey weights using leximin objectives. 
Unlike standard calibration that either (a) hits all margins exactly (sometimes creating 
spiky weights) or (b) accepts uneven misses, leximin prioritizes uniform guarantees: it 
shrinks the worst margin error first (then the next worst, etc.) and can also cap how 
far any weight is allowed to move from its base value.

Why use it?
-----------

When exact calibration is infeasible under weight caps.

1. When targets are noisy/inconsistent and you want bounded misses rather than fragile exact hits.
2. When you need fairness/stability—no margin (or subgroup) becomes the sacrificial lamb.
3. In rolling waves, to prevent whiplash by bounding the worst per-unit weight changes.

``fairlex`` is designed to be both easy to use and
flexible enough to support different calibration objectives. The two
principal calibration strategies are:

* **Residual leximin** – finds weights that minimise the worst absolute
  deviation from the target margins (``min–max`` residuals). This can drive
  margin errors down to machine precision, but may result in large weight
  adjustments.
* **Weight‐fair leximin** – first performs residual leximin, then
  minimises the largest relative change from the base weights while keeping
  residuals at their optimum level. This yields a more stable set of weights.

Installation
------------

``fairlex`` requires Python 3.12+ and depends on ``numpy>=1.26`` and
``scipy>=1.11``. You can install it via pip once uploaded to PyPI:

```bash
pip install fairlex
```

For development, clone this repository and install the dependencies:

```bash
git clone https://github.com/finite-sample/fairlex.git
cd fairlex
pip install -e .[dev]
```

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
A = np.array([
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
], dtype=float)
target = np.array([6, 4, 6, 4, 10], dtype=float)  # Feasible targets
w0 = np.array([1, 1, 1, 1, 1], dtype=float)

# Calibrate using weight‐fair leximin
res = leximin_weight_fair(A, target, w0, min_ratio=0.5, max_ratio=2.0)

# Inspect the weights and diagnostics
weights = res.w
metrics = evaluate_solution(A, target, weights, base_weights=w0)
print(metrics)
```

``evaluate_solution`` returns a dictionary with a variety of diagnostics,
including the maximum absolute residual, effective sample size (ESS), design
effect and quantiles of the weight distribution. If you supply the base
weights via ``base_weights``, it also reports relative deviations from the
original weights.
