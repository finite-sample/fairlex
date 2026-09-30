fairlex: Leximin Calibration for Survey Weights
===============================================

**fairlex** is a Python package for leximin calibration of survey weights. It implements two primary calibration strategies designed to balance fairness in both margin errors and weight adjustments.

Features
--------

- **Residual leximin**: makes the largest margin miss as small as possible, then the next largest, and so on
- **Weight-fair leximin**: keeps those misses and makes the relative weight changes leximin-fair as well
- Misses compared as a percentage of each target by default (``scale="relative"``)
- Linear programmes solved with SciPy's HiGHS over distinct membership cells, so large samples are cheap
- Comprehensive solution evaluation and diagnostic metrics
- Type-safe implementation with full type annotations

Installation
------------

Install fairlex from PyPI:

.. code-block:: bash

    pip install fairlex

Or for development:

.. code-block:: bash

    git clone https://github.com/finite-sample/fairlex.git
    cd fairlex
    uv sync --all-groups

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from fairlex import leximin_weight_fair, evaluate_solution

    # Example: Survey of 5 people, calibrate on sex and age
    A = np.array([
        [1, 0, 1, 0, 1],  # female
        [0, 1, 0, 1, 0],  # male
        [1, 1, 0, 0, 1],  # young (<=40)
        [0, 0, 1, 1, 0],  # old (>40)
        [1, 1, 1, 1, 1],  # total
    ], dtype=float)
    
    w0 = np.ones(5)  # base weights (equal)
    target = np.array([6, 4, 6, 4, 10], dtype=float)  # target totals
    
    # Perform weight-fair leximin calibration
    result = leximin_weight_fair(A, target, w0, min_ratio=0.5, max_ratio=2.0)
    
    print("Calibrated weights:", result.w)
    print("Largest relative miss:", result.epsilon)
    print("Max relative weight change:", result.t)
    
    # Evaluate solution quality
    diagnostics = evaluate_solution(A, target, result.w, base_weights=w0)
    print("Effective sample size:", diagnostics['ESS'])
    print("Design effect:", diagnostics['deff'])

Calibration Methods
-------------------

fairlex provides two calibration approaches:

**leximin_residual**
    Finds weights whose margin misses are leximin-optimal: the largest miss is as small as
    the weight bounds allow, then the second largest given that, and so on. The misses are
    unique; the weights achieving them usually are not, and this function returns one set.

**leximin_weight_fair**
    Keeps every margin at its leximin miss (plus an optional ``slack``) and, among those
    weights, makes the relative changes ``|w - w0| / w0`` leximin-optimal. Each group's
    adjustment is spread evenly over its members, and units no margin needs to move stay
    at their base weight.

Both methods take weight bounds as multiplicative ratios of the base weights and a
``scale`` argument: ``"relative"`` (default) compares misses as a fraction of each target,
``"absolute"`` compares raw misses.

Contents
--------

.. toctree::
   :maxdepth: 2

   theory
   regimes
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`