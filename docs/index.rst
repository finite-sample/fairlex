fairlex: Leximin Calibration for Survey Weights
===============================================

**fairlex** is a Python package for leximin calibration of survey weights. It implements two primary calibration strategies designed to balance fairness in both margin errors and weight adjustments.

Features
--------

- **Residual leximin**: Minimizes the worst absolute deviation from target margins
- **Weight-fair leximin**: Performs residual leximin then minimizes largest relative change from base weights
- Fast linear programming implementation using SciPy's HiGHS solver
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
    pip install -e .[dev]

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
    print("Max residual:", result.epsilon)
    print("Max relative weight change:", result.t)
    
    # Evaluate solution quality
    diagnostics = evaluate_solution(A, target, result.w, base_weights=w0)
    print("Effective sample size:", diagnostics['ESS'])
    print("Design effect:", diagnostics['deff'])

Calibration Methods
-------------------

fairlex provides two calibration approaches:

**leximin_residual**
    Minimizes the worst absolute margin residual across all constraints (min-max problem). 
    This approach will tend to squeeze margin errors to near zero at the cost of increased 
    leverage on the weights.

**leximin_weight_fair**
    After achieving the smallest possible worst residual, this method minimizes the largest 
    relative change from the base weights. It balances fairness in both the errors and the 
    weight movements, leading to more stable calibrated weights.

Both methods support:
- Flexible weight bounds specified as multiplicative ratios
- Optional slack parameters for practical tolerance
- Comprehensive diagnostic output

Contents
--------

.. toctree::
   :maxdepth: 2

   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`