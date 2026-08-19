"""Top level package for fairlex.

This package provides routines for leximin-style calibration of survey weights.

Two primary calibration strategies are exposed:

* ``leximin_residual`` - minimises the worst absolute margin residual across all
  constraints (min-max), optionally refining the next worst in lexicographic
  order. This approach will tend to squeeze margin errors to near zero at the
  cost of increased leverage on the weights.

* ``leximin_weight_fair`` - after achieving the smallest possible worst
  residual, this method minimises the largest relative change from the base
  weights. It balances fairness in both the errors and the weight movements,
  offering a compromise between calibration accuracy and variance inflation.

The core implementation lives in :mod:`fairlex.calibration`. Convenience
functions and metric helpers live in :mod:`fairlex.metrics`.

"""

from importlib.metadata import version

__all__ = [
    "CalibrationResult",
    "evaluate_solution",
    "leximin_residual",
    "leximin_weight_fair",
]

# Public API
from .calibration import CalibrationResult, leximin_residual, leximin_weight_fair
from .metrics import evaluate_solution

# Expose the package version at runtime
__version__ = version("fairlex")
