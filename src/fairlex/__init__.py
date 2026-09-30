"""Top level package for fairlex.

This package provides routines for leximin-style calibration of survey weights.

Two primary calibration strategies are exposed:

* ``leximin_residual`` - makes the vector of scaled margin misses
  leximin-optimal: the largest miss is minimised, then the next largest, and
  so on.

* ``leximin_weight_fair`` - keeps each margin at its leximin miss and makes
  the relative weight changes leximin-optimal as well, spreading each
  adjustment evenly over the units that supply it.

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
