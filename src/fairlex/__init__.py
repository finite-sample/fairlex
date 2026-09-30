"""Leximin calibration of survey weights.

:func:`calibrate` takes a data frame and targets for each variable and
returns weights whose misses are shared out as evenly as possible across the
margins: the largest relative miss is as small as the weight bounds allow,
then the next largest, and so on. Among weights with those misses, it moves
each respondent as little and as evenly as possible.
:func:`calibrate_replicates` does the same for replicate weights, for
standard errors.
"""

from importlib.metadata import version

__all__ = ["CalibrationReport", "calibrate", "calibrate_replicates"]

from .frame import CalibrationReport, calibrate, calibrate_replicates

__version__ = version("fairlex")
