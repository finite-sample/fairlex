fairlex: Leximin Calibration for Survey Weights
===============================================

**fairlex** calibrates survey weights to population targets. When the targets cannot all be hit, because sources disagree or weights are capped, it shares the misses out as evenly as possible and tells you which targets are the problem.

Features
--------

- Leximin calibration: the largest relative miss across your targets is as small as the
  weight bounds allow, then the next largest, and so on
- Weights move as little and as evenly as possible among those with the best misses
- A report naming the worst-missed targets and why they miss
- Per-variable priorities (``importance=``), targets as totals or shares
- Replicate weights for standard errors (``calibrate_replicates``)

Installation
------------

.. code-block:: bash

    pip install fairlex

Quick Start
-----------

.. code-block:: python

    import pandas as pd
    from fairlex import calibrate

    df = pd.DataFrame({
        "sex": ["f", "m", "f", "m", "f"],
        "age": ["young", "young", "old", "old", "young"],
    })
    targets = {"sex": {"f": 5, "m": 5}, "age": {"young": 7, "old": 3}}

    report = calibrate(df, targets, total=10, bounds=(0.5, 2.0))
    print(report)          # misses, weight changes, design effect
    report.weights         # calibrated weights, indexed like df

Contents
--------

.. toctree::
   :maxdepth: 2

   theory
   standard_errors
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`