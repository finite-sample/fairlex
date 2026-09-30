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
contradict each other by 30 people. A method that compares misses in raw counts splits
them evenly, 10 people per margin, which leaves the small group 16.7% short of its
target of 60. fairlex compares misses as percentages of their targets and spreads them
as an equal 1.48% on every margin.

You get one number that bounds the damage. The largest relative miss is the most bias
the misses can add to *any* weighted estimate, among outcomes whose dependence on the
margins is limited in proportion to the targets. The
[documentation](https://finite-sample.github.io/fairlex/) proves this in one line and
says what it does not cover.

You set the priorities. ``importance={"race": 4}`` makes misses on race count four times
as much, so race is held closer to its targets at the expense of the other variables.

You learn which targets are the problem. The report names the worst-missed targets and
says whether they conflict with other targets or are out of reach within the weight
bounds.

Weights move as little as they can. Among weights with the best misses, the relative
changes ``|w - w0| / w0`` are leximin-optimal too. Each group's adjustment is spread
evenly over its members, respondents no margin needs to move keep their base weight, and
the design effect is bounded by the largest change. Use last wave's weights as the base
weights and a new wave moves each respondent as little as the new targets allow.

It is fast. Respondents with the same answers on every variable share one ratio, so the
linear programmes grow with the number of distinct answer patterns, not respondents:
50,000 respondents on four variables take about 0.2 s.

When not to use it
------------------

If your targets are consistent and reachable within your weight caps, every calibration
method hits them, and the choice among methods matters less. Standard raking in
[svy](https://github.com/samplics-org/svy) or [balance](https://github.com/facebookresearch/balance)
is fine. If you care about a single outcome and know how it depends on the margins, a
method tuned to that outcome can do better on it than a worst-case guarantee.

Installation
------------

``fairlex`` requires Python 3.12+ and depends on ``numpy>=1.26``, ``pandas>=2.1.1`` and
``scipy>=1.12``:

```bash
pip install fairlex
```

Usage
-----

Give ``calibrate`` your data and the targets for each variable, as
``{variable: {level: target}}``:

```python
import pandas as pd
from fairlex import calibrate

# 1,000 respondents; 30 belong to a small group and carry base weight 2
df = pd.DataFrame(
    {"group": ["small"] * 30 + ["rest"] * 970, "w": [2.0] * 30 + [1.0] * 970}
)
# Sources disagree: the group counts say 60 + 970 = 1,030, the census total says 1,000
report = calibrate(
    df,
    {"group": {"small": 60, "rest": 970}},
    total=1000,
    base_weight="w",
    bounds=(0.5, 2.0),
)
print(report)
```

```text
largest miss: 1.48% of target   largest weight change: 1.5%   ESS: 973   design effect: 1.03
worst-missed targets: group=small (conflicting targets), group=rest (conflicting targets), (total)=all (conflicting targets)
variable level  target  before   after  miss miss_pct
   group small    60.0    60.0    59.1  -0.9   -1.48%
   group  rest   970.0   970.0   955.7 -14.3   -1.48%
 (total)   all 1,000.0 1,030.0 1,014.8  14.8   +1.48%
```

``report.weights`` holds the calibrated weights, indexed like ``df``. The report names the
targets with the worst miss and why they miss. "Conflicting targets" means meeting one
better would push another further off. "Weight bounds" means the target is out of reach
even with every member at the bound. With several variables, pass ``shares=True`` and
proportions when your sources agree on proportions but not on the population size, and
``importance={"race": 4}`` to protect a variable's margins four times as much.

### Standard errors

Generate replicate base weights with your survey package (for example svy's
``create_bs_wgts``) and calibrate each one exactly as the full sample:

```python
from fairlex import calibrate_replicates

reps = calibrate_replicates(df, targets, replicate_columns, total=N, bounds=(0.5, 2.0))
```

Then combine the replicate estimates as your replicate method prescribes. Under Poisson
sampling use a Poisson bootstrap: when the weight bounds bind, a bootstrap that holds the
sample size fixed understates the standard error about threefold. The documentation's
"Standard errors" page has the simulation behind this.

Development
-----------

```bash
git clone https://github.com/finite-sample/fairlex.git
cd fairlex
uv sync --all-groups
```

`make help` lists the development targets; `make ci` runs the same checks CI does.
