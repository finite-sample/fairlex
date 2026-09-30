Why leximin
===========

This page states what the misses reported by fairlex guarantee, and what
choosing ``scale`` means. Each claim is checked numerically in
``tests/test_theory.py`` by a route that shares no code with the solver.

Setup
-----

Margin :math:`j` has membership row :math:`A_j` and target :math:`b_j`. Weights
:math:`w` leave a miss :math:`r_j = A_j w - b_j`. Let :math:`s_j > 0` be the
scale of margin :math:`j`:

* :math:`s_j = |b_j|` for ``scale="relative"``;
* :math:`s_j = 1` for ``scale="absolute"``;
* any positive numbers you pass as an array.

fairlex reports :math:`\epsilon = \max_j |r_j| / s_j`.

Misses bound the bias of every estimate
---------------------------------------

Take any outcome :math:`y` and write its unit-level values as a linear function
of the margin memberships plus a remainder, :math:`y_i = x_i^\top \beta + e_i`,
where :math:`x_i` is unit :math:`i`'s column of :math:`A`. The weighted total
:math:`\sum_i w_i y_i` then differs from the total implied by the targets,
:math:`b^\top \beta + \sum_{\text{pop}} e`, by

.. math::

   r^\top \beta \;+\; \Big(\sum_i w_i e_i - \sum_{\text{pop}} e\Big).

The first term is the part calibration controls. The second depends on how
well the margins explain :math:`y`, which no choice among these weights can
change.

**Proposition.** Over all outcomes whose loadings satisfy
:math:`\sum_j s_j |\beta_j| \le 1`, the largest possible value of
:math:`|r^\top \beta|` is exactly :math:`\epsilon`.

*Proof.* By Hölder's inequality,
:math:`|r^\top \beta| = |\sum_j (r_j / s_j)(s_j \beta_j)| \le
\max_j (|r_j| / s_j) \sum_j s_j |\beta_j| \le \epsilon`. Equality holds for
:math:`\beta = \operatorname{sign}(r_k)\, e_k / s_k`, where :math:`k` is the
margin with the largest scaled miss. :math:`\square`

So minimising :math:`\epsilon` minimises the worst-case calibration bias over
that set of outcomes. Leximin refines this. Once the margins that attain the
minimum are fixed, it minimises the worst case over outcomes that load only on
the remaining margins, and so on. That is the precise sense in which no
margin is sacrificed.

What ``scale`` means
--------------------

The set of outcomes you are protecting against is
:math:`\sum_j s_j |\beta_j| \le 1`. A small :math:`s_j` allows outcomes that
depend strongly on margin :math:`j`, so misses there are expensive. A large
:math:`s_j` says outcomes depend on margin :math:`j` only weakly, so it may
absorb more miss. **The scale is your statement of which margins matter.**

* ``scale="relative"`` measures each loading per unit of *share* of the
  target: a 1% miss on any margin counts the same. It is the neutral choice
  when you have no view about the outcomes.
* ``scale="absolute"`` treats one person as one person on every margin, so
  large margins dominate.
* A custom array expresses priorities. To make education twice as important
  as the other margins, relative to their targets, halve its scale:

  .. code-block:: python

     s = np.abs(b)
     s[education_rows] /= 2
     leximin_weight_fair(A, b, w0, scale=s)

Adding a total row alongside categories that sum to it is not double
counting. It adds outcomes that load on the total to the protected set. Leave
it out if you do not want that.

Other properties
----------------

* **Unique misses.** The leximin vector of scaled misses is unique. The weights
  achieving it need not be; :func:`fairlex.leximin_weight_fair` picks the ones
  whose relative changes are themselves leximin-optimal.
* **Units do not matter under the relative scale.** Multiplying one margin's
  row and target by a constant leaves every scaled miss unchanged. Under
  ``scale="absolute"`` it changes the result.
* **One categorical variable is capped post-stratification.** If the margins
  are the categories of a single variable, every unit in category :math:`k`
  gets ratio :math:`\operatorname{clip}(b_k / W_k,\ \text{min\_ratio},\
  \text{max\_ratio})`, where :math:`W_k` is the category's base-weight total.
* **The weight stage bounds variance inflation.** Every ratio
  :math:`w_i / w_{0,i}` lies in :math:`[1 - t, 1 + t]`. For :math:`t < 1` the
  Kish design effect therefore satisfies
  :math:`\text{deff}(w) \le \text{deff}(w_0)\,\big((1 + t) / (1 - t)\big)^2`.

What this does not cover
------------------------

The guarantee is about the calibration term :math:`r^\top \beta` only, for
outcomes in the chosen set. It says nothing about the remainder term or about
sampling variance beyond the design-effect bound. It also does not say that
leximin minimises average error for a *particular* outcome: if you know which
outcome you care about, a method tuned to it can do better on it.
