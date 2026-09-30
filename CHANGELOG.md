# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

## [0.4.0] - 2026-09-29

### Fixed

- `leximin_residual` was only min-max: it minimised the largest miss and left
  the rest wherever the solver landed. It now computes the full leximin
  vector (saturation with dual variables, Ogryczak & Śliwiński 2006).
- `CalibrationResult.epsilon` reported the first stage's optimum even when
  `slack` let the final weights miss by more; it now reports the achieved
  miss.
- The weight stage built a dense `(2m + 2n) x n` constraint matrix (about
  160 GB at n = 100,000). Problems are now solved on sparse matrices over
  distinct membership cells, in ratio space `w / w0`, which also fixes
  failures with very small or very large weights.
- Results are leximin-optimal to about 1e-5 of the target scale: each
  round's level is held with a 3e-7 relative tolerance, above HiGHS's 1e-7
  feasibility tolerance. Tighter settings made the weight stage report
  feasible survey problems as infeasible.
- `evaluate_solution` labelled whatever quantiles were requested as
  `weight_p99` / `weight_p95`, and divided by 1 for zero base weights.

### Changed

- **Breaking:** misses are compared relative to each target by default
  (`scale="relative"`); pass `scale="absolute"` for raw misses.
- **Breaking:** `leximin_weight_fair` makes the relative weight changes
  leximin-optimal instead of only minimising the largest one, which had let
  every unit move by the maximum. `slack` is per margin, in `scale` units.
- **Breaking:** `leximin_weight_fair` always returns one `CalibrationResult`;
  `return_stages` is gone (call `leximin_residual` for the first stage).
  `CalibrationResult` gains `residuals`, and `t` is `None` on failure.
- **Breaking:** invalid inputs (non-finite values, negative base weights,
  `min_ratio > max_ratio`, zero targets under a relative scale) raise
  `ValueError` instead of surfacing as solver failures.
- **Breaking:** requires SciPy >= 1.12 for the sparse-array constructors
  (`block_array`, `eye_array`, `diags_array`).
- **Breaking:** `evaluate_solution` names quantile keys after the request
  (`weight_p50` replaces `weight_median`), drops `total_error` (it assumed
  the last margin was the population total), adds `resid_max_rel` and
  `n_zero_base_moved`, and raises on non-finite or negative weights and on
  different quantiles close enough to round to the same key.

- Adopted the py-canon fleet standard: reusable CI, docs and release
  workflows, the shared Sphinx configuration, and the canon ruff rule set.
- Moved the package from a flat `fairlex/` layout to `src/fairlex/`.
- Rewrote all docstrings in Google style, the convention the linters are
  configured for.
- Docs notebooks now render through `myst-nb` instead of `nbsphinx`, which
  removes the build's dependency on a system `pandoc` binary.

## [0.3.0] - 2025-12-28

### Added

- `leximin_residual` and `leximin_weight_fair` calibration routines.
- `evaluate_solution`, `effective_sample_size` and `design_effect` diagnostics.

[Unreleased]: https://github.com/finite-sample/fairlex/commits/main
