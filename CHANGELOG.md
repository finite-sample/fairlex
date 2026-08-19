# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

### Changed

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
