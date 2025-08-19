# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fairlex` is a Python package for leximin calibration of survey weights. It implements two primary calibration strategies:

- **Residual leximin** (`leximin_residual`) - minimizes the worst absolute deviation from target margins
- **Weight-fair leximin** (`leximin_weight_fair`) - performs residual leximin then minimizes largest relative change from base weights

## Development Commands

### Installation
```bash
# For development with test dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest -q

# Run tests with color output (as used in CI)
pytest -q --color=yes
```

### Running Examples
```bash
# Run the basic example
python examples/basic_example.py
```

## Code Architecture

### Core Package Structure
- `fairlex/calibration.py` - Main calibration algorithms and linear programming solvers
- `fairlex/metrics.py` - Solution evaluation and diagnostic metrics
- `fairlex/__init__.py` - Public API exports

### Key Components

**CalibrationResult**: Dataclass containing calibrated weights, optimization metrics (epsilon, t), and solver status.

**Linear Programming**: All optimization uses `scipy.optimize.linprog` with HiGHS method. The `_solve_lp()` helper centralizes LP calls and provides clear error handling for missing SciPy.

**Input Validation**: `_validate_inputs()` ensures membership matrix A (m×n), target totals b (m,), and base weights w0 (n,) have compatible shapes.

### Algorithm Implementation

Both calibration methods solve sequential linear programming problems:

1. **leximin_residual**: Single-stage min-max problem minimizing worst absolute residual
2. **leximin_weight_fair**: Two-stage approach - first minimizes residuals, then minimizes weight changes subject to residual constraints

Weight bounds are specified as multiplicative ratios relative to base weights (e.g., `min_ratio=0.5, max_ratio=2.0`).

### Dependencies
- **Required**: numpy>=1.21, scipy>=1.6
- **Development**: pytest>=7.0
- **Python**: 3.8+ (tested on 3.9, 3.10, 3.11 in CI)

### Testing Strategy
- CI runs on Python 3.9, 3.10, 3.11
- Tests located in `tests/` directory with comprehensive coverage
- Use `pytest -q` for quiet output matching CI configuration  
- Test categories: input validation, simple cases, edge cases, numerical stability
- All algorithms verified for mathematical correctness