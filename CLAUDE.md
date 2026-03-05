# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`twoblock` is a scikit-learn compatible Python library implementing simultaneous two-block dimension reduction for multivariate X and Y data blocks, based on Cook, Forzani & Liu (2023). It supports both standard and sparse variants of the method.

## Build & Install

```bash
pip install -e .                    # editable install
pip install -r requirements.txt     # install dependencies (numpy, scipy, scikit-learn, pandas)
```

## Running Tests

Tests use unittest and require the `data/` directory with cookie NIR/lab CSV files. Run from the project root:

```bash
python -m pytest                              # all tests
python -m pytest src/twoblock/test_twoblock.py  # specific test file
python -m unittest twoblock.test_twoblock       # via unittest runner
```

## Linting

```bash
flake8 . --select=E9,F63,F7,F82 --show-source    # syntax errors only
flake8 . --max-complexity=10 --max-line-length=127  # full lint
```

## Architecture

- **`src/twoblock/`** - single-package layout under `src/`
  - `twoblock.py` - main `twoblock` class (extends sklearn's `BaseEstimator`, `TransformerMixin`, `RegressorMixin`, `MultiOutputMixin`). Implements `fit(X, Y)` and `predict(Xn)`. The fit method runs two sequential SVD-based deflation loops (one for X components, one for Y components) with optional sparsity via soft-thresholding controlled by `eta_x`/`eta_y`.
  - `utils.py` - input validation helpers (`_check_input`, `_predict_check_input`) and constraint functions
  - `_preproc_utilities.py` - robust centering/scaling functions (mean, median, l1median, MAD, scaleTau2, kstepLTS, `scale_data`)
  - `__init__.py` - exports the `twoblock` class; defines `__version__`

- **External dependency**: `twoblock.py` imports `VersatileScaler` from `.prepro`, which is **not in this repo**. This module needs to exist or be provided (likely from the `sprm` package or similar) for the package to work.

## Key Conventions

- pandas is imported as `ps` (not the usual `pd`) throughout the codebase
- The class name `twoblock` is lowercase (matching the module name)
- CI runs on Python 3.9, 3.10, 3.11; main branch is `master`
