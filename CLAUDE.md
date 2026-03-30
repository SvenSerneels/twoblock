# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`twoblock` is a scikit-learn compatible Python library implementing simultaneous two-block dimension reduction for multivariate X and Y data blocks. It provides:
- Dense twoblock (Cook, Forzani & Liu, 2023)
- Sparse twoblock with soft-thresholding variable selection (Serneels, 2025)
- Robust twoblock (`rtb`) with iterative M-estimation reweighting (Serneels, 2025)
- SPADIMO (`spadimo`) for identifying outlier-contributing variables
- Optional GPU acceleration via CuPy

## Build & Install

```bash
pip install -e .                    # editable install
pip install -r requirements.txt     # install dependencies (numpy, scipy, scikit-learn, pandas)
pip install -e ".[gpu]"             # with optional CuPy GPU support
```

## Running Tests

Tests use pytest and require the `data/` directory with CSV files. Run from project root:

```bash
python -m pytest                              # all tests (includes GPU if available)
python -m pytest -m "not gpu"                 # skip GPU tests (used in CI)
python -m pytest src/twoblock/test_twoblock.py  # specific test file
python -m pytest src/twoblock/test_twoblock.py::TestTwoblock::test_fit  # single test
```

## Linting

```bash
flake8 . --select=E9,F63,F7,F82 --show-source    # syntax errors only (CI gate)
flake8 . --max-complexity=10 --max-line-length=127  # full lint
```

## Architecture

- **`src/twoblock/`** - single-package layout under `src/`
  - `twoblock.py` - main `twoblock` class (extends sklearn's `BaseEstimator`, `TransformerMixin`, `RegressorMixin`, `MultiOutputMixin`). Implements `fit(X, Y)` and `predict(Xn)`. The fit method runs two sequential SVD-based deflation loops (one for X components, one for Y components) with optional sparsity via soft-thresholding controlled by `eta_x`/`eta_y`.
  - `rtb.py` - robust twoblock class with iterative M-estimation reweighting; supports Hampel, Fair, and Huber downweighting functions
  - `spadimo.py` - SPADIMO (SPArse DIrections of Maximal Outlyingness) class for identifying which variables contribute to an observation being an outlier; supports multiple robust scale estimators (Qn, MAD, scaleTau2) and GPU acceleration
  - `prepro.py` - `VersatileScaler` class for flexible centering/scaling (mean/median/l1median, std/mad/scaleTau2)
  - `utils.py` - input validation helpers (`_check_input`, `_predict_check_input`) and M-estimation weighting functions (Fair, Huber, Hampel)
  - `_preproc_utilities.py` - robust centering/scaling functions (mean, median, l1median, MAD, Qn, scaleTau2, kstepLTS, `scale_data`)
  - `_gpu_utils.py` - CuPy/NumPy array module abstraction for GPU acceleration
  - `__init__.py` - exports `twoblock`, `rtb`, and `spadimo` classes; defines `__version__`

## Key Conventions

- pandas is imported as `ps` (not the usual `pd`) throughout the codebase
- Class names `twoblock` and `rtb` are lowercase (matching module names)
- Fitted attributes use trailing underscore convention (sklearn style): `coef_`, `x_weights_`, etc.
- CI runs on Python 3.9, 3.10, 3.11, 3.12; main branch is `master`
