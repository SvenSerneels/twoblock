# twoblock
Two-block dense and sparse simultaneous dimension reduction

The dense version is a `scikit-learn` compatible implementation of simultaneous two-block dimension reduction, as proposed in [1].

The sparse version is a `scikit-learn` compatible implementation of sparse twoblock dimension reduction, recently published by the author [2].

The robust version (`rtb`) extends twoblock with iterative M-estimation reweighting, providing resistance to outliers in both X and Y blocks [3].

The cellwise robust version (`crtb`) extends `rtb` with per-cell outlier weighting for both X and Y blocks, using SPADIMO to identify contaminated cells within flagged observations [4].

The diagnostic tool `spadimo` (SPArse DIrections of Maximal Outlyingness) identifies which variables contribute most to making an observation an outlier [5].

The `crm` (Cellwise Robust M-regression) method detects and handles cellwise outliers - individual contaminated cells in the data matrix rather than entire rows [6].

Optional `plotly`-based plot builders in `twoblock.plots` provide ready-made diagnostic figures (scree, scores, loadings, coefficients, predicted-vs-observed, case-weight histograms, cellwise-weight heatmaps, SPADIMO contributions).

## Installation

```bash
pip install twoblock
```

Or install from source:

```bash
git clone https://github.com/SvenSerneels/twoblock.git
cd twoblock
pip install -e .
```

### Dependencies

- numpy >= 1.22.0
- scikit-learn >= 1.3.0
- pandas >= 1.4.0
- scipy >= 1.8.0

## Usage

### twoblock — Dense and sparse two-block dimension reduction

```python
from twoblock import twoblock
from sklearn.model_selection import GridSearchCV

# Dense twoblock
tb = twoblock(n_components_x=5, n_components_y=2, scale='std')
tb.fit(X_train, Y_train)
Y_pred = tb.predict(X_test)

# Sparse twoblock (variable selection via soft-thresholding)
tb_sparse = twoblock(n_components_x=5, n_components_y=2,
                     sparse=True, eta_x=0.7, eta_y=0, scale='std')
tb_sparse.fit(X_train, Y_train)
Y_pred = tb_sparse.predict(X_test)

# Cross-validation with scikit-learn
gcv = GridSearchCV(twoblock(),
                   {'n_components_x': range(1, 10),
                    'n_components_y': range(1, 3),
                    'scale': ['std', 'None']},
                   scoring='r2', cv=5)
gcv.fit(X_train, Y_train)
```

### rtb — Robust twoblock with iterative reweighting

```python
from twoblock import rtb

# Dense robust twoblock (Hampel weighting, robust centering/scaling)
r = rtb(n_components_x=5, n_components_y=2,
        centre='l1median', scale='mad',
        fun='Hampel', probp1=0.95, probp2=0.975, probp3=0.999)
r.fit(X_train, Y_train)
Y_pred = r.predict(X_test)

# Sparse robust twoblock
r_sparse = rtb(n_components_x=5, n_components_y=2,
               sparse=True, eta_x=0.5, eta_y=0,
               centre='l1median', scale='mad',
               fun='Hampel', probp1=0.95, probp2=0.975, probp3=0.999)
r_sparse.fit(X_train, Y_train)

# Inspect case weights (outliers receive low weights)
print(r_sparse.caseweights_)

# Cross-validation
gcv = GridSearchCV(rtb(verbose=False),
                   {'n_components_x': range(1, 10),
                    'n_components_y': [1, 2],
                    'scale': ['mad', 'kstepLTS'],
                    'probp1': [0.75, 0.95]},
                   scoring='r2', cv=5)
gcv.fit(X_train, Y_train)
```

### crtb — Cellwise Robust Twoblock

CRTB extends RTB with per-cell outlier weighting. In each M-estimation iteration, SPADIMO identifies which variables drive outlyingness for flagged observations, and those individual cells are downweighted while the row continues to contribute through its case weight. An optional DDC-based pre-treatment provides cellwise-robust starting values, pushing resistance beyond the 50 % row-contamination breakdown of row-wise methods.

```python
from twoblock import crtb
import numpy as np

# Default: fast column-wise MAD pre-filter for starting values
c = crtb(n_components_x=5, n_components_y=2,
         centre='l1median', scale='scaleTau2',
         fun='Hampel', probp1=0.95, probp2=0.975, probp3=0.999,
         start_cellwise='prefilter')
c.fit(X_train, Y_train)
Y_pred = c.predict(X_test)

# DDC-based cellwise starting values (requires robpy)
c_ddc = crtb(n_components_x=5, n_components_y=2,
             centre='l1median', scale='scaleTau2',
             start_cellwise='DDC', crit_cellwise=0.99)
c_ddc.fit(X_train, Y_train)

# Inspect row- and cell-level diagnostics
print(f"Row case weights: {c.caseweights_}")
print(f"X cellwise outliers: {np.sum(c.x_cellwise_outliers_)}")
print(f"Y cellwise outliers: {np.sum(c.y_cellwise_outliers_)}")

# Sparse CRTB — variable selection + cellwise robustness
c_sparse = crtb(n_components_x=5, n_components_y=2,
                sparse=True, eta_x=0.5, eta_y=0,
                centre='l1median', scale='scaleTau2')
c_sparse.fit(X_train, Y_train)

# Impute outlying cells from the fitted model
X_imputed, Y_imputed = c.impute(X_train, Y_train)
```

Key parameters:
- `start_cellwise`: Cellwise starting-value strategy (`'prefilter'`, `'DDC'`, or `False`)
- `crit_cellwise`: Chi-squared quantile used to flag cells and observations for SPADIMO (default 0.99)
- `spadieta`: Sparsity sequence passed to SPADIMO (default `np.arange(0.9, 0.05, -0.1)`)
- Inherits `fun`, `probp1/2/3`, `centre`, `scale`, `sparse`, `eta_x/y` from `rtb`

Key attributes (in addition to those from `rtb`):
- `x_cellwise_outliers_`, `y_cellwise_outliers_`: Boolean cell-outlier maps
- `x_cellweights_`, `y_cellweights_`: Cellwise weights (0 = flagged, 1 = clean)
- `ddc_x_outliers_`, `ddc_y_outliers_`: Cellwise outliers from DDC initialisation (if `start_cellwise='DDC'`)

### spadimo — Sparse directions of maximal outlyingness

SPADIMO identifies which variables contribute most to making an observation an outlier. Given case weights from a robust estimator (e.g., `rtb`), it computes a sparse direction of maximal outlyingness and flags the contributing variables.

```python
from twoblock import rtb, spadimo

# First, fit a robust model to get case weights
r = rtb(n_components_x=5, n_components_y=2,
        centre='l1median', scale='mad', fun='Hampel')
r.fit(X, Y)

# Find observations with low weights (potential outliers)
outlier_indices = np.where(r.caseweights_ < 0.5)[0]

# Analyze an outlier to find contributing variables
sp = spadimo(scale='Qn', stop_early=True)
sp.fit(X, r.caseweights_, obs=outlier_indices[0])

# Get the flagged variables
print(f"Outlying variables: {sp.outlvars_}")
print(f"Outlyingness before: {sp.outlyingness_before_:.2f}")
print(f"Outlyingness after removing flagged vars: {sp.outlyingness_after_:.2f}")

# With a DataFrame, get variable names directly
sp.fit(X_df, r.caseweights_, obs=outlier_indices[0])
print(f"Outlying variable names: {sp.get_outlying_variables(names=True)}")

# Print a summary
sp.summary()
```

Key parameters:
- `scale`: Robust scale estimator ('Qn', 'mad', 'scaleTau2')
- `etas`: Sparsity parameters (default: sequence from 0.9 to 0.1)
- `stop_early`: Stop at first eta where observation becomes non-outlying
- `csq_critv`: Chi-squared quantile for outlyingness threshold (default: 0.975)

### crm — Cellwise Robust M-regression

CRM detects and handles cellwise outliers - individual contaminated cells rather than entire rows. It provides regression coefficients robust against both vertical outliers and leverage points, a map of cellwise outliers, and an imputed dataset with outlying cells replaced.

```python
from twoblock import crm
import numpy as np

# Fit CRM model with casewise robust starting values (default)
model = crm(center='median', scale='Qn', fun='Hampel')
model.fit(X, y)

# Fit CRM with cellwise robust starting values via DDC
# (requires robpy: pip install robpy)
model_ddc = crm(start_cellwise=True, center='median', scale='Qn')
model_ddc.fit(X, y)

# Predictions
y_pred = model.predict(X_new)

# View cellwise outlier map
print(f"Cellwise outliers detected: {np.sum(model.cellwise_outliers_)}")
print(f"Casewise outliers: {model.get_casewise_outliers()}")

# Get imputed data (outlying cells replaced)
X_imputed = model.X_imputed_

# Inspect which cells are outliers for a specific row
row_outliers = model.get_cellwise_outliers(row=0)
print(f"Outlying variables in row 0: {row_outliers}")

# With a DataFrame, get variable names
model.fit(X_df, y)
print(model.get_cellwise_outliers(row=0, names=True))

# Print summary
model.summary()
```

Key parameters:
- `start_cellwise`: If True, use DDC for cellwise robust starting values (default: False, requires robpy)
- `center`: Centering method ('median', 'mean', 'l1median')
- `scale`: Scale estimator ('Qn', 'mad', 'scaleTau2')
- `regtype`: Initial regression type ('MM', 'LTS')
- `fun`: M-estimation psi-function ('Hampel', 'Huber', 'Fair')
- `crit_cellwise`: Chi-squared quantile for cellwise outlier detection (default: 0.99)
- `maxiter`: Maximum IRLS iterations (default: 100)
- `tolerance`: Convergence threshold (default: 0.01)

Key attributes:
- `coef_`: Regression coefficients
- `cellwise_outliers_`: Boolean matrix of cell outliers (n, p)
- `casewise_outliers_`: Boolean array of row outliers (n,)
- `X_imputed_`: Imputed X matrix with outliers replaced
- `caseweights_`: Case weights from M-estimation
- `ddc_outliers_`: Cellwise outliers from DDC initialization (if `start_cellwise=True`)

### plots — Plotly diagnostic builders

`twoblock.plots` provides a small set of `plotly`-based builders that accept plain numpy arrays (e.g. `est.x_scores_`, `est.coef_`, `est.caseweights_`) and return a `plotly.graph_objects.Figure`. Because the API is array-first, the same builders work with any fitted twoblock estimator and with sklearn's `PLSRegression`.

Install the optional dependency:

```bash
pip install "twoblock[plots]"
```

```python
from twoblock import crtb
from twoblock.plots import (
    scree, score_scatter, loadings_bar, coefficients_bar,
    y_pred_vs_obs, caseweight_hist, cellweight_heatmap,
    spadimo_contributions,
)

c = crtb(n_components_x=3, n_components_y=1).fit(X, Y)

# Latent-space diagnostics
score_scatter(c.x_scores_, comp_x=0, comp_y=1,
              case_weights=c.caseweights_).show()
loadings_bar(c.x_loadings_, component=0,
             feature_names=X.columns).show()

# Regression diagnostics
coefficients_bar(c.coef_, feature_names=X.columns).show()
y_pred_vs_obs(Y, c.predict(X)).show()

# Outlier diagnostics
caseweight_hist(c.caseweights_).show()
cellweight_heatmap(c.x_cellweights_, feature_names=X.columns).show()

# SPADIMO contributions for a flagged observation
import numpy as np
from twoblock import spadimo
sp = spadimo(scale='scaleTau2', stop_early=True).fit(
    X, c.caseweights_, obs=int(np.argmin(c.caseweights_)))
spadimo_contributions(sp.contributions_,
                      feature_names=X.columns,
                      flagged_indices=sp.outlvars_).show()
```

## Examples

Example notebooks are provided in the [`examples/`](examples/) folder:
- `cookie_example.ipynb` — Cookie dough NIR spectroscopy
- `gas_turbine_example.ipynb` — Gas turbine CO/NOx emissions
- `simulation_rtb.ipynb` — Simulation study comparing twoblock, sparse twoblock, rtb, and sparse rtb
- `crm_simulation.ipynb` — CRM simulation under cellwise contamination with normal and Cauchy noise
- `cookie_example_crtb.ipynb`, `gas_turbine_example_crtb.ipynb`, `voc_example_crtb.ipynb` — CRTB applied to real datasets
- `simulation_crtb.py` — CRTB simulation study under cellwise contamination

## References

[1] R.D. Cook, L. Forzani, L. Liu.
    ["Partial least squares for simultaneous reduction of response and predictor
    vectors in regression."](https://doi.org/10.1016/j.jmva.2023.105163) Journal
    of Multivariate Analysis 196 (2023): 105163.

[2] S. Serneels. ["Sparse Twoblock Dimension Reduction: A Versatile Alternative
    to Sparse PLS2 and CCA."](https://doi.org/10.1002/cem.70051) Journal of
    Chemometrics, 39 (2025): e70051.

[3] S. Serneels. ["Robust Twoblock Dimension Reduction."] (https://arxiv.org/pdf/2603.24820) 
(2025, submitted). Preprint available at arXiv.org,  arXiv: 2603.24820.

[4] S. Serneels. "Cellwise Robust Twoblock Dimension Reduction." (2026, in
    preparation).

[5] M. Debruyne, S. Höppner, S. Serneels, T. Verdonck. ["Outlyingness: which
    variables contribute most?"](https://link.springer.com/article/10.1007/s11222-018-9831-5)
    Statistics and Computing 29 (4), 707-723.

[6] P. Filzmoser, S. Höppner, I. Ortner, S. Serneels, T. Verdonck. ["Cellwise Robust
    M regression."](https://doi.org/10.1016/j.csda.2020.106944) Computational
    Statistics & Data Analysis 147 (2020): 106944.