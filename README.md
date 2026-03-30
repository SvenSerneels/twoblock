# twoblock
Two-block dense and sparse simultaneous dimension reduction

The dense version is a `scikit-learn` compatible implementation of simultaneous two-block dimension reduction, as proposed in [1].

The sparse version is a `scikit-learn` compatible implementation of sparse twoblock dimension reduction, recently published by the author [2].

The robust version (`rtb`) extends twoblock with iterative M-estimation reweighting, providing resistance to outliers in both X and Y blocks [3].

The diagnostic tool `spadimo` (SPArse DIrections of Maximal Outlyingness) identifies which variables contribute most to making an observation an outlier [4].

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

## Examples

Example notebooks are provided in the [`examples/`](examples/) folder:
- `cookie_example.ipynb` — Cookie dough NIR spectroscopy
- `gas_turbine_example.ipynb` — Gas turbine CO/NOx emissions
- `simulation_rtb.ipynb` — Simulation study comparing twoblock, sparse twoblock, rtb, and sparse rtb

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

[4] M. Debruyne, S. Höppner, S. Serneels, T. Verdonck. ["Outlyingness: which 
    variables contribute most?"](https://link.springer.com/article/10.1007/s11222-018-9831-5) 
    Statistics and Computing 29 (4), 707-723.