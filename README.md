# twoblock
Two-block dense and sparse simultaneous dimension reduction

The dense version is a `scikit-learn` compatible implementation of simultaneous two-block dimension reduction, as proposed in [1].

The sparse version is a `scikit-learn` compatible implementation of sparse twoblock dimension reduction, recently published by the author [2].

The robust version (`rtb`) extends twoblock with iterative M-estimation reweighting, providing resistance to outliers in both X and Y blocks [3].

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

## Examples

Example notebooks are provided in the [`examples/`](examples/) folder:
- `cookie_example.ipynb` — Cookie dough NIR spectroscopy
- `gas_turbine_example.ipynb` — Gas turbine CO/NOx emissions
- `simulation_rtb.ipynb` — Simulation study comparing twoblock, sparse twoblock, rtb, and sparse rtb

## References

[1] Cook, R. Dennis, Liliana Forzani, and Lan Liu.
    ["Partial least squares for simultaneous reduction of response and predictor
    vectors in regression."](https://doi.org/10.1016/j.jmva.2023.105163) Journal
    of Multivariate Analysis 196 (2023): 105163.

[2] S. Serneels. ["Sparse Twoblock Dimension Reduction: A Versatile Alternative
    to Sparse PLS2 and CCA."](https://doi.org/10.1002/cem.70051) Journal of
    Chemometrics, 39 (2025): e70051.

[3] S. Serneels. ["Robust Twoblock Dimension Reduction."] (https://arxiv.org/pdf/2603.24820) 
(2025, submitted). Preprint available at arXiv.org,  arXiv: 2603.24820.