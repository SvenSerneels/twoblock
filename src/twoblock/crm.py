# -*- coding: utf-8 -*-
"""
CRM - Cellwise Robust M-regression

Detects and handles cellwise outliers - individual contaminated cells in the
data matrix rather than entire rows. Provides regression coefficients robust
against vertical outliers and leverage points, a map of cellwise outliers
consistent with the linear model, and an imputed dataset with outlying cells
replaced by fitted values.

Reference:
    Filzmoser, P., Höppner, S., Ortner, I., Serneels, S., & Verdonck, T. (2020).
    "Cellwise Robust M regression." Computational Statistics & Data Analysis,
    147, 106944.

@author: Sven Serneels
"""

import numpy as np
import pandas as ps
from sklearn.base import (
    RegressorMixin,
    BaseEstimator,
    TransformerMixin,
)
from scipy.stats import chi2
import copy
import warnings

from .utils import _check_input, _predict_check_input, Fair, Huber, Hampel
from .spadimo import spadimo
from ._preproc_utilities import mad, scaleTau2, Qn, median, mean, l1median
from ._gpu_utils import get_array_module, to_xp, to_numpy


class crm(BaseEstimator, TransformerMixin, RegressorMixin):
    """
    CRM - Cellwise Robust M-regression

    Detects and handles cellwise outliers in regression by combining
    M-estimation with SPADIMO for cellwise outlier identification.

    Parameters
    ----------
    maxiter : int, default 100
        Maximum number of IRLS iterations.

    tolerance : float, default 0.01
        Convergence threshold for coefficient change.

    center : str, default 'median'
        Centering method. Options: 'median', 'mean', 'l1median'.

    scale : str, default 'Qn'
        Scale method. Options: 'Qn', 'mad', 'scaleTau2'.

    regtype : str, default 'MM'
        Regression type for initial estimate. Options: 'MM', 'LTS'.

    alphaLTS : float, default 0.5
        LTS trimming proportion (used if regtype='LTS').

    fun : str, default 'Hampel'
        M-estimation psi-function. Options: 'Hampel', 'Huber', 'Fair'.

    probp1 : float, default 0.95
        Probability cutoff for start of downweighting.

    probp2 : float, default 0.975
        Probability cutoff for start of steep downweighting (Hampel only).

    probp3 : float, default 0.999
        Probability cutoff for outlier omission (Hampel only).

    spadieta : array-like or None, default None
        SPADIMO sparsity parameters. If None, uses np.arange(0.9, 0.05, -0.1).

    crit_cellwise : float, default 0.99
        Chi-squared quantile for cellwise outlier detection.

    outlyingness_factor : float, default 1.0
        Multiplier for outlier threshold.

    start_cellwise : bool, default False
        If True, use cellwise robust starting values via DDC (DetectDeviatingCells)
        from robpy. DDC detects cellwise outliers and imputes them before
        computing initial estimates. Requires robpy package.
        If False (default), use casewise robust starting values (MM or LTS).

    gpu : bool, default False
        Enable GPU acceleration via CuPy.

    verbose : bool, default True
        Print progress during fitting.

    copy : bool, default True
        Copy input data.

    Attributes (post-fit)
    ---------------------
    coef_ : ndarray of shape (p,) or (p, q)
        Regression coefficients.

    intercept_ : float or ndarray
        Intercept term.

    fitted_ : ndarray of shape (n,) or (n, q)
        Fitted values.

    residuals_ : ndarray of shape (n,) or (n, q)
        Residuals.

    caseweights_ : ndarray of shape (n,)
        Case weights from M-estimation.

    cellweights_ : ndarray of shape (n, p)
        Cellwise weights matrix.

    casewise_outliers_ : ndarray of shape (n,)
        Boolean array indicating row outliers.

    cellwise_outliers_ : ndarray of shape (n, p)
        Boolean matrix indicating cell outliers.

    X_imputed_ : ndarray of shape (n, p)
        Imputed X matrix with outliers replaced.

    n_iter_ : int
        Number of iterations performed.

    x_center_ : ndarray of shape (p,)
        Center values used for X.

    x_scale_ : ndarray of shape (p,)
        Scale values used for X.

    y_center_ : float or ndarray
        Center value(s) for y.

    y_scale_ : float or ndarray
        Scale value(s) for y.

    ddc_outliers_ : ndarray of shape (n, p) or None
        Boolean matrix of cellwise outliers detected by DDC at initialization.
        Only set if start_cellwise=True.
    """

    def __init__(
        self,
        maxiter=100,
        tolerance=0.01,
        center='median',
        scale='Qn',
        regtype='MM',
        alphaLTS=0.5,
        fun='Hampel',
        probp1=0.95,
        probp2=0.975,
        probp3=0.999,
        spadieta=None,
        crit_cellwise=0.99,
        outlyingness_factor=1.0,
        start_cellwise=False,
        gpu=False,
        verbose=True,
        copy=True,
    ):
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.center = center
        self.scale = scale
        self.regtype = regtype
        self.alphaLTS = alphaLTS
        self.fun = fun
        self.probp1 = probp1
        self.probp2 = probp2
        self.probp3 = probp3
        self.spadieta = spadieta
        self.crit_cellwise = crit_cellwise
        self.outlyingness_factor = outlyingness_factor
        self.start_cellwise = start_cellwise
        self.gpu = gpu
        self.verbose = verbose
        self.copy = copy

    def fit(self, X, y):
        """
        Fit the CRM model.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Predictor matrix.

        y : array-like of shape (n,) or (n, q)
            Response variable(s).

        Returns
        -------
        self : crm
            Fitted estimator.
        """
        # Get array module for GPU support
        xp, xp_linalg = get_array_module(self.gpu)

        # Input validation and conversion
        if type(X) == ps.core.frame.DataFrame:
            self.colnames_ = list(X.columns)
            X = X.to_numpy().astype('float64')
        else:
            self.colnames_ = None

        X = _check_input(X)
        n, p = X.shape

        if type(y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            y = y.to_numpy().astype('float64')
        y = np.atleast_1d(np.asarray(y).ravel()).astype('float64')

        if len(y) != n:
            raise ValueError(f"X has {n} samples but y has {len(y)} samples")

        if self.copy:
            X = X.copy()
            y = y.copy()

        # Store original X and y for use at the end of fit
        X_original = X.copy()
        y_original = y.copy()

        # Validate parameters
        if self.fun not in ('Hampel', 'Huber', 'Fair'):
            raise ValueError(
                "Invalid weighting function. Choose Hampel, Huber or Fair."
            )
        if self.fun == 'Hampel':
            if not (self.probp1 < self.probp2 < self.probp3 <= 1):
                raise ValueError(
                    "Wrong choice of parameters for Hampel function. "
                    "Use 0 < probp1 < probp2 < probp3 <= 1"
                )

        # Set default SPADIMO etas
        if self.spadieta is None:
            spadieta = np.arange(0.9, 0.05, -0.1)
        else:
            spadieta = np.asarray(self.spadieta)

        # Initialize cellwise weights to 1 (no downweighting)
        cellweights = np.ones((n, p))
        cellwise_outliers = np.zeros((n, p), dtype=bool)
        ddc_outliers = None

        # Cellwise robust starting values via DDC
        if self.start_cellwise:
            if self.verbose:
                print("Computing cellwise robust starting values via DDC...")
            try:
                from robpy.outliers.ddc import DDC

                # DDC expects a pandas DataFrame
                if self.colnames_ is not None:
                    X_df = ps.DataFrame(X, columns=self.colnames_)
                else:
                    X_df = ps.DataFrame(X, columns=[f'X{i}' for i in range(p)])

                # Run DDC on X to detect cellwise outliers
                ddc = DDC()
                ddc.fit(X_df)

                # Get DDC outlier flags and imputed data
                # DDC stores cellwise outliers as boolean ndarray
                ddc_outliers = ddc.cellwise_outliers_
                if hasattr(ddc_outliers, 'to_numpy'):
                    ddc_outliers = ddc_outliers.to_numpy()
                X_ddc_imputed = ddc.impute(X_df)
                if hasattr(X_ddc_imputed, 'to_numpy'):
                    X_ddc_imputed = X_ddc_imputed.to_numpy()

                # Initialize cellwise outliers and weights from DDC
                cellwise_outliers = ddc_outliers.copy()
                cellweights[ddc_outliers] = 0.0

                # Use DDC-imputed data for subsequent calculations
                X = X_ddc_imputed

                if self.verbose:
                    n_ddc_outliers = np.sum(ddc_outliers)
                    pct_ddc = 100 * n_ddc_outliers / (n * p)
                    print(f"DDC detected {n_ddc_outliers} cellwise outliers "
                          f"({pct_ddc:.2f}%)")

            except ImportError:
                warnings.warn(
                    "robpy package not available. Install with: pip install robpy. "
                    "Falling back to casewise robust starting values."
                )
                self.start_cellwise = False
            except Exception as e:
                warnings.warn(
                    f"DDC failed: {e}. "
                    "Falling back to casewise robust starting values."
                )
                self.start_cellwise = False

        # Robust centering and scaling of X (possibly DDC-imputed)
        x_center, x_scale = self._robust_center_scale(X)
        Xs = (X - x_center) / x_scale

        # Robust centering and scaling of y
        y_center, y_scale = self._robust_center_scale_1d(y)
        ys = (y - y_center) / y_scale

        # Initial estimate via MM or LTS
        if self.verbose:
            print("Computing initial estimate...")
        beta = self._initial_estimate(Xs, ys)

        # Compute initial case weights
        residuals = ys - Xs @ beta
        residual_scale = self._robust_scale_1d(residuals)
        if residual_scale < 1e-10:
            residual_scale = 1.0
        std_resid = residuals / residual_scale
        caseweights = self._compute_case_weights(np.abs(std_resid))

        # Chi-squared thresholds
        chi2_threshold = np.sqrt(chi2.ppf(self.crit_cellwise, 1))
        chi2_threshold *= self.outlyingness_factor

        # IRLS loop
        beta_old = beta.copy()
        converged = False

        for iteration in range(self.maxiter):
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.maxiter}")

            # Step 1: Identify cellwise outliers using SPADIMO
            residuals = ys - Xs @ beta
            residual_scale = self._robust_scale_1d(residuals)
            if residual_scale < 1e-10:
                residual_scale = 1.0
            std_resid = residuals / residual_scale

            # Find observations with large standardized residuals
            large_resid_mask = np.abs(std_resid) > chi2_threshold

            # For observations with large residuals, identify cellwise outliers
            for i in np.where(large_resid_mask)[0]:
                try:
                    sp = spadimo(
                        scale=self.scale,
                        etas=spadieta,
                        stop_early=True,
                        gpu=self.gpu,
                        copy=False,
                    )
                    sp.fit(Xs, caseweights, obs=i)

                    if len(sp.outlvars_) > 0:
                        cellwise_outliers[i, sp.outlvars_] = True
                        # Downweight outlying cells
                        cellweights[i, sp.outlvars_] = 0.0
                except Exception:
                    # If SPADIMO fails, skip this observation
                    pass

            # Step 2: Weighted least squares with cellwise weights
            # Combine case weights and cellwise weights
            combined_weights = caseweights[:, np.newaxis] * cellweights

            # Build weighted design matrix
            W = np.sqrt(combined_weights)
            Xw = Xs * W
            yw = ys * np.sqrt(caseweights)

            # Solve weighted least squares
            try:
                beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                beta = np.linalg.pinv(Xw) @ yw

            # Step 3: Update case weights based on new residuals
            residuals = ys - Xs @ beta
            residual_scale = self._robust_scale_1d(residuals)
            if residual_scale < 1e-10:
                residual_scale = 1.0
            std_resid = residuals / residual_scale
            caseweights = self._compute_case_weights(np.abs(std_resid))

            # Floor case weights
            caseweights = np.maximum(caseweights, 1e-6)

            # Step 4: Check convergence
            beta_diff = np.sqrt(np.sum((beta - beta_old) ** 2))
            beta_norm = np.sqrt(np.sum(beta_old ** 2))
            if beta_norm < 1e-10:
                beta_norm = 1.0

            if beta_diff / beta_norm < self.tolerance:
                converged = True
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            beta_old = beta.copy()

        if not converged and self.verbose:
            warnings.warn(
                f"CRM did not converge after {self.maxiter} iterations. "
                f"Final relative change: {beta_diff / beta_norm:.6f}"
            )

        # Compute final predictions and residuals on original scale
        # Rescale coefficients: beta_orig = (y_scale / x_scale) * beta
        coef = (y_scale / x_scale) * beta
        intercept = y_center - np.sum(coef * x_center)

        # Use original X and y for final predictions and residuals
        fitted = X_original @ coef + intercept
        residuals = y_original - fitted

        # Identify casewise outliers (those with very low case weights)
        casewise_outliers = caseweights < 0.1

        # Impute outlying cells using original X and y
        X_imputed = self._impute_cells(
            X_original, y_original, coef, intercept, cellwise_outliers
        )

        # Store fitted attributes
        setattr(self, 'coef_', coef)
        setattr(self, 'intercept_', intercept)
        setattr(self, 'fitted_', fitted)
        setattr(self, 'residuals_', residuals)
        setattr(self, 'caseweights_', caseweights)
        setattr(self, 'cellweights_', cellweights)
        setattr(self, 'casewise_outliers_', casewise_outliers)
        setattr(self, 'cellwise_outliers_', cellwise_outliers)
        setattr(self, 'X_imputed_', X_imputed)
        setattr(self, 'n_iter_', iteration + 1)
        setattr(self, 'x_center_', x_center)
        setattr(self, 'x_scale_', x_scale)
        setattr(self, 'y_center_', y_center)
        setattr(self, 'y_scale_', y_scale)
        setattr(self, 'ddc_outliers_', ddc_outliers)
        setattr(self, 'X_', X_original)
        setattr(self, 'y_', y_original)

        return self

    def predict(self, Xn):
        """
        Predict using the fitted CRM model.

        Parameters
        ----------
        Xn : array-like of shape (n_samples, p)
            New predictor data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        n, p, Xn = _predict_check_input(Xn)
        if p != len(self.coef_):
            raise ValueError(
                f"New data must have {len(self.coef_)} columns, got {p}"
            )
        return Xn @ self.coef_ + self.intercept_

    def transform(self, Xn):
        """
        Return imputed X (for new data, identify and replace outlying cells).

        For training data, returns the already computed X_imputed_.
        For new data, detects cellwise outliers and imputes them.

        Parameters
        ----------
        Xn : array-like of shape (n_samples, p)
            Input data.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, p)
            Imputed data with outlying cells replaced.
        """
        n, p, Xn = _predict_check_input(Xn)
        if p != len(self.coef_):
            raise ValueError(
                f"New data must have {len(self.coef_)} columns, got {p}"
            )

        # For new data, detect outlying cells and impute
        # Standardize using training center/scale
        Xn_s = (Xn - self.x_center_) / self.x_scale_

        # Compute residuals
        y_pred = self.predict(Xn)

        # For new data, we don't have y, so we use predicted values
        # and identify cells that are unusual given the model
        cellwise_outliers = np.zeros((n, p), dtype=bool)

        # Identify cells that deviate significantly from expected
        for i in range(n):
            # Compute influence of each cell on prediction
            for j in range(p):
                # If cell is unusually large/small, flag it
                expected_contrib = Xn_s[i, j] * self.coef_[j]
                cell_std = np.abs(Xn_s[i, j])
                threshold = chi2.ppf(self.crit_cellwise, 1)
                if cell_std > np.sqrt(threshold):
                    cellwise_outliers[i, j] = True

        # Impute flagged cells
        X_imputed = Xn.copy()
        for i in range(n):
            outlying_cols = np.where(cellwise_outliers[i, :])[0]
            if len(outlying_cols) > 0:
                # Replace with values that would give predicted y
                # This is a simplified approach
                clean_cols = np.where(~cellwise_outliers[i, :])[0]
                if len(clean_cols) > 0:
                    # Predict contribution from clean cells
                    clean_contrib = np.sum(X_imputed[i, clean_cols] * self.coef_[clean_cols])
                    # Distribute remaining prediction to outlying cells
                    remaining = y_pred[i] - self.intercept_ - clean_contrib
                    if len(outlying_cols) > 0:
                        coef_sum = np.sum(np.abs(self.coef_[outlying_cols]))
                        if coef_sum > 1e-10:
                            for j in outlying_cols:
                                X_imputed[i, j] = remaining * self.coef_[j] / coef_sum / self.coef_[j]

        return X_imputed

    def _robust_center_scale(self, X):
        """
        Compute robust center and scale for X.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Data matrix.

        Returns
        -------
        center : ndarray of shape (p,)
            Center estimates.
        scale : ndarray of shape (p,)
            Scale estimates.
        """
        # Center
        if self.center == 'median':
            center = median(X)
        elif self.center == 'mean':
            center = mean(X)
        elif self.center == 'l1median':
            center = l1median(X)
        else:
            raise ValueError(f"Unknown center method: {self.center}")

        # Scale
        if self.scale == 'Qn':
            scale = Qn(X)
        elif self.scale == 'mad':
            scale = mad(X)
        elif self.scale == 'scaleTau2':
            scale = scaleTau2(X)
        else:
            raise ValueError(f"Unknown scale method: {self.scale}")

        # Handle zero scales
        scale = np.where(scale < 1e-10, 1.0, scale)

        return center, scale

    def _robust_center_scale_1d(self, y):
        """
        Compute robust center and scale for 1D array.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Data vector.

        Returns
        -------
        center : float
            Center estimate.
        scale : float
            Scale estimate.
        """
        y = y.reshape(-1, 1)

        if self.center == 'median':
            center = median(y)[0]
        elif self.center == 'mean':
            center = mean(y)[0]
        elif self.center == 'l1median':
            center = median(y)[0]  # l1median reduces to median for 1D
        else:
            raise ValueError(f"Unknown center method: {self.center}")

        if self.scale == 'Qn':
            scale = Qn(y)[0]
        elif self.scale == 'mad':
            scale = mad(y)[0]
        elif self.scale == 'scaleTau2':
            scale = scaleTau2(y)[0]
        else:
            raise ValueError(f"Unknown scale method: {self.scale}")

        if scale < 1e-10:
            scale = 1.0

        return center, scale

    def _robust_scale_1d(self, y):
        """
        Compute robust scale for 1D array.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Data vector.

        Returns
        -------
        scale : float
            Scale estimate.
        """
        y = y.reshape(-1, 1)

        if self.scale == 'Qn':
            return Qn(y)[0]
        elif self.scale == 'mad':
            return mad(y)[0]
        elif self.scale == 'scaleTau2':
            return scaleTau2(y)[0]
        else:
            return mad(y)[0]

    def _initial_estimate(self, X, y):
        """
        Compute initial regression estimate using MM or LTS.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Standardized predictor matrix.
        y : ndarray of shape (n,)
            Standardized response.

        Returns
        -------
        beta : ndarray of shape (p,)
            Initial coefficient estimates.
        """
        n, p = X.shape

        if self.regtype == 'LTS':
            # Least Trimmed Squares
            beta = self._lts_estimate(X, y)
        else:
            # MM-estimation (default)
            # Use LTS as starting point, then refine with M-estimation
            beta = self._mm_estimate(X, y)

        return beta

    def _lts_estimate(self, X, y):
        """
        Compute Least Trimmed Squares estimate.

        Uses concentration steps (C-steps) for efficiency.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Predictor matrix.
        y : ndarray of shape (n,)
            Response.

        Returns
        -------
        beta : ndarray of shape (p,)
            LTS coefficient estimates.
        """
        n, p = X.shape
        h = int(np.floor(n * self.alphaLTS)) + int(np.floor((p + 1) / 2))
        h = min(h, n)

        # Start with OLS estimate
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X) @ y

        # C-steps
        for _ in range(10):
            residuals = y - X @ beta
            squared_resid = residuals ** 2

            # Select h observations with smallest squared residuals
            h_subset = np.argsort(squared_resid)[:h]

            # Recompute estimate on subset
            Xh = X[h_subset, :]
            yh = y[h_subset]
            try:
                beta_new, _, _, _ = np.linalg.lstsq(Xh, yh, rcond=None)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(Xh) @ yh

            # Check convergence
            if np.allclose(beta, beta_new, rtol=1e-6):
                break
            beta = beta_new

        return beta

    def _mm_estimate(self, X, y):
        """
        Compute MM-estimate.

        Uses LTS as starting point, then applies M-estimation.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Predictor matrix.
        y : ndarray of shape (n,)
            Response.

        Returns
        -------
        beta : ndarray of shape (p,)
            MM coefficient estimates.
        """
        n, p = X.shape

        # Start with LTS
        beta = self._lts_estimate(X, y)

        # Refine with M-estimation iterations
        for _ in range(20):
            residuals = y - X @ beta
            residual_scale = self._robust_scale_1d(residuals)
            if residual_scale < 1e-10:
                residual_scale = 1.0

            std_resid = residuals / residual_scale
            weights = self._compute_case_weights(np.abs(std_resid))
            weights = np.maximum(weights, 1e-6)

            # Weighted least squares
            W = np.sqrt(weights)
            Xw = X * W[:, np.newaxis]
            yw = y * W

            try:
                beta_new, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.pinv(Xw) @ yw

            # Check convergence
            if np.allclose(beta, beta_new, rtol=1e-6):
                break
            beta = beta_new

        return beta

    def _compute_case_weights(self, abs_std_resid):
        """
        Compute case weights using the selected psi-function.

        Parameters
        ----------
        abs_std_resid : ndarray of shape (n,)
            Absolute standardized residuals.

        Returns
        -------
        weights : ndarray of shape (n,)
            Case weights in [0, 1].
        """
        # Get cutoffs
        probct = chi2.ppf(self.probp1, 1)

        if self.fun == 'Fair':
            weights = Fair(abs_std_resid.copy(), probct)
        elif self.fun == 'Huber':
            weights = Huber(abs_std_resid.copy(), probct)
        elif self.fun == 'Hampel':
            hampelb = chi2.ppf(self.probp2, 1)
            hampelr = chi2.ppf(self.probp3, 1)
            weights = Hampel(abs_std_resid.copy(), probct, hampelb, hampelr)
        else:
            weights = np.ones_like(abs_std_resid)

        return weights

    def _impute_cells(self, X, y, coef, intercept, cellwise_outliers):
        """
        Impute outlying cells based on the regression model.

        For each observation with outlying cells, replace those cells
        with values that are consistent with the linear model and
        the non-outlying cells.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Original predictor matrix.
        y : ndarray of shape (n,)
            Response variable.
        coef : ndarray of shape (p,)
            Regression coefficients.
        intercept : float
            Intercept.
        cellwise_outliers : ndarray of shape (n, p)
            Boolean matrix of outlying cells.

        Returns
        -------
        X_imputed : ndarray of shape (n, p)
            Imputed predictor matrix.
        """
        n, p = X.shape
        X_imputed = X.copy()

        for i in range(n):
            outlying_cols = np.where(cellwise_outliers[i, :])[0]
            if len(outlying_cols) == 0:
                continue

            clean_cols = np.where(~cellwise_outliers[i, :])[0]

            if len(clean_cols) == 0:
                # All cells are outlying, use median replacement
                for j in outlying_cols:
                    X_imputed[i, j] = self.x_center_[j]
                continue

            # Compute prediction from clean cells
            clean_contrib = np.sum(X[i, clean_cols] * coef[clean_cols])

            # Remaining contribution needed to match y
            remaining = y[i] - intercept - clean_contrib

            # Distribute remaining contribution proportionally to coefficients
            coef_outlying = coef[outlying_cols]
            coef_sum = np.sum(np.abs(coef_outlying))

            if coef_sum < 1e-10:
                # Coefficients near zero, use median
                for j in outlying_cols:
                    X_imputed[i, j] = self.x_center_[j]
            else:
                # Distribute proportionally
                for j in outlying_cols:
                    if np.abs(coef[j]) > 1e-10:
                        contrib_share = remaining * np.abs(coef[j]) / coef_sum
                        X_imputed[i, j] = contrib_share / coef[j]
                    else:
                        X_imputed[i, j] = self.x_center_[j]

        return X_imputed

    def summary(self):
        """
        Print a summary of the CRM fit.
        """
        if not hasattr(self, 'coef_'):
            raise ValueError("Model not fitted. Call fit() first.")

        p = len(self.coef_)
        n = len(self.y_)

        n_cellwise = np.sum(self.cellwise_outliers_)
        n_casewise = np.sum(self.casewise_outliers_)
        pct_cellwise = 100 * n_cellwise / (n * p)
        pct_casewise = 100 * n_casewise / n

        print("CRM - Cellwise Robust M-regression Summary")
        print("=" * 55)
        print(f"Number of observations: {n}")
        print(f"Number of predictors: {p}")
        print(f"Iterations: {self.n_iter_}")
        print(f"Regression type: {self.regtype}")
        print(f"M-estimation function: {self.fun}")
        print("-" * 55)
        print(f"Cellwise outliers: {n_cellwise} ({pct_cellwise:.2f}%)")
        print(f"Casewise outliers: {n_casewise} ({pct_casewise:.2f}%)")
        print("-" * 55)
        print("Coefficients:")

        if self.colnames_ is not None:
            for j, name in enumerate(self.colnames_):
                print(f"  {name}: {self.coef_[j]:.6f}")
        else:
            for j in range(min(p, 10)):
                print(f"  X[{j}]: {self.coef_[j]:.6f}")
            if p > 10:
                print(f"  ... ({p - 10} more)")

        print(f"Intercept: {self.intercept_:.6f}")

    def get_cellwise_outliers(self, row=None, names=False):
        """
        Get cellwise outlier information.

        Parameters
        ----------
        row : int or None, default None
            If specified, return outliers for that row only.
        names : bool, default False
            If True and column names available, return names.

        Returns
        -------
        outliers : ndarray or list
            Indices or names of outlying cells.
        """
        if not hasattr(self, 'cellwise_outliers_'):
            raise ValueError("Model not fitted. Call fit() first.")

        if row is not None:
            outlier_cols = np.where(self.cellwise_outliers_[row, :])[0]
            if names and self.colnames_ is not None:
                return [self.colnames_[j] for j in outlier_cols]
            return outlier_cols

        return self.cellwise_outliers_

    def get_casewise_outliers(self):
        """
        Get indices of casewise outliers.

        Returns
        -------
        outliers : ndarray
            Indices of casewise outlying observations.
        """
        if not hasattr(self, 'casewise_outliers_'):
            raise ValueError("Model not fitted. Call fit() first.")

        return np.where(self.casewise_outliers_)[0]
