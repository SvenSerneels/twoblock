# -*- coding: utf-8 -*-
"""
Cellwise Robust Twoblock (CRTB)

Extends RTB with per-cell outlier weighting for both the X and Y blocks.
SPADIMO identifies which variables drive outlyingness for each flagged
observation, and the corresponding cells are downweighted. Optionally uses
DDC-based cellwise imputation for starting values, which makes is resistant to
more than fifty percent of row containing contaminated cells.

@author: Sven Serneels
"""

import numpy as np
from scipy.stats import chi2
import copy
import pandas as ps
import warnings

from .utils import _check_input, _predict_check_input, Fair, Huber, Hampel
from .prepro import VersatileScaler
from .twoblock import twoblock
from .spadimo import spadimo
from ._preproc_utilities import mad
from .rtb import rtb


class crtb(rtb):
    """
    CRTB - Cellwise Robust Twoblock

    Extends RTB with per-cell (cellwise) outlier weighting for both the X
    and Y blocks. In each M-estimation iteration, SPADIMO identifies which
    variables drive outlyingness for observations flagged as outlying in the
    X or Y score space. Those cells are set to weight zero, so only their
    row-level case weight continues to suppress them in subsequent iterations.

    Optionally uses DDC-based cellwise imputation (from robpy) for the
    starting values, analogous to the start_cellwise option in CRM.

    Parameters
    ----------
    n_components_x : int
        Number of X-block components.

    n_components_y : int or None
        Number of Y-block components. Defaults to n_components_x.

    sparse : bool, default False
        Use sparse (soft-thresholded) twoblock.

    eta_x : float, default 0.5
        X-block sparsity parameter (used when sparse=True).

    eta_y : float, default 0.5
        Y-block sparsity parameter (used when sparse=True).

    fun : str, default 'Hampel'
        M-estimation downweighting function: 'Hampel', 'Huber', or 'Fair'.

    probp1 : float, default 0.95
        Probability cutoff for start of downweighting.

    probp2 : float, default 0.975
        Probability cutoff for start of steep downweighting (Hampel only).

    probp3 : float, default 0.999
        Probability cutoff for outlier omission (Hampel only).

    centre : str, default 'median'
        Centering: 'mean', 'median', 'l1median', or 'kstepLTS'.

    scale : str, default 'mad'
        Scaling: 'std', 'mad', 'scaleTau2', or 'None'.

    verbose : bool, default True
        Print progress messages.

    maxit : int, default 100
        Maximum number of M-estimation iterations.

    tol : float, default 0.01
        Convergence tolerance on relative change in coefficient norm.

    start_cutoff_mode : str, default 'specific'
        'specific' sets starting cutoffs from chi2(relcomp); any other value
        uses normal quantiles.

    start_X_init : str, default 'pcapp'
        'pcapp' uses brokenstick-trimmed PCA for starting weights; any other
        value uses the raw scaled block.

    copy : bool, default True
        Copy input data into the object.

    gpu : bool, default False
        Enable GPU acceleration via CuPy.

    start_cellwise : bool, default False
        If True, run DDC (from robpy) on X and Y before computing starting
        case weights. DDC imputes cellwise outliers so that robust centering,
        scaling, and brokenstick PCA are not distorted by them. This is the
        recommended option when more than half the rows may contain
        contaminated cells. Requires the robpy package.

    spadieta : array-like or None, default None
        Sparsity parameter sequence passed to SPADIMO. If None, uses
        np.arange(0.9, 0.05, -0.1).

    crit_cellwise : float, default 0.99
        Chi-squared quantile used to flag observations for SPADIMO analysis.
        Observation i is sent to SPADIMO if its normalized score distance
        exceeds sqrt(chi2.ppf(crit_cellwise, n_components)).

    Attributes
    ----------
    x_weights_ : ndarray (p, n_components_x)
    y_weights_ : ndarray (q, n_components_y)
    x_loadings_ : ndarray (p, n_components_x)
    y_loadings_ : ndarray (q, n_components_y)
    x_scores_ : ndarray (n, n_components_x)
    y_scores_ : ndarray (n, n_components_y)
    coef_ : ndarray (p, q)
        Regression coefficients on the original scale.
    intercept_ : ndarray (q,)
    coef_scaled_ : ndarray
    intercept_scaled_ : ndarray
    residuals_ : ndarray (n, q)
    fitted_ : ndarray (n, q)
    x_loc_ : ndarray (p,)
    y_loc_ : ndarray (q,)
    x_sca_ : ndarray (p,)
    y_sca_ : ndarray (q,)
    x_caseweights_ : ndarray (n,)
    y_caseweights_ : ndarray (n,)
    caseweights_ : ndarray (n,)
        Combined X * Y case weights.
    x_cellweights_ : ndarray (n, p)
        Cellwise weight matrix for X block. Entry [i, j] is 0 if variable j
        of observation i was identified as a cellwise outlier by SPADIMO.
    y_cellweights_ : ndarray (n, q)
        Cellwise weight matrix for Y block.
    x_cellwise_outliers_ : ndarray of bool (n, p)
        Boolean map of X-block cellwise outliers detected across all iterations.
    y_cellwise_outliers_ : ndarray of bool (n, q)
        Boolean map of Y-block cellwise outliers.
    ddc_x_outliers_ : ndarray of bool (n, p) or None
        Cellwise outlier map from DDC for X (set only when start_cellwise=True
        and robpy is available).
    ddc_y_outliers_ : ndarray of bool (n, q) or None
        Cellwise outlier map from DDC for Y.
    centring_ : VersatileScaler
    scalingt_ : VersatileScaler
    scalingu_ : VersatileScaler
    """

    def __init__(
        self,
        n_components_x=1,
        n_components_y=None,
        sparse=False,
        eta_x=0.5,
        eta_y=0.5,
        fun="Hampel",
        probp1=0.95,
        probp2=0.975,
        probp3=0.999,
        centre="median",
        scale="mad",
        verbose=True,
        maxit=100,
        tol=0.01,
        start_cutoff_mode="specific",
        start_X_init="pcapp",
        copy=True,
        gpu=False,
        start_cellwise=False,
        spadieta=None,
        crit_cellwise=0.99,
    ):
        super().__init__(
            n_components_x=n_components_x,
            n_components_y=n_components_y,
            sparse=sparse,
            eta_x=eta_x,
            eta_y=eta_y,
            fun=fun,
            probp1=probp1,
            probp2=probp2,
            probp3=probp3,
            centre=centre,
            scale=scale,
            verbose=verbose,
            maxit=maxit,
            tol=tol,
            start_cutoff_mode=start_cutoff_mode,
            start_X_init=start_X_init,
            copy=copy,
            gpu=gpu,
        )
        self.start_cellwise = start_cellwise
        self.spadieta = spadieta
        self.crit_cellwise = crit_cellwise

    def fit(self, X, Y):
        """
        Fit a Cellwise Robust Twoblock model.

        Parameters
        ----------
        X : numpy array or pandas DataFrame of shape (n, p)
            Predictor block.
        Y : numpy array or pandas DataFrame of shape (n, q)
            Response block.

        Returns
        -------
        self : crtb
        """
        if type(X) == ps.core.frame.DataFrame:
            X = X.to_numpy()
        (n, p) = X.shape
        if type(Y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            Y = Y.to_numpy()
        X = _check_input(X)
        Y = _check_input(Y)
        ny, q = Y.shape
        if ny != n:
            raise ValueError("Number of cases in X and Y needs to agree")

        Y = Y.astype("float64")

        if self.n_components_y is None:
            self.n_components_y = self.n_components_x

        if self.copy:
            self.X = copy.deepcopy(X)
            self.Y = copy.deepcopy(Y)

        if not (self.fun in ("Hampel", "Huber", "Fair")):
            raise ValueError(
                "Invalid weighting function. Choose Hampel, Huber or Fair."
            )
        if (self.probp1 > 1) or (self.probp1 <= 0):
            raise ValueError(
                "probp1 is a probability. Choose a value between 0 and 1"
            )
        if self.fun == "Hampel":
            if not (
                (self.probp1 < self.probp2)
                and (self.probp2 < self.probp3)
                and (self.probp3 <= 1)
            ):
                raise ValueError(
                    "Wrong choice of parameters for Hampel function. "
                    "Use 0 < probp1 < probp2 < probp3 <= 1"
                )

        if self.scale == "scaleTau2":
            if (mad(X) == 0).any():
                self.scale = "mad"
                warnings.warn("Scale set to `mad`")

        spadieta = (
            np.arange(0.9, 0.05, -0.1)
            if self.spadieta is None
            else np.asarray(self.spadieta)
        )
        spadi_scale = self.scale if self.scale != "None" else "mad"

        # Distance thresholds for SPADIMO observation selection
        chi2_thr_x = np.sqrt(chi2.ppf(self.crit_cellwise, self.n_components_x))
        chi2_thr_y = np.sqrt(chi2.ppf(self.crit_cellwise, self.n_components_y))

        # --- Cellwise starting values via DDC (optional) ---
        ddc_x_outliers = None
        ddc_y_outliers = None
        X_orig = X.astype("float64")
        Y_orig = Y.astype("float64")
        X_fit = X_orig.copy()
        Y_fit = Y_orig.copy()

        if self.start_cellwise:
            if self.verbose:
                print("Computing cellwise starting values via DDC...")
            X_fit, ddc_x_outliers = self._ddc_impute(X_fit, "X")
            Y_fit, ddc_y_outliers = self._ddc_impute(Y_fit, "Y")

        # --- Robust centering and scaling ---
        # loc/sca are estimated on the original contaminated data: robust
        # estimators (scaleTau2, l1median) have a ~50% breakdown point so they
        # handle the 10-30% cell contamination expected in practice.  Using
        # DDC-imputed data for scaling can produce artificially small scale
        # estimates when DDC over-imputes a variable, which would amplify noise
        # in the final coefficient rescaling.
        scaling = VersatileScaler(center=self.centre, scale=self.scale)
        Xs = scaling.fit_transform(X_orig).astype("float64")
        mX = scaling.col_loc_
        sX = scaling.col_sca_
        Ys = scaling.fit_transform(Y_orig).astype("float64")
        my = scaling.col_loc_
        sy = scaling.col_sca_

        from ._preproc_utilities import scale_data

        # Starting case weights use DDC-imputed data (same loc/sca) so that
        # heavily contaminated rows do not corrupt the brokenstick PCA
        # initialisation of the M-estimation loop.
        Xs_init = scale_data(X_fit, mX, sX).astype("float64")
        Ys_init = scale_data(Y_fit, my, sy).astype("float64")

        # --- Starting case weights ---
        we = self._compute_starting_weights(Xs_init, scaling, n, p)
        wf = self._compute_starting_weights(Ys_init, scaling, n, q)

        # --- Cellwise weight matrices (1 = clean, 0 = outlying cell) ---
        cell_wx = np.ones((n, p), dtype=np.float64)
        cell_wy = np.ones((n, q), dtype=np.float64)
        x_cellwise_outliers = np.zeros((n, p), dtype=bool)
        y_cellwise_outliers = np.zeros((n, q), dtype=bool)

        if ddc_x_outliers is not None:
            cell_wx[ddc_x_outliers] = 0.0
            x_cellwise_outliers |= ddc_x_outliers
        if ddc_y_outliers is not None:
            cell_wy[ddc_y_outliers] = 0.0
            y_cellwise_outliers |= ddc_y_outliers

        # Combined per-cell weight matrix:  WEmat[i,j] = sqrt(case_w[i] * cell_w[i,j])
        WEmat_x = np.sqrt(we[:, np.newaxis] * cell_wx)
        WEmat_y = np.sqrt(wf[:, np.newaxis] * cell_wy)
        # Use DDC-imputed Xs_init for the twoblock fit so that undetected cells
        # in low-scale variables (which may be extreme in the original Xs) do
        # not dominate the SVD when their case weight has not yet been reduced.
        Xw = (Xs_init * WEmat_x).astype("float64")
        Yw = (Ys_init * WEmat_y).astype("float64")

        scalingt = copy.deepcopy(scaling)
        scalingu = copy.deepcopy(scaling)

        res_tb = twoblock(
            n_components_x=self.n_components_x,
            n_components_y=self.n_components_y,
            sparse=self.sparse,
            eta_x=self.eta_x,
            eta_y=self.eta_y,
            centre="mean",
            scale="None",
            verbose=False,
            copy=False,
            gpu=self.gpu,
        )

        loops = 1
        rold = 1e-5
        difference = 1
        w0 = []

        while (difference > self.tol) and (loops < self.maxit):
            res_tb.fit(Xw, Yw)

            # Dual-reference scores for case-weight update:
            #   T_ref  = Xs_init @ weights  (DDC-imputed, "clean" reference)
            #   T_cont = Xs @ weights       (original contaminated data)
            # scalingt is fit on T_ref so the Hampel cutoffs are calibrated to
            # the clean distribution.  Rows with undetected contamination then
            # produce elevated distances and get downweighted correctly.
            # When start_cellwise=False, Xs_init=Xs so T_ref=T_cont and the
            # behaviour is identical to the single-reference case.
            Tx_ref = np.matmul(Xs_init, res_tb.x_weights_)
            Ty_ref = np.matmul(Ys_init, res_tb.y_weights_)
            Tx = np.matmul(Xs, res_tb.x_weights_)
            Ty = np.matmul(Ys, res_tb.y_weights_)

            # Case weight update + normalized distances for SPADIMO flagging
            wte, dists_x = self._update_weights_unweighted(
                Tx, self.n_components_x, scalingt, T_ref=Tx_ref
            )
            wue, dists_y = self._update_weights_unweighted(
                Ty, self.n_components_y, scalingu, T_ref=Ty_ref
            )

            b = res_tb.coef_scaled_
            b2sum = np.sum(np.power(b, 2))
            difference = abs(b2sum - rold) / rold
            rold = b2sum

            # Process X case weights
            wte = np.array(wte).reshape(-1)
            we = wte.astype("float64")
            w0 = []
            if any(we < 1e-06):
                w0 = np.where(we < 1e-06)[0]
                we[w0] = 1e-06
                we = np.array(we, dtype=np.float64)
            if len(w0) >= (n / 2):
                break

            # Process Y case weights
            wue = np.array(wue).reshape(-1)
            wf = wue.astype("float64")
            w0 = []
            if any(wf < 1e-06):
                w0 = np.where(wf < 1e-06)[0]
                wf[w0] = 1e-06
                wf = np.array(wf, dtype=np.float64)
            if len(w0) >= (n / 2):
                break

            # --- Cellwise outlier detection via SPADIMO ---
            # Cells accumulate: once flagged they remain at zero.
            for i in np.where(dists_x > chi2_thr_x)[0]:
                try:
                    sp = spadimo(
                        scale=spadi_scale,
                        etas=spadieta,
                        stop_early=True,
                        gpu=self.gpu,
                        copy=False,
                    )
                    sp.fit(Xs, we, obs=int(i))
                    if len(sp.outlvars_) > 0:
                        x_cellwise_outliers[i, sp.outlvars_] = True
                        cell_wx[i, sp.outlvars_] = 0.0
                except Exception:
                    pass

            for i in np.where(dists_y > chi2_thr_y)[0]:
                try:
                    sp = spadimo(
                        scale=spadi_scale,
                        etas=spadieta,
                        stop_early=True,
                        gpu=self.gpu,
                        copy=False,
                    )
                    sp.fit(Ys, wf, obs=int(i))
                    if len(sp.outlvars_) > 0:
                        y_cellwise_outliers[i, sp.outlvars_] = True
                        cell_wy[i, sp.outlvars_] = 0.0
                except Exception:
                    pass

            # Rebuild combined weight matrices and weighted data
            WEmat_x = np.sqrt(we[:, np.newaxis] * cell_wx)
            WEmat_y = np.sqrt(wf[:, np.newaxis] * cell_wy)
            Xw = (Xs_init * WEmat_x).astype("float64")
            Yw = (Ys_init * WEmat_y).astype("float64")
            loops += 1

        if difference > self.tol:
            warnings.warn(
                "Method did not converge. The scaled difference between "
                "norms of the coefficient vectors is "
                + str(round(difference, 4))
            )

        # Final case weights: zero out fully-rejected observations
        w = wf.copy()
        if len(w0) > 0:
            w[w0] = 0
        wt = we.copy()
        if len(w0) > 0:
            wt[w0] = 0

        # Final unweighted X scores
        T = np.matmul(Xs, res_tb.x_weights_)

        # Rescale coefficients to original scale
        if self.centre == "None" and self.scale == "None":
            B_rescaled = res_tb.coef_scaled_
        else:
            sX_safe = sX.copy()
            sX_safe[sX_safe < 1e-10] = 1.0
            B_rescaled = np.multiply(
                np.outer(sy, np.divide(1, sX_safe)).T, res_tb.coef_scaled_
            )

        Yp_rescaled = np.matmul(X, B_rescaled)
        if self.centre == "None":
            intercept = 0
        elif self.centre == "mean":
            intercept = np.mean(Y - Yp_rescaled, axis=0)
        else:
            intercept = np.median(Y - Yp_rescaled, axis=0)

        Yfit = Yp_rescaled + intercept
        R = Y - Yfit

        setattr(self, "x_weights_", res_tb.x_weights_)
        setattr(self, "y_weights_", res_tb.y_weights_)
        setattr(self, "x_loadings_", res_tb.x_loadings_)
        setattr(self, "y_loadings_", res_tb.y_loadings_)
        setattr(self, "x_scores_", T)
        setattr(self, "y_scores_", res_tb.y_scores_)
        setattr(self, "coef_", B_rescaled)
        setattr(self, "intercept_", intercept)
        setattr(self, "coef_scaled_", res_tb.coef_scaled_)
        setattr(self, "intercept_scaled_", res_tb.intercept_)
        setattr(self, "residuals_", R)
        setattr(self, "fitted_", Yfit)
        setattr(self, "x_loc_", mX)
        setattr(self, "y_loc_", my)
        setattr(self, "x_sca_", sX)
        setattr(self, "y_sca_", sy)
        setattr(self, "x_caseweights_", wt)
        setattr(self, "y_caseweights_", w)
        setattr(self, "caseweights_", wt * w)
        setattr(self, "x_colret_", res_tb.x_colret_)
        setattr(self, "y_colret_", res_tb.y_colret_)
        setattr(self, "x_indret_", res_tb.x_indret_)
        setattr(self, "y_indret_", res_tb.y_indret_)
        setattr(self, "centring_", scaling)
        setattr(self, "scalingt_", scalingt)
        setattr(self, "scalingu_", scalingu)
        # Cellwise-specific attributes
        setattr(self, "x_cellweights_", cell_wx)
        setattr(self, "y_cellweights_", cell_wy)
        setattr(self, "x_cellwise_outliers_", x_cellwise_outliers)
        setattr(self, "y_cellwise_outliers_", y_cellwise_outliers)
        setattr(self, "ddc_x_outliers_", ddc_x_outliers)
        setattr(self, "ddc_y_outliers_", ddc_y_outliers)
        return self

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    def predict(self, Xn):
        """
        Predict response values for new observations.

        Parameters
        ----------
        Xn : array-like (n, p)
            New predictor data.

        Returns
        -------
        Yhat : ndarray (n, q)
        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise ValueError(
                "New data must have same number of columns as the training X"
            )
        return np.matmul(Xn, self.coef_) + self.intercept_

    def transform(self, Xn):
        """
        Project new X data into the X score space.

        Parameters
        ----------
        Xn : array-like (n, p)
            New predictor data.

        Returns
        -------
        T : ndarray (n, n_components_x)
        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise ValueError(
                "New data must have same number of columns as the training X"
            )
        from ._preproc_utilities import scale_data
        Xnc = scale_data(Xn, self.x_loc_, self.x_sca_)
        return np.matmul(Xnc, self.x_weights_)

    def impute(self, X, Y):
        """
        Impute cellwise outliers in X and Y using the fitted twoblock model.

        Outlying cells are replaced by their reconstruction from the twoblock
        loadings. For each block, the contribution of the outlying cells to the
        score is zeroed out (treated as the column mean in scaled space), the
        scores are projected, and the reconstructed values are back-transformed
        to the original scale.

        For data with the same number of rows as the training set the stored
        cellwise outlier maps (x_cellwise_outliers_, y_cellwise_outliers_) are
        used directly. For new data SPADIMO is run on each observation using
        uniform case weights to detect outlying cells before imputing.

        Parameters
        ----------
        X : array-like (n, p)
            X block.
        Y : array-like (n, q)
            Y block.

        Returns
        -------
        X_imputed : ndarray (n, p)
            X with outlying cells replaced by model-consistent values.
        Y_imputed : ndarray (n, q)
            Y with outlying cells replaced by model-consistent values.
        """
        from ._preproc_utilities import scale_data

        if type(X) == ps.core.frame.DataFrame:
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        if p != self.X.shape[1]:
            raise ValueError(
                f"X must have {self.X.shape[1]} columns, got {p}"
            )

        if type(Y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            Y = Y.to_numpy()
        Y = np.asarray(Y, dtype=np.float64)
        ny, q = Y.shape
        if ny != n:
            raise ValueError("X and Y must have the same number of rows")
        if q != self.Y.shape[1]:
            raise ValueError(
                f"Y must have {self.Y.shape[1]} columns, got {q}"
            )

        if n == self.X.shape[0]:
            # Training data: use stored outlier maps directly
            x_out = self.x_cellwise_outliers_.copy()
            y_out = self.y_cellwise_outliers_.copy()
        else:
            # New data: detect outlying cells per observation via SPADIMO
            spadi_scale = self.scale if self.scale != "None" else "mad"
            spadieta = (
                np.arange(0.9, 0.05, -0.1)
                if self.spadieta is None
                else np.asarray(self.spadieta)
            )
            uniform_w = np.ones(n)
            Xs_new = scale_data(X, self.x_loc_, self.x_sca_)
            Ys_new = scale_data(Y, self.y_loc_, self.y_sca_)
            x_out = np.zeros((n, p), dtype=bool)
            y_out = np.zeros((n, q), dtype=bool)
            for i in range(n):
                try:
                    sp = spadimo(
                        scale=spadi_scale, etas=spadieta,
                        stop_early=True, gpu=self.gpu, copy=False,
                    )
                    sp.fit(Xs_new, uniform_w, obs=i)
                    if len(sp.outlvars_) > 0:
                        x_out[i, sp.outlvars_] = True
                except Exception:
                    pass
                try:
                    sp = spadimo(
                        scale=spadi_scale, etas=spadieta,
                        stop_early=True, gpu=self.gpu, copy=False,
                    )
                    sp.fit(Ys_new, uniform_w, obs=i)
                    if len(sp.outlvars_) > 0:
                        y_out[i, sp.outlvars_] = True
                except Exception:
                    pass

        X_imputed = self._impute_block(
            X, x_out, self.x_loc_, self.x_sca_,
            self.x_weights_, self.x_loadings_,
        )
        Y_imputed = self._impute_block(
            Y, y_out, self.y_loc_, self.y_sca_,
            self.y_weights_, self.y_loadings_,
        )
        return X_imputed, Y_imputed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _impute_block(self, Z, outlier_map, loc, sca, weights, loadings):
        """
        Impute outlying cells in a single data block.

        For each row with at least one outlying cell:
          1. Scale the row.
          2. Zero out the outlying cells in the scaled copy (treat as column mean).
          3. Project onto the weight vectors to get scores.
          4. Reconstruct all variables via the loadings.
          5. Back-transform the imputed cells to the original scale.

        Parameters
        ----------
        Z : ndarray (n, d)
        outlier_map : bool ndarray (n, d)
        loc : ndarray (d,)
        sca : ndarray (d,)
        weights : ndarray (d, k)
        loadings : ndarray (d, k)

        Returns
        -------
        Z_imputed : ndarray (n, d)
        """
        Z_imputed = Z.copy()
        rows_with_outliers = np.where(outlier_map.any(axis=1))[0]
        if len(rows_with_outliers) == 0:
            return Z_imputed

        sca_safe = sca.copy()
        sca_safe[sca_safe < 1e-10] = 1.0
        Zs = (Z - loc) / sca_safe

        for i in rows_with_outliers:
            out_cols = np.where(outlier_map[i])[0]
            zs_mod = Zs[i].copy()
            zs_mod[out_cols] = 0.0                   # zero out outlying cells
            t_i = zs_mod @ weights                   # scores from clean cells (k,)
            zs_hat_i = t_i @ loadings.T              # full reconstruction (d,)
            Z_imputed[i, out_cols] = (
                loc[out_cols] + sca_safe[out_cols] * zs_hat_i[out_cols]
            )
        return Z_imputed

    def _update_weights_unweighted(self, T, n_components, scalingt, T_ref=None):
        """
        Compute case weights from unweighted score matrix T.

        Unlike rtb._update_weights, which recovers unweighted scores by
        dividing by a row-uniform WEmat column, this method takes the
        unweighted scores directly (T = Xs @ x_weights_).  This is necessary
        because with per-cell weights WEmat_x[i,j] = sqrt(we[i]*cell_wx[i,j])
        the columns of WEmat are no longer identical and the simple division
        trick no longer recovers the unweighted scores.

        Parameters
        ----------
        T : ndarray (n, n_components)
            Unweighted scores from original (possibly contaminated) data.
        n_components : int
            Number of components (used for chi2 cutoffs).
        scalingt : VersatileScaler
            Scaler used to robustly center/scale the scores column-wise.
        T_ref : ndarray (n, n_components) or None
            Clean reference scores (e.g. from DDC-imputed data).  When
            provided, scalingt is fit on T_ref so the Hampel cutoffs are
            calibrated to the clean distribution.  Rows with undetected
            contamination in T then show elevated distances relative to this
            reference and get properly downweighted.  When None (or when
            T_ref is the same array as T, i.e. start_cellwise=False) the
            behaviour reduces to fitting scalingt on T itself.

        Returns
        -------
        wte : ndarray (n,)
            Updated case weights from the M-estimation function.
        wtn : ndarray (n,)
            Normalized score distances used to select observations for SPADIMO.
        """
        if self.scale == "None":
            scalingt.set_params(scale="mad")
        if T_ref is not None:
            # Fit on clean reference; transform the (possibly contaminated) T
            dt_ref = scalingt.fit_transform(T_ref)
            ref_norms = np.sqrt(
                np.array(np.sum(np.square(dt_ref), 1), dtype=np.float64)
            )
            ref_med = float(np.median(ref_norms))
            if ref_med < 1e-10:
                ref_med = 1.0
            dt = scalingt.transform(T)
        else:
            dt = scalingt.fit_transform(T)
        wtn = np.sqrt(np.array(np.sum(np.square(dt), 1), dtype=np.float64))
        if T_ref is not None:
            wtn = wtn / ref_med
        else:
            wtn = wtn / np.median(wtn)
        wtn = wtn.reshape(-1)

        if self.fun == "Fair":
            wte = Fair(wtn.copy(), self.probctx_)
        elif self.fun == "Huber":
            wte = Huber(wtn.copy(), self.probctx_)
        elif self.fun == "Hampel":
            self.probctx_ = chi2.ppf(self.probp1, n_components)
            self.hampelbx_ = chi2.ppf(self.probp2, n_components)
            self.hampelrx_ = chi2.ppf(self.probp3, n_components)
            wte = Hampel(wtn.copy(), self.probctx_, self.hampelbx_, self.hampelrx_)
        return wte, wtn

    def _ddc_impute(self, Z, block_name):
        """
        Run DDC on data block Z and return (imputed_Z, outlier_bool_matrix).

        Falls back to (Z, None) if robpy is unavailable or DDC raises.

        Parameters
        ----------
        Z : ndarray (n, d)
        block_name : str
            Label used in messages ('X' or 'Y').

        Returns
        -------
        Z_out : ndarray (n, d)
        outliers : ndarray of bool (n, d) or None
        """
        try:
            from robpy.outliers.ddc import DDC
            n, d = Z.shape
            Z_df = ps.DataFrame(
                Z, columns=[f"{block_name}{j}" for j in range(d)]
            )
            ddc = DDC()
            ddc.fit(Z_df)
            outliers = ddc.cellwise_outliers_
            if hasattr(outliers, "to_numpy"):
                outliers = outliers.to_numpy()
            Z_imputed = ddc.impute(Z_df)
            if hasattr(Z_imputed, "to_numpy"):
                Z_imputed = Z_imputed.to_numpy()
            if self.verbose:
                n_out = int(np.sum(outliers))
                pct = 100.0 * n_out / outliers.size
                print(
                    f"DDC ({block_name} block): {n_out} cellwise outliers "
                    f"({pct:.2f}%) detected and imputed"
                )
            return Z_imputed.astype("float64"), outliers.astype(bool)
        except ImportError:
            warnings.warn(
                "robpy not available (pip install robpy). "
                f"Falling back to casewise starting values for {block_name} block."
            )
            return Z, None
        except Exception as exc:
            warnings.warn(
                f"DDC failed for {block_name} block: {exc}. "
                "Falling back to casewise starting values."
            )
            return Z, None
