# -*- coding: utf-8 -*-
"""
Robust Twoblock (RTB) - Iteratively Reweighted Two-Block Dimension Reduction

Applies M-estimation style iterative reweighting to the twoblock method,
using case weights derived from the X block score space (analogous to SPRM)
applied to both X and Y blocks.

@author: Sven Serneels
"""

import numpy as np
from sklearn.base import (
    RegressorMixin,
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
)
from sklearn.utils.metaestimators import _BaseComposition
from scipy.stats import norm, chi2
import copy
import pandas as ps
import warnings
from .utils import (
    _check_input, _predict_check_input,
    Fair, Huber, Hampel, brokenstick,
)
from .prepro import VersatileScaler
from .twoblock import twoblock
from ._preproc_utilities import mad


class rtb(
    _BaseComposition,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
):
    """
    RTB Robust Twoblock - Iteratively Reweighted Two-Block Dimension Reduction

    Applies M-estimation style iterative reweighting (as in SPRM) to the
    twoblock method. Case weights are derived from the X block score space
    and applied to both X and Y blocks.

    Parameters
    -----------

    n_components_x : int, min 1. Note that if applied on data,
        n_components_x shall take a value <= min(x_data.shape)

    n_components_y : int, min 1. Note that if applied on data,
        n_components_y shall take a value <= min(y_data.shape)
        If unspecified, set to equal n_components_x

    sparse : bool, default False
        If False, the dense twoblock method by Cook et al.
        If True, the sparse twoblock method by Serneels, which requires
            specifying the eta_x and eta_y sparsity parameters

    eta_x : float, default 0.5
        X block sparsity parameter

    eta_y : float, default 0.5
        Y block sparsity parameter

    fun : str
        Downweighting function. 'Hampel' (recommended), 'Fair' or 'Huber'

    probp1 : float
        Probability cutoff for start of downweighting (e.g. 0.95)

    probp2 : float
        Probability cutoff for start of steep downweighting
        (e.g. 0.975, only relevant if fun='Hampel')

    probp3 : float
        Probability cutoff for start of outlier omission
        (e.g. 0.999, only relevant if fun='Hampel')

    centre : str,
        Type of centring ('mean', 'median', 'l1median' or 'kstepLTS')

    scale : str,
        Type of scaling ('std', 'mad', 'scaleTau2' or 'None')

    verbose : boolean (def True)
        Specifying verbose mode

    maxit : int
        Maximal number of iterations in M algorithm

    tol : float
        Tolerance for convergence in M algorithm

    start_cutoff_mode : str,
        'specific' will set starting value cutoffs specific to X;
        any other value will set X cutoffs identically to y cutoffs.

    start_X_init : str,
        'pcapp' will include a PCA/broken stick projection to calculate
        the starting weights; any other value will calculate starting
        values based on the X matrix itself.

    copy : (def True): boolean,
        Whether to copy data into the object.


    Attributes
    ------------
    Attributes always provided:

        -  `x_weights_`: X block PLS weighting vectors (W)
        -  `y_weights_`: Y block PLS weighting vectors (V)
        -  `x_loadings_`: X block PLS loading vectors (P)
        -  `y_loadings_`: Y block PLS loading vectors (Q)
        -  `x_scores_`: X block PLS score vectors (T)
        -  `y_scores_`: Y block PLS score vectors (U)
        -  `coef_`: regression coefficients
        -  `intercept_`: intercept
        -  `coef_scaled_`: scaled regression coefficients
        -  `intercept_scaled_`: scaled intercept
        -  `residuals_`: regression residuals
        -  `fitted_`: fitted response
        -  `x_loc_`: X block location estimate
        -  `y_loc_`: Y block location estimate
        -  `x_sca_`: X block scale estimate
        -  `y_sca_`: Y block scale estimate
        -  `x_caseweights_`: X block case weights
        -  `caseweights_`: combined case weights
        -  `centring_`: scaling object used internally

    """

    def __init__(
        self,
        n_components_x=1,
        n_components_y=None,
        sparse=False,
        eta_x=.5,
        eta_y=.5,
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
    ):
        self.n_components_x = n_components_x
        self.n_components_y = n_components_y
        self.sparse = sparse
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.fun = fun
        self.probp1 = probp1
        self.probp2 = probp2
        self.probp3 = probp3
        self.centre = centre
        self.scale = scale
        self.verbose = verbose
        self.maxit = maxit
        self.tol = tol
        self.start_cutoff_mode = start_cutoff_mode
        self.start_X_init = start_X_init
        self.copy = copy
        self.probctx_ = "irrelevant"
        self.hampelbx_ = "irrelevant"
        self.hampelrx_ = "irrelevant"

    def fit(self, X, Y):
        """
        Fit a Robust Twoblock model.

        Parameters
        ------------

            X : numpy array or Pandas data frame
                Predictor data.

            Y : numpy array or Pandas data frame
                Response data.

        Returns
        -------
        rtb class object containing the estimated parameters.

        """
        if type(X) == ps.core.frame.DataFrame:
            colx = X.columns
            X = X.to_numpy()
        else:
            colx = []
        (n, p) = X.shape
        if type(Y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            coly = Y.columns
            Y = Y.to_numpy()
        else:
            coly = []
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

        # Centre and scale
        scaling = VersatileScaler(center=self.centre, scale=self.scale)
        Xs = scaling.fit_transform(X).astype("float64")
        mX = scaling.col_loc_
        sX = scaling.col_sca_
        Ys = scaling.fit_transform(Y).astype("float64")
        my = scaling.col_loc_
        sy = scaling.col_sca_

        # Starting weights for both blocks
        we = self._compute_starting_weights(Xs, scaling, n, p)
        wf = self._compute_starting_weights(Ys, scaling, n, q)

        # Weight matrices for X and Y
        WEmat_x = np.array(
            [np.sqrt(we) for _ in range(p)], ndmin=1
        ).T
        WEmat_y = np.array(
            [np.sqrt(wf) for _ in range(q)], ndmin=1
        ).T
        Xw = np.multiply(Xs, WEmat_x).astype("float64")
        Yw = np.multiply(Ys, WEmat_y).astype("float64")

        scalingt = copy.deepcopy(scaling)
        scalingu = copy.deepcopy(scaling)

        # Iteratively reweighted twoblock
        loops = 1
        rold = 1e-5
        difference = 1

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
        )

        while (difference > self.tol) and (loops < self.maxit):
            res_tb.fit(Xw, Yw)

            # Update weights based on X score distances
            wte = self._update_weights(
                res_tb.x_scores_, WEmat_x, self.n_components_x, scalingt
            )
            wue = self._update_weights(
                res_tb.y_scores_, WEmat_y, self.n_components_y, scalingu
            )

            b = res_tb.coef_scaled_

            # Check convergence
            b2sum = np.sum(np.power(b, 2))
            difference = abs(b2sum - rold) / rold
            rold = b2sum

            wte = np.array(wte).reshape(-1)
            we = wte.astype("float64")
            w0 = []
            if any(we < 1e-06):
                w0 = np.where(we < 1e-06)[0]
                we[w0] = 1e-06
                we = np.array(we, dtype=np.float64)
            if len(w0) >= (n / 2):
                break

            wue = np.array(wue).reshape(-1)
            wf = wue.astype("float64")
            w0 = []
            if any(wf < 1e-06):
                w0 = np.where(wf < 1e-06)[0]
                wf[w0] = 1e-06
                wf = np.array(wf, dtype=np.float64)
            if len(w0) >= (n / 2):
                break

            # Reweight both X and Y
            WEmat_x = np.array(
                [np.sqrt(we) for _ in range(p)], ndmin=1
            ).T
            WEmat_y = np.array(
                [np.sqrt(wf) for _ in range(q)], ndmin=1
            ).T
            Xw = np.multiply(Xs, WEmat_x).astype("float64")
            Yw = np.multiply(Ys, WEmat_y).astype("float64")
            loops += 1

        if difference > self.tol:
            warnings.warn(
                "Method did not converge. The scaled difference between "
                "norms of the coefficient vectors is "
                + str(round(difference, 4))
            )

        # Final weights
        w = wf.copy()
        if len(w0) > 0:
            w[w0] = 0
        wt = we.copy() if isinstance(wte, np.ndarray) else np.array(wte)
        if len(w0) > 0:
            wt[w0] = 0

        # Recompute scores on unweighted scaled data
        T = np.matmul(Xs, res_tb.x_weights_)

        # Rescale coefficients
        if self.centre == "None" and self.scale == "None":
            B_rescaled = res_tb.coef_scaled_
        else:
            B_rescaled = np.multiply(
                np.outer(sy, np.divide(1, sX)).T, res_tb.coef_scaled_
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
        return self

    def _update_weights(self, scores, WEmat, n_components, scalingt):
        """
        Unweight scores and compute updated case weights from score distances.

        Parameters
        ----------
        scores : numpy array
            Weighted score matrix (n x n_components).
        WEmat : numpy array
            Weight expansion matrix (n x d).
        n_components : int
            Number of components to use.
        scalingt : VersatileScaler
            Scaling object for transforming scores.

        Returns
        -------
        wte : numpy array
            Updated case weights (n,).
        """
        T = np.divide(scores, WEmat[:, 0:n_components])

        scalet = self.scale
        if scalet == "None":
            scalingt.set_params(scale="mad")
        dt = scalingt.fit_transform(T)
        wtn = np.sqrt(
            np.array(np.sum(np.square(dt), 1), dtype=np.float64)
        )
        wtn = wtn / np.median(wtn)
        wtn = wtn.reshape(-1)

        if self.fun == "Fair":
            wte = Fair(wtn, self.probctx_)
        elif self.fun == "Huber":
            wte = Huber(wtn, self.probctx_)
        elif self.fun == "Hampel":
            self.probctx_ = chi2.ppf(self.probp1, n_components)
            self.hampelbx_ = chi2.ppf(self.probp2, n_components)
            self.hampelrx_ = chi2.ppf(self.probp3, n_components)
            wte = Hampel(
                wtn, self.probctx_, self.hampelbx_, self.hampelrx_
            )
        return wte

    def _compute_starting_weights(self, Zs, scaling, n, d):
        """
        Compute starting case weights for a data block.

        Parameters
        ----------
        Zs : numpy array
            Centred/scaled data block (n x d).
        scaling : VersatileScaler
            Scaling object for transforming projections.
        n : int
            Number of cases.
        d : int
            Number of variables in the block.

        Returns
        -------
        we : numpy array
            Case weights (n,), floored at 1e-06.
        """
        if self.start_X_init == "pcapp":
            U, S, V = np.linalg.svd(Zs, full_matrices=False)
            spc = np.square(S)
            spc /= np.sum(spc)
            relcomp = max(
                np.where(spc - brokenstick(min(d, n))[:, 0] <= 0)[0][0], 1
            )
            Urc = np.array(U[:, 0:relcomp])
            Us = scaling.fit_transform(Urc)
        else:
            Us = Zs
            relcomp = d

        wz = np.sqrt(np.array(np.sum(np.square(Us), 1), dtype=np.float64))
        wz = wz / np.median(wz)

        probct = chi2.ppf(self.probp1, relcomp) if self.start_cutoff_mode == "specific" else norm.ppf(self.probp1)
        self.probctx_ = probct

        if self.fun == "Fair":
            wz = Fair(wz, probct)
        elif self.fun == "Huber":
            wz = Huber(wz, probct)
        elif self.fun == "Hampel":
            if self.start_cutoff_mode == "specific":
                self.hampelbx_ = chi2.ppf(self.probp2, relcomp)
                self.hampelrx_ = chi2.ppf(self.probp3, relcomp)
            else:
                self.hampelbx_ = norm.ppf(self.probp2)
                self.hampelrx_ = norm.ppf(self.probp3)
            wz = Hampel(wz, probct, self.hampelbx_, self.hampelrx_)

        wz = np.array(wz).reshape(-1)
        we = wz.astype("float64")
        if (we < 1e-06).any():
            w0 = np.where(we < 1e-06)[0]
            we[w0] = 1e-06
        return we

    def predict(self, Xn):
        """
        Predict cases.

        Parameters
        ------------

            Xn : numpy array or data frame
                Input data.

        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise ValueError(
                "New data must have same number of columns as the ones "
                "the model has been trained with"
            )
        return np.matmul(Xn, self.coef_) + self.intercept_

    def transform(self, Xn):
        """
        Transform input data to X score space.

        Parameters
        ------------

            Xn : numpy array or data frame
                Input data.

        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise ValueError(
                "New data must have same number of columns as the ones "
                "the model has been trained with"
            )
        from ._preproc_utilities import scale_data
        Xnc = scale_data(Xn, self.x_loc_, self.x_sca_)
        return np.matmul(Xnc, self.x_weights_)

    def weightnewx(self, Xn):
        """
        Calculate case weights for new data based on the projection
        in the twoblock X score space.
        """
        n, p, Xn = _predict_check_input(Xn)
        if p != self.X.shape[1]:
            raise ValueError(
                "New data must have same number of columns as the ones "
                "the model has been trained with"
            )
        Tn = self.transform(Xn)
        scaling = self.scalingt_
        scalet = self.scale
        if scalet == "None":
            scaling.set_params(scale="mad")
        if isinstance(Tn, np.matrix):
            Tn = np.array(Tn)
        dtn = scaling.fit_transform(Tn)
        wtn = np.sqrt(np.array(np.sum(np.square(dtn), 1), dtype=np.float64))
        wtn = wtn / np.median(wtn)
        wtn = wtn.reshape(-1)
        if self.fun == "Fair":
            wtn = Fair(wtn, self.probctx_)
        elif self.fun == "Huber":
            wtn = Huber(wtn, self.probctx_)
        elif self.fun == "Hampel":
            wtn = Hampel(wtn, self.probctx_, self.hampelbx_, self.hampelrx_)
        return wtn
