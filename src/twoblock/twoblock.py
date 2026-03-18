# -*- coding: utf-8 -*-
"""
Created on Tue Aug 6 16:39:57 2024

@author: SERNEELS
"""

import numpy as np
from sklearn.base import (
    RegressorMixin,
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
)
from sklearn.utils.metaestimators import _BaseComposition
import copy
import pandas as ps
from .utils import _check_input, _predict_check_input
from .prepro import VersatileScaler
from .gsspp import GenSpatialSignPreProcessor
from scipy.linalg import pinv

# Draft version


class twoblock(
    _BaseComposition,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
):
    """
    TWOBLOCK Dense and Sparse Two-Block Simultaneous Dimension Reduction of 
    Multivariate X and Y data blocks

    Parameters
    -----------

    n_components_x : int, min 1. Note that if applied on data,
        n_components_x shall take a value <= min(x_data.shape)

    n_components_y : int, min 1. Note that if applied on data,
        n_components_x shall take a value <= min(x_data.shape)
        If unspecified, set to equal n_components_x

    verbose: Boolean (def True)
                to print intermediate set of columns retained

    centre : str,
                type of centring (`'mean'` [recommended], `'median'` or `'l1median'`),

    scale : str,
             type of scaling ('std','mad' or 'None')
             
    sparse : bool, default False 
        If False, the dense twoblock (or 'XY-PLS') method by Cook et al. [1]
        If True, the Sparse twoblock method by Serneels [2], which requires 
            specifying the eta_x and eta_y sparsity parameters
            
    eta_x: float, default 0.5, 
        X block sparsity parameter 
        
    eta_y. float, default 0.5
        Y block sparsity parameter (in the paper: kappa) 
        
    robust, Boolean (def False)
        whether to robustify the estimate by appplying generalized 
        spatial sign pre-processing. 
        
    fun, str, default 'linear_redescending.' Function to apply in generalized   
        spatial sign transformation. For options, see `gsspp.py`.

    copy : (def True): boolean,
             whether to copy data into twoblock object.


    Attributes
    ------------
    Attributes always provided:

        -  `x_weights_`: X block PLS weighting vectors (usually denoted W)
        -  `y_weights_`: Y block PLS weighting vectors (usually denoted V)
        -  `x_loadings_`: X block PLS loading vectors (usually denoted P)
        -  `y_loadings_`: Y block PLS loading vectors (usually denoted Q)
        -  `x_scores_`: X block PLS score vectors (usually denoted T)
        -  `y_scores_`: Y block PLS score vectors (usually denoted U)
        -  `coef_`: vector of regression coefficients
        -  `intercept_`: intercept
        -  `coef_scaled_`: vector of scaled regression coeeficients (when scaling option used)
        -  `intercept_scaled_`: scaled intercept
        -  `residuals_`: vector of regression residuals
        -  `fitted_`: fitted response
        -  `x_loc_`: X block location estimate
        -  `y_loc_`: y location estimate
        -  `x_sca_`: X block scale estimate
        -  `y_sca_`: y scale estimate
        -  `centring_`: scaling object used internally (type: `VersatileScaler`)
        
    If robust = True, 
        
        -  `gss_x_`: The X block generalized spatial sign preprocessing object
        -  `gss_y_`: The Y block generalized spatial sign preprocessing object


    References
    ----------
    [1] (dense)
    Cook, R. Dennis, Liliana Forzani, and Lan Liu.
    "Partial least squares for simultaneous reduction of response and predictor
    vectors in regression." Journal of Multivariate Analysis 196 (2023): 105163.
    
    [2] (sparse)
    S. Serneels. "Sparse Twoblock Dimension Reduction: A Versatile Alternative 
    to Sparse PLS2 and CCA." Journal of Chemometrics, 39 (2025): e70051.

    """

    def __init__(
        self,
        n_components_x=1,
        n_components_y=None,
        centre="mean",
        scale="None",
        sparse=False,
        eta_x=.5,
        eta_y=.5,
        robust = False,
        fun = 'linear_redescending',
        verbose=True,
        copy=True,
        **kwargs
    ):
        self.n_components_x = n_components_x
        self.n_components_y = n_components_y
        self.centre = centre
        self.scale = scale
        self.sparse = sparse
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.robust = robust
        self.fun = fun
        self.verbose = verbose
        self.copy = copy

        if 'zero_value' in kwargs:
            self.zero_value = kwargs.pop('zero_value')
        else:
            self.zero_value = 1e-8

    def fit(self, X, Y):
        """
        Fit a Twoblock model.

        Parameters
        ------------

            X : numpy array or Pandas data frame
                Predictor data.

            Y : numpy array or Pandas data frame
                Response data

        Returns
        -------
        twoblock class object containing the estimated parameters.

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
            raise (ValueError("Number of cases in X and Y needs to agree"))

        Y = Y.astype("float64")

        if self.n_components_y is None:
            self.n_components_y = self.n_components_x

        assert self.n_components_x <= min(
            np.linalg.matrix_rank(np.matmul(X.T, X)), n - 1
        ), "Number of components cannot exceed covariance rank or number of cases"

        assert self.n_components_y <= min(
            np.linalg.matrix_rank(np.matmul(Y.T, Y)), n - 1
        ), "Number of components cannot exceed covariance rank or number of cases"

        if self.copy:
            X0 = copy.deepcopy(X)
            Y0 = copy.deepcopy(Y)
        else:
            X0 = X
            Y0 = Y
        if self.copy:
            self.X = X0
            self.Y = Y0

        X0 = X0.astype("float64")
        centring = VersatileScaler(
            center=self.centre, scale=self.scale, trimming=0
        )
        X0 = centring.fit_transform(X0).astype("float64")
        mX = centring.col_loc_
        sX = centring.col_sca_
        Y0 = centring.fit_transform(Y0).astype("float64")
        my = centring.col_loc_
        sy = centring.col_sca_

        self.x_scores_ = np.empty((n, self.n_components_x), float)
        self.y_scores_ = np.empty((n, self.n_components_y), float)
        self.x_weights_ = np.empty((p, self.n_components_x), float)
        self.y_weights_ = np.empty((q, self.n_components_y), float)
        self.x_loadings_ = np.empty((p, self.n_components_x), float)
        self.y_loadings_ = np.empty((q, self.n_components_y), float)
        self.x_expvar_ = np.empty(self.n_components_x,float)
        self.y_expvar_ = np.empty(self.n_components_y,float)

        X0var = np.var(X0.ravel())
        Y0var = np.var(Y0.ravel())
        Xvare = 0
        Yvare = 0

        if self.sparse:
            oldgoodies_x = np.array([])
            oldgoodies_y = np.array([])
            
        if self.robust:
            robopt = {'center': self.centre, 'fun': self.fun}
            gssx = GenSpatialSignPreProcessor(**robopt)
            X0 = gssx.fit_transform(X0)
            gssy = GenSpatialSignPreProcessor(**robopt)
            Y0 = gssy.fit_transform(Y0)
            
        Xh = copy.deepcopy(X0)
        Yh = copy.deepcopy(Y0)

        for i in range(self.n_components_x):
            
            sXY = np.dot(Xh.T, Y0) / n

            u, _, _ = np.linalg.svd(sXY)
            x_weights = u[:, 0].reshape((p,))
            if self.sparse:
                wh = x_weights / np.linalg.norm(x_weights)
                # goodies = abs(wh)-llambda/2 lambda definition
                goodies_x = abs(wh) - self.eta_x * max(abs(wh))
                wh = np.multiply(goodies_x, np.sign(wh))
                goodies_x = np.where((goodies_x > 0))[0]
                goodies_x = np.union1d(oldgoodies_x, goodies_x)
                oldgoodies_x = goodies_x
                if len(goodies_x) == 0:
                    colret = None
                    print(
                        "No X variables retained at"
                        + str(i)
                        + "latent variables"
                        + "and eta_x = "
                        + str(self.eta_x)
                        + ", try lower sparsity"
                    )
                    break
                elimvars_x = np.setdiff1d(range(0, p), goodies_x)
                x_weights[elimvars_x] = self.zero_value
            x_scores = np.dot(Xh, x_weights)
            x_loadings = np.dot(Xh.T, x_scores) / \
                np.dot(x_scores, x_scores)
            if self.sparse:
                goodies_x = goodies_x.astype(int)
                x_loadings[elimvars_x] = self.zero_value
                X0var = np.var(X0[:, goodies_x].ravel())

            scolo = np.outer(x_scores, x_loadings)
            Xh -= scolo
            Xvare += np.var(scolo.ravel())

            self.x_weights_[:, i] = x_weights
            self.x_scores_[:, i] = x_scores
            self.x_loadings_[:, i] = x_loadings
            self.x_expvar_[i] = Xvare/X0var

        for i in range(self.n_components_y):

            sYX = np.dot(Yh.T, X0) / n

            v, _, _ = np.linalg.svd(sYX)
            y_weights = v[:, 0].reshape((q,))

            if self.sparse:
                wh = y_weights / np.linalg.norm(y_weights)
                # goodies = abs(wh)-llambda/2 lambda definition
                goodies_y = abs(wh) - self.eta_y * max(abs(wh))
                wh = np.multiply(goodies_y, np.sign(wh))
                goodies_y = np.where((goodies_y > 0))[0]
                goodies_y = np.union1d(oldgoodies_y, goodies_y)
                oldgoodies_y = goodies_y
                if len(goodies_y) == 0:
                    colret = None
                    print(
                        "No Y variables retained at"
                        + str(i)
                        + "latent variables"
                        + "and eta_y = "
                        + str(self.eta_y)
                        + ", try lower sparsity"
                    )
                    break
                elimvars_y = np.setdiff1d(range(0, q), goodies_y)
                y_weights[elimvars_y] = self.zero_value

            y_scores = np.dot(Yh, y_weights)
            y_loadings = np.dot(Yh.T, y_scores) / \
                np.dot(y_scores, y_scores)

            if self.sparse:
                goodies_y = goodies_y.astype(int)
                y_loadings[elimvars_y] = self.zero_value
                Y0var = np.var(Y0[:, goodies_y].ravel())

            scolo = np.outer(y_scores, y_loadings)
            Yh -= scolo
            Yvare += np.var(scolo.ravel())

            self.y_weights_[:, i] = y_weights
            self.y_scores_[:, i] = y_scores
            self.y_loadings_[:, i] = y_loadings
            self.y_expvar_[i] = Yvare/Y0var

        wtx = np.dot(X0, self.x_weights_)
        if self.sparse:
            wti = pinv(np.dot(wtx.T, wtx))
        else:
            wti = np.linalg.inv(np.dot(wtx.T, wtx))
        swg = np.dot(wtx.T, np.dot(Y0, self.y_weights_))
        self.coef_scaled_ = np.matmul(
            np.matmul(self.x_weights_, wti), np.dot(swg, self.y_weights_.T)
        )

        if self.centre == "None" and self.scale == "None":
            B_rescaled = self.coef_scaled_
        else:
            # sklearn has this line wrong
            B_rescaled = np.multiply(
                np.outer(sy, np.divide(1, sX)).T, self.coef_scaled_
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
        
        if not(self.sparse):
            goodies_x = np.arange(p)
            goodies_y = np.arange(q)
        if len(colx) > 0:
            colrx = colx[goodies_x]
        else:
            colrx = goodies_x
        if len(coly) > 0:
            colry = coly[goodies_y]
        else:
            colry = goodies_y
            

        setattr(self, "coef_", B_rescaled)
        setattr(self, "intercept_", intercept)
        setattr(self, "fitted_", Yfit)
        setattr(self, "residuals_", R)
        setattr(self, "x_loc_", mX)
        setattr(self, "y_loc_", my)
        setattr(self, "x_sca_", sX)
        setattr(self, "y_sca_", sy)
        setattr(self, "centring_", centring)
        setattr(self, "x_colret_", colrx)
        setattr(self, "y_colret_", colry)
        setattr(self, "x_indret_", goodies_x)
        setattr(self, "y_indret_", goodies_y)
        
        if self.robust:
            setattr(self, "gss_x_", gssx)
            setattr(self, "gss_y_", gssx)
        
        return self

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
            raise (
                ValueError(
                    "New data must have same number of columns as the ones the model has been trained with"
                )
            )
        return np.matmul(Xn, self.coef_) + self.intercept_
