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
from direpack.preprocessing.robcent import VersatileScaler
from direpack.preprocessing._preproc_utilities import scale_data
from direpack.utils.utils import (
    _predict_check_input,
    _check_input,
)

# Draft version


class twoblock(
    _BaseComposition,
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
):
    """
    TWOBLOCK Two-Block Simultaneous Dimension Reduction of Multivariate X and Y
    data blocks

    Parameters
    -----------

    n_components_x : int, min 1. Note that if applied on data,
        n_components_x shall take a value <= min(x_data.shape)

    n_components_y : int, min 1. Note that if applied on data,
        n_components_x shall take a value <= min(x_data.shape)
        If unspecified, set to equal n_components_x

    verbose: Boolean (def true)
                to print intermediate set of columns retained

    centre : str,
                type of centring (`'mean'` [recommended], `'median'` or `'l1median'`),

    scale : str,
             type of scaling ('std','mad' or 'None')

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


    Reference
    ---------
    Cook, R. Dennis, Liliana Forzani, and Lan Liu.
    "Partial least squares for simultaneous reduction of response and predictor
    vectors in regression." Journal of Multivariate Analysis 196 (2023): 105163.

    """

    def __init__(
        self,
        n_components_x=1,
        n_components_y=None,
        verbose=True,
        centre="mean",
        scale="None",
        copy=True,
    ):
        self.n_components_x = n_components_x
        self.n_components_y = n_components_y
        self.verbose = verbose
        self.centre = centre
        self.scale = scale
        self.copy = copy

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
            X = X.to_numpy()
        (n, p) = X.shape
        if type(Y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
            Y = Y.to_numpy()
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

        Xh = copy.deepcopy(X0)
        Yh = copy.deepcopy(Y0)

        for i in range(self.n_components_x):

            sXY = np.dot(Xh.T, Y0) / n
            u, _, _ = np.linalg.svd(sXY)
            x_weights = u[:, 0].reshape((p,))
            x_scores = np.dot(Xh, x_weights)
            x_loadings = np.dot(Xh.T, x_scores) / np.dot(x_scores, x_scores)

            Xh -= np.outer(x_scores, x_loadings)

            self.x_weights_[:, i] = x_weights
            self.x_scores_[:, i] = x_scores
            self.x_loadings_[:, i] = x_loadings

        for i in range(self.n_components_y):

            sYX = np.dot(Yh.T, X0) / n

            v, _, _ = np.linalg.svd(sYX)
            y_weights = v[:, 0].reshape((q,))
            y_scores = np.dot(Yh, y_weights)
            y_loadings = np.dot(Yh.T, y_scores) / np.dot(y_scores, y_scores)

            Yh -= np.outer(y_scores, y_loadings)

            self.y_weights_[:, i] = y_weights
            self.y_scores_[:, i] = y_scores
            self.y_loadings_[:, i] = y_loadings

        wtx = np.dot(X0, self.x_weights_)
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

        setattr(self, "coef_", B_rescaled)
        setattr(self, "intercept_", intercept)
        setattr(self, "fitted_", Yfit)
        setattr(self, "residuals_", R)
        setattr(self, "x_loc_", mX)
        setattr(self, "y_loc_", my)
        setattr(self, "x_sca_", sX)
        setattr(self, "y_sca_", sy)
        setattr(self, "centring_", centring)
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
