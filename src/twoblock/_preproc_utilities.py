#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 10:55:24 2019

Set of help functions for robust centring and scaling 

@author: Sven Serneels, Ponalytics
"""

import numpy as np
import pandas as ps
import scipy.stats as sps
import scipy.optimize as spo
import copy


def _handle_zeros_in_scale(scale, copy=True):
    """
    Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    Taken from ScikitLearn.preprocesssing"""

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def _check_trimming(t):

    if (t > 0.99) or (t < 0):
        raise (ValueError("Trimming fraction must be in [0,1)"))


def mad(X, c=0.6744897501960817, **kwargs):
    """
    Column-wise median absolute deviation. **kwargs included to allow
    general function call in scale_data.
    """

    s = median(np.abs(X - median(X, axis=0)), axis=0) / c
    s = np.array(s).reshape(-1)
    # statsmodels.robust.mad is not as flexible toward matrix input,
    # sometimes throws a value error in ufunc
    return s


def median(X, **kwargs):
    """
    Column-wise median. **kwargs included to allow
    general function call in scale_data.
    """

    if np.isnan(X).any():
        m = np.nanmedian(X, axis=0)
    else:
        m = np.median(X, axis=0)
    m = np.array(m).reshape(-1)

    return m


def mean(X, trimming=0):
    """
    Column-wise mean or trimmed mean. Trimming to be entered as fraction.
    """

    if trimming == 0:
        if np.isnan(X).any():
            m = np.nanmean(X, axis=0)
        else:
            m = np.mean(X, axis=0)
    else:
        # Returns all NaN if missings in X
        m = sps.trim_mean(X, trimming, 0)

    return m


def std(X, trimming=0):
    """
    Column-wise standard devaition or trimmed std.
    Trimming to be entered as fraction.
    """

    if trimming == 0:
        if np.isnan(X).any():
            s = np.power(np.nanvar(X, axis=0), 0.5)
        else:
            s = np.power(np.var(X, axis=0), 0.5)
        s = np.array(s).reshape(-1)
    else:
        var = sps.trim_mean(
            np.square(X - sps.trim_mean(X, trimming, 0)), trimming, 0
        )
        s = np.sqrt(var)
    return s


def _euclidnorm(x):
    """
    Euclidean norm of a vector
    """

    if np.isnan(x).any():
        return np.sqrt(np.nansum(np.square(x)))
    else:
        return np.sqrt(np.sum(np.square(x)))


def _diffmat_objective(a, X):
    """
    Utility to l1median, matrix of differences
    """

    (n, p) = X.shape
    return X - np.tile(a, (n, 1))


def _l1m_objective(a, X, *args):
    """
    Optimization objective for l1median
    """

    if np.isnan(X).any():
        return np.nansum(
            np.apply_along_axis(_euclidnorm, 1, _diffmat_objective(a, X))
        )
    else:
        return np.sum(
            np.apply_along_axis(_euclidnorm, 1, _diffmat_objective(a, X))
        )


def _l1m_jacobian(a, X):
    """
    Jacobian for l1median
    """

    (n, p) = X.shape
    dX = _diffmat_objective(a, X)
    dists = np.apply_along_axis(_euclidnorm, 1, dX)
    dists = _handle_zeros_in_scale(dists)
    dX /= np.tile(np.array(dists).reshape(n, 1), (1, p))
    if np.isnan(X).any():
        return -np.nansum(dX, axis=0)
    else:
        return -np.sum(dX, axis=0)


def _l1median(
    X, x0, method="SLSQP", tol=1e-8, options={"maxiter": 2000}, **kwargs
):
    """
    Optimization for l1median
    """

    mu = spo.minimize(
        _l1m_objective,
        x0,
        args=(X),
        jac=_l1m_jacobian,
        tol=tol,
        options=options,
        method=method,
    )
    return mu


def l1median(X, **kwargs):
    """
    l1median wrapper to generically convert matrices as some of the scipy
    optimization options will crash when provided matrix input.
    """

    if "x0" not in kwargs:
        x0 = median(X)

    if type(X) == np.matrix:
        X = np.array(X)

    if len(X.shape) == 2:
        (n, p) = X.shape
    else:
        p = 1

    if p < 2:
        return median(X)
    else:
        return _l1median(X, x0, **kwargs).x


def kstepLTS(X, maxit=5, tol=1e-10, **kwargs):
    """
    Computes the K-step LTS estimator of location
    It uses the spatial median as a starting value, and yields an
    estimator with improved statistical efficiency, but at a higher
    computational cost.
    Inputs:
        X: data matrix
        maxit: maximum number of iterations
        tol: convergence tolerance
    Outputs:
        m2: location estimate
    """
    n, p = X.shape
    m1 = l1median(X)  # initial estimate
    m2 = copy.deepcopy(m1)
    iteration = 0
    unconverged = True
    while unconverged and (iteration < maxit):
        if np.isnan(X).any():
            dists = np.nansum(np.square(X - m1), axis=1)
        else:
            dists = np.sum(np.square(X - m1), axis=1)
        cutdist = np.sort(dists, axis=0)[int(np.floor((n + 1) / 2)) - 1]
        hsubset = np.where(dists <= cutdist)[0]
        m2 = np.array(mean(X[hsubset, :])).reshape((p,))
        unconverged = max(abs(m1 - m2)) > tol
        iteration += 1
        m1 = copy.deepcopy(m2)

    return m2


def scaleTau2(x0, c1=4.5, c2=3, consistency=True, **kwargs):
    """
    Tau estimator of scale
    Inputs:
        x0: array or matrix, data
        c1: consistency factor for initial estimate
        c2: consistency factor for final estimate
        consistency: str or bool,
            False, True, or "finiteSample"
    Output:
        the scale estimate
    """

    x = copy.deepcopy(x0)
    n, p = x.shape
    if np.isnan(x).any():
        summ = np.nansum
    else:
        summ = np.sum
    medx = median(x)
    xc = abs(x - medx)
    sigma0 = median(xc)
    if c1 > 0:
        xc /= sigma0 * c1
        w = 1 - np.square(xc)
        w = np.square((abs(w) + w) / 2)
        mu = summ(np.multiply(x, w)) / summ(w)
    else:
        mu = medx
    x -= mu
    x /= sigma0
    rho = np.square(x)
    rho[np.where(rho > c2**2)[0]] = c2**2
    if consistency:

        def Erho(b):
            return (
                2 * ((1 - b**2) * sps.norm.cdf(b) - b * sps.norm.pdf(b) + b**2)
                - 1
            )

        def Es2(c2):
            return Erho(c2 * sps.norm.ppf(3 / 4))

        if consistency == "finiteSample":
            nEs2 = (n - 2) * Es2(c2)
        else:
            nEs2 = n * Es2(c2)
    else:
        nEs2 = n
    return np.array(sigma0 * np.sqrt(summ(rho) / nEs2)).reshape((p,))


def scale_data(X, m, s):
    """
    Column-wise data scaling on location and scale estimates.

    """

    n = X.shape
    if len(n) > 1:
        p = n[1]
    else:
        p = 1
    n = n[0]

    s = _handle_zeros_in_scale(s)

    if p == 1:
        Xm = X - float(m)
        Xs = Xm / s
    else:
        Xm = X - np.array([m for i in range(1, n + 1)])
        Xs = Xm / np.array([s for i in range(1, n + 1)])
    return Xs
