#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:08:22 2020

@author: sven
"""

import pandas as ps
import numpy as np


def convert_X_input(X):

    if type(X) == ps.core.frame.DataFrame:
        X = X.to_numpy().astype("float64")
    return X


def convert_y_input(y):

    if type(y) in [ps.core.frame.DataFrame, ps.core.series.Series]:
        y = y.to_numpy().T.astype("float64")
    return y


def const_xscale(beta, *args):
    X = args[0]
    h = args[1]
    i = args[2]
    j = args[3]
    beta = np.reshape(beta, (-1, h), order="F")
    covx = np.cov(X, rowvar=False)
    ans = np.matmul(np.matmul(beta.T, covx), beta) - np.identity(h)
    return ans[i, j]


def const_zscale(beta, *args):
    X = args[0]
    h = args[1]
    i = args[2]
    j = args[3]
    beta = np.reshape(beta, (-1, h), order="F")
    covx = np.identity(X.shape[1])
    ans = np.matmul(np.matmul(beta.T, covx), beta) - np.identity(h)
    return ans[i, j]


def _predict_check_input(Xn):
    if type(Xn) == ps.core.series.Series:
        Xn = Xn.to_numpy()
    if Xn.ndim == 1:
        Xn = Xn.reshape((1, -1))
    if type(Xn) == ps.core.frame.DataFrame:
        Xn = Xn.to_numpy()
    n, p = Xn.shape
    return (n, p, Xn)


def _check_input(X):

    if type(X) in (np.matrix, ps.core.frame.DataFrame, ps.core.series.Series):
        X = np.array(X)

    if X.dtype == np.dtype("O"):
        X = X.astype("float64")

    if X.ndim == 1:
        X = X.reshape((1, -1))

    n, p = X.shape

    if n == 1:
        if p >= 2:
            X = X.reshape((-1, 1))
    return X


def Fair(x, probct, *args):
    return 1 / (1 + abs(x / (probct * 2))) ** 2


def Huber(x, probct, *args):
    x[np.where(x <= probct)[0]] = 1
    x[np.where(x > probct)] = probct / abs(x[np.where(x > probct)])
    return x


def Hampel(x, probct, hampelb, hampelr):
    wx = x.copy()
    wx[np.where(x <= probct)[0]] = 1
    wx[np.where((x > probct) & (x <= hampelb))[0]] = probct / abs(
        x[np.where((x > probct) & (x <= hampelb))[0]]
    )
    wx[np.where((x > hampelb) & (x <= hampelr))[0]] = np.divide(
        probct * (hampelr - x[np.where((x > hampelb) & (x <= hampelr))[0]]),
        (hampelr - hampelb) * abs(x[np.where((x > hampelb) & (x <= hampelr))[0]]),
    )
    wx[np.where(x > hampelr)[0]] = 0
    return wx


def brokenstick(n_components):
    q = np.triu(np.ones((n_components, n_components)))
    r = np.empty((n_components, 1), float)
    r[0:n_components, 0] = range(1, n_components + 1)
    q = np.matmul(q, 1 / r)
    q /= n_components
    return q
