# -*- coding: utf-8 -*-
"""
Plotly builders for twoblock-family estimators.

Array-first API: each function accepts plain numpy arrays extracted from a
fitted estimator (e.g. ``est.x_expvar_``, ``est.x_scores_``) and returns a
``plotly.graph_objects.Figure``. Keeping the plots decoupled from estimator
internals makes them reusable by sklearn's ``PLSRegression`` as well as by
the paper-figure notebook scripts.

Install with ``pip install twoblock[plots]``.
"""

import numpy as np
import plotly.graph_objects as go


_DEFAULT_HEIGHT = 420


def _as_1d(a):
    a = np.asarray(a)
    return a.ravel()


def _as_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _feature_labels(feature_names, p, prefix="X"):
    if feature_names is None:
        return [f"{prefix}{i}" for i in range(p)]
    return [str(n) for n in feature_names]


def _response_labels(response_names, q, prefix="Y"):
    if response_names is None:
        return [f"{prefix}{i}" for i in range(q)]
    return [str(n) for n in response_names]


def scree(expvar, title=None, block="x"):
    """Bar plot of explained variance per component.

    Parameters
    ----------
    expvar : array-like, shape (k,)
        Explained variance per component (e.g. ``est.x_expvar_``).
    title : str, optional
    block : {"x", "y"}
        Only used to label the default title.
    """
    expvar = _as_1d(expvar)
    comps = [f"comp {i + 1}" for i in range(len(expvar))]
    fig = go.Figure(go.Bar(x=comps, y=expvar, marker_color="steelblue"))
    fig.update_layout(
        title=title or f"Scree ({block.upper()} block)",
        xaxis_title="component",
        yaxis_title="explained variance",
        height=_DEFAULT_HEIGHT,
        template="simple_white",
    )
    return fig


def score_scatter(scores, comp_x=0, comp_y=1, case_weights=None,
                  labels=None, title=None):
    """2D scatter of latent scores, optionally colored by case weights.

    Parameters
    ----------
    scores : array-like, shape (n, k)
    comp_x, comp_y : int
        Component indices (0-based).
    case_weights : array-like of shape (n,), optional
        Values in [0, 1]; weights near 0 signal outliers. Colored via a
        red-to-steelblue scale when provided.
    labels : array-like of shape (n,), optional
        Per-observation hover text.
    """
    scores = np.asarray(scores)
    x = scores[:, comp_x]
    y = scores[:, comp_y]

    marker = dict(size=8, line=dict(width=0.5, color="white"))
    if case_weights is not None:
        w = _as_1d(case_weights)
        marker.update(
            color=w,
            colorscale="RdBu",
            cmin=0.0,
            cmax=1.0,
            showscale=True,
            colorbar=dict(title="case weight"),
        )
    else:
        marker["color"] = "steelblue"

    hover = labels if labels is not None else np.arange(len(x))
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", marker=marker,
        text=[str(h) for h in hover],
        hovertemplate="%{text}<br>t%{xaxis.title.text}=%{x:.3f}"
                      "<br>t%{yaxis.title.text}=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title or f"Scores t{comp_x + 1} vs t{comp_y + 1}",
        xaxis_title=f"t{comp_x + 1}",
        yaxis_title=f"t{comp_y + 1}",
        height=_DEFAULT_HEIGHT,
        template="simple_white",
    )
    return fig


def loadings_bar(loadings, component=0, feature_names=None, title=None):
    """Horizontal bar chart of loadings for one component.

    Parameters
    ----------
    loadings : array-like, shape (p, k)
    component : int
        Which component column to plot.
    feature_names : sequence of str, optional
    """
    loadings = _as_2d(loadings)
    v = loadings[:, component]
    names = _feature_labels(feature_names, len(v), prefix="X")

    order = np.argsort(np.abs(v))
    v_sorted = v[order]
    names_sorted = [names[i] for i in order]

    fig = go.Figure(go.Bar(
        x=v_sorted, y=names_sorted, orientation="h",
        marker_color=["crimson" if val < 0 else "steelblue" for val in v_sorted],
    ))
    fig.update_layout(
        title=title or f"Loadings (component {component + 1})",
        xaxis_title="loading",
        yaxis_title="variable",
        height=max(_DEFAULT_HEIGHT, 18 * len(v) + 80),
        template="simple_white",
    )
    return fig


def coefficients_bar(coef, feature_names=None, response_names=None, title=None):
    """Horizontal bar chart of regression coefficients.

    Accepts 1D ``(p,)`` (univariate response, e.g. ``crm``) or 2D ``(p, q)``
    (multivariate response, twoblock convention). sklearn's
    ``PLSRegression.coef_`` has shape ``(q, p)`` — transpose before passing.
    """
    coef = _as_2d(coef)
    p, q = coef.shape
    fnames = _feature_labels(feature_names, p, prefix="X")
    rnames = _response_labels(response_names, q, prefix="Y")

    fig = go.Figure()
    for j in range(q):
        fig.add_trace(go.Bar(
            x=coef[:, j], y=fnames, orientation="h", name=rnames[j],
        ))
    fig.update_layout(
        barmode="group",
        title=title or "Regression coefficients",
        xaxis_title="coefficient",
        yaxis_title="variable",
        height=max(_DEFAULT_HEIGHT, 18 * p + 80),
        template="simple_white",
        showlegend=q > 1,
    )
    return fig


def y_pred_vs_obs(y_true, y_pred, response_names=None, title=None):
    """Predicted vs observed scatter per response, with the y=x reference line."""
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    q = y_true.shape[1]
    rnames = _response_labels(response_names, q, prefix="Y")

    fig = go.Figure()
    for j in range(q):
        fig.add_trace(go.Scatter(
            x=y_true[:, j], y=y_pred[:, j], mode="markers",
            name=rnames[j], marker=dict(size=7, opacity=0.7),
        ))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
        line=dict(color="grey", dash="dash"), showlegend=False,
    ))
    fig.update_layout(
        title=title or "Predicted vs observed",
        xaxis_title="observed",
        yaxis_title="predicted",
        height=_DEFAULT_HEIGHT,
        template="simple_white",
    )
    return fig


def caseweight_hist(case_weights, title=None):
    """Histogram of case weights. Values near 0 signal row-wise outliers."""
    w = _as_1d(case_weights)
    fig = go.Figure(go.Histogram(x=w, nbinsx=30, marker_color="steelblue"))
    fig.update_layout(
        title=title or "Case-weight distribution",
        xaxis_title="case weight",
        yaxis_title="count",
        xaxis=dict(range=[0, 1.02]),
        height=_DEFAULT_HEIGHT,
        template="simple_white",
    )
    return fig


def cellweight_heatmap(cellweights, feature_names=None, sample_labels=None,
                       title=None):
    """Heatmap of cellwise weights. Cells near 0 are flagged cellwise outliers.

    Parameters
    ----------
    cellweights : array-like, shape (n, p)
    """
    W = np.asarray(cellweights)
    n, p = W.shape
    cols = _feature_labels(feature_names, p, prefix="X")
    rows = (list(map(str, sample_labels)) if sample_labels is not None
            else [str(i) for i in range(n)])

    fig = go.Figure(go.Heatmap(
        z=W, x=cols, y=rows,
        colorscale="RdBu", zmin=0.0, zmax=1.0,
        colorbar=dict(title="cell weight"),
    ))
    fig.update_layout(
        title=title or "Cellwise weights",
        xaxis_title="variable",
        yaxis_title="observation",
        height=max(_DEFAULT_HEIGHT, min(900, 16 * n + 100)),
        template="simple_white",
    )
    return fig


def spadimo_contributions(contributions, feature_names=None,
                          flagged_indices=None, title=None):
    """Horizontal bar of per-variable contributions to an observation's outlyingness.

    ``flagged_indices`` (from ``spadimo.outlvars_``) are drawn in crimson;
    non-flagged variables in steelblue.
    """
    c = _as_1d(contributions)
    names = _feature_labels(feature_names, len(c), prefix="X")

    flagged = set(int(i) for i in flagged_indices) if flagged_indices is not None else set()
    colors = ["crimson" if i in flagged else "steelblue" for i in range(len(c))]

    order = np.argsort(c)
    c_sorted = c[order]
    names_sorted = [names[i] for i in order]
    colors_sorted = [colors[i] for i in order]

    fig = go.Figure(go.Bar(
        x=c_sorted, y=names_sorted, orientation="h",
        marker_color=colors_sorted,
    ))
    fig.update_layout(
        title=title or "SPADIMO: per-variable contributions",
        xaxis_title="contribution",
        yaxis_title="variable",
        height=max(_DEFAULT_HEIGHT, 18 * len(c) + 80),
        template="simple_white",
    )
    return fig
