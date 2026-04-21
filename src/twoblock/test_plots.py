# -*- coding: utf-8 -*-
"""
Smoke tests for :mod:`twoblock.plots`.

Each builder is tested on small synthetic arrays — the goal is only to
confirm the function runs and returns a ``plotly.graph_objects.Figure``.
Visual correctness is validated by the paper-figure notebook scripts.
"""

import unittest
import numpy as np

try:
    import plotly.graph_objects as go
    from twoblock import plots
    _HAVE_PLOTLY = True
except ImportError:  # pragma: no cover - exercised only when extra is missing
    _HAVE_PLOTLY = False


@unittest.skipUnless(_HAVE_PLOTLY, "plotly not installed (install with 'pip install twoblock[plots]')")
class TestPlots(unittest.TestCase):
    """Each builder returns a plotly Figure on synthetic inputs."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.n, cls.p, cls.q, cls.k = 40, 8, 2, 3
        cls.scores = rng.normal(size=(cls.n, cls.k))
        cls.loadings = rng.normal(size=(cls.p, cls.k))
        cls.expvar = np.array([0.6, 0.25, 0.1])
        cls.coef_2d = rng.normal(size=(cls.p, cls.q))
        cls.coef_1d = rng.normal(size=cls.p)
        cls.y_true = rng.normal(size=(cls.n, cls.q))
        cls.y_pred = cls.y_true + 0.1 * rng.normal(size=(cls.n, cls.q))
        cls.case_w = rng.uniform(size=cls.n)
        cls.cell_w = rng.uniform(size=(cls.n, cls.p))
        cls.contrib = rng.uniform(size=cls.p)

    def test_scree(self):
        fig = plots.scree(self.expvar)
        self.assertIsInstance(fig, go.Figure)

    def test_score_scatter_plain(self):
        fig = plots.score_scatter(self.scores)
        self.assertIsInstance(fig, go.Figure)

    def test_score_scatter_with_weights(self):
        fig = plots.score_scatter(self.scores, case_weights=self.case_w)
        self.assertIsInstance(fig, go.Figure)

    def test_loadings_bar(self):
        fig = plots.loadings_bar(self.loadings, component=1,
                                 feature_names=[f"v{i}" for i in range(self.p)])
        self.assertIsInstance(fig, go.Figure)

    def test_coefficients_bar_2d(self):
        fig = plots.coefficients_bar(self.coef_2d,
                                     response_names=["Y1", "Y2"])
        self.assertIsInstance(fig, go.Figure)

    def test_coefficients_bar_1d(self):
        fig = plots.coefficients_bar(self.coef_1d)
        self.assertIsInstance(fig, go.Figure)

    def test_y_pred_vs_obs(self):
        fig = plots.y_pred_vs_obs(self.y_true, self.y_pred,
                                  response_names=["CO", "NOx"])
        self.assertIsInstance(fig, go.Figure)

    def test_caseweight_hist(self):
        fig = plots.caseweight_hist(self.case_w)
        self.assertIsInstance(fig, go.Figure)

    def test_cellweight_heatmap(self):
        fig = plots.cellweight_heatmap(self.cell_w)
        self.assertIsInstance(fig, go.Figure)

    def test_spadimo_contributions(self):
        fig = plots.spadimo_contributions(self.contrib,
                                          flagged_indices=[0, 3, 5])
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
