# -*- coding: utf-8 -*-
"""
Unit tests for the rtb (Robust Twoblock) class.

@author: Sven Serneels
"""

import unittest
import numpy as np
import pandas as ps
from sklearn.metrics import r2_score
from .rtb import rtb


class TestRTB(unittest.TestCase):
    """Test methods in the rtb class"""

    @classmethod
    def setUpClass(cls):
        print("...setupClass TestRTB")

    @classmethod
    def tearDownClass(cls):
        print("...teardownClass TestRTB")

    def setUp(self):
        self.Yt = ps.read_csv("./data/cookie_lab_train.csv", index_col=0).T
        self.Xt = ps.read_csv("./data/cookie_nir_train.csv", index_col=0).T
        self.Yv = ps.read_csv("./data/cookie_lab_test.csv", index_col=0).T
        self.Xv = ps.read_csv("./data/cookie_nir_test.csv", index_col=0).T
        self.p = self.Xt.shape[1]
        self.q = self.Yt.shape[1]
        self.n = self.Xt.shape[0]

    def tearDown(self):
        del self.Xt, self.Yt, self.Xv, self.Yv, self.p, self.q, self.n

    def test_fit_dense_hampel(self):
        """Tests dense fit with default Hampel weighting"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        self.assertEqual(r.x_weights_.shape, (self.p, 5))
        self.assertEqual(r.y_weights_.shape, (self.q, 2))
        self.assertEqual(r.x_scores_.shape, (self.n, 5))
        self.assertEqual(r.y_scores_.shape, (self.n, 2))
        self.assertEqual(r.x_loadings_.shape, (self.p, 5))
        self.assertEqual(r.y_loadings_.shape, (self.q, 2))
        self.assertEqual(r.caseweights_.shape, (self.n,))
        self.assertEqual(r.x_caseweights_.shape, (self.n,))
        self.assertEqual(r.y_caseweights_.shape, (self.n,))
        self.assertEqual(r.fitted_.shape, (self.n, self.q))
        self.assertEqual(r.residuals_.shape, (self.n, self.q))

    def test_predict(self):
        """Tests predict on new data"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)

        ypv = r.predict(self.Xv)
        self.assertEqual(ypv.shape, self.Yv.shape)

        r2 = [r2_score(self.Yv.iloc[:, i], ypv[:, i]) for i in range(self.q)]
        for i, val in enumerate(r2):
            self.assertGreater(val, 0.0, f"R2 for Y variable {i} should be positive")

    def test_predict_wrong_columns(self):
        """Tests predict raises on wrong number of columns"""
        r = rtb(n_components_x=3, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)
        with self.assertRaises(ValueError):
            r.predict(self.Xv.iloc[:, :10])

    def test_fit_dense_fair(self):
        """Tests dense fit with Fair weighting"""
        r = rtb(n_components_x=5, n_components_y=2, fun="Fair", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        ypv = r.predict(self.Xv)
        self.assertEqual(ypv.shape, self.Yv.shape)

    def test_fit_dense_huber(self):
        """Tests dense fit with Huber weighting"""
        r = rtb(n_components_x=5, n_components_y=2, fun="Huber", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        ypv = r.predict(self.Xv)
        self.assertEqual(ypv.shape, self.Yv.shape)

    def test_fit_sparse(self):
        """Tests sparse fit with sparsity in X block"""
        r = rtb(
            n_components_x=5, n_components_y=2,
            sparse=True, eta_x=0.7, eta_y=0,
            verbose=False
        )
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        self.assertTrue(hasattr(r, "x_indret_"))
        self.assertTrue(hasattr(r, "y_indret_"))
        self.assertTrue(hasattr(r, "x_colret_"))
        self.assertTrue(hasattr(r, "y_colret_"))

        ypv = r.predict(self.Xv)
        self.assertEqual(ypv.shape, self.Yv.shape)

    def test_transform(self):
        """Tests transform returns correct shape"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)

        T = r.transform(self.Xv)
        self.assertEqual(T.shape, (self.Xv.shape[0], 5))

    def test_transform_wrong_columns(self):
        """Tests transform raises on wrong number of columns"""
        r = rtb(n_components_x=3, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)
        with self.assertRaises(ValueError):
            r.transform(self.Xv.iloc[:, :10])

    def test_weightnewx(self):
        """Tests weightnewx returns correct shape and range"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)

        wts = r.weightnewx(self.Xv)
        self.assertEqual(wts.shape, (self.Xv.shape[0],))
        self.assertTrue(np.all(wts >= 0), "Weights should be non-negative")
        self.assertTrue(np.all(wts <= 1), "Weights should be at most 1")

    def test_weightnewx_wrong_columns(self):
        """Tests weightnewx raises on wrong number of columns"""
        r = rtb(n_components_x=3, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)
        with self.assertRaises(ValueError):
            r.weightnewx(self.Xv.iloc[:, :10])

    def test_numpy_input(self):
        """Tests fit/predict with numpy array input"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt.to_numpy(), self.Yt.to_numpy())

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        ypv = r.predict(self.Xv.to_numpy())
        self.assertEqual(ypv.shape, (self.Xv.shape[0], self.q))

    def test_n_components_y_default(self):
        """Tests that n_components_y defaults to n_components_x"""
        r = rtb(n_components_x=3, verbose=False)
        r.fit(self.Xt, self.Yt)
        self.assertEqual(r.n_components_y, 3)

    def test_invalid_fun(self):
        """Tests that invalid fun raises ValueError"""
        r = rtb(n_components_x=3, fun="InvalidFun", verbose=False)
        with self.assertRaises(ValueError):
            r.fit(self.Xt, self.Yt)

    def test_invalid_probp1(self):
        """Tests that invalid probp1 raises ValueError"""
        r = rtb(n_components_x=3, probp1=1.5, verbose=False)
        with self.assertRaises(ValueError):
            r.fit(self.Xt, self.Yt)

    def test_invalid_hampel_params(self):
        """Tests that invalid Hampel parameter ordering raises ValueError"""
        r = rtb(n_components_x=3, probp1=0.95, probp2=0.90, probp3=0.999,
                verbose=False)
        with self.assertRaises(ValueError):
            r.fit(self.Xt, self.Yt)

    def test_mismatched_xy(self):
        """Tests that mismatched X/Y rows raises ValueError"""
        r = rtb(n_components_x=3, verbose=False)
        with self.assertRaises(ValueError):
            r.fit(self.Xt.iloc[:10, :], self.Yt)

    def test_start_X_init_raw(self):
        """Tests fit with start_X_init != 'pcapp'"""
        r = rtb(n_components_x=5, n_components_y=2,
                start_X_init="raw", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        ypv = r.predict(self.Xv)
        self.assertEqual(ypv.shape, self.Yv.shape)

    def test_start_cutoff_mode_nonspecific(self):
        """Tests fit with start_cutoff_mode != 'specific'"""
        r = rtb(n_components_x=5, n_components_y=2,
                start_cutoff_mode="identical", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))

    def test_scale_none(self):
        """Tests fit with scale='None' triggers rescale path"""
        r = rtb(n_components_x=5, n_components_y=2,
                scale="None", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))

    def test_centre_mean(self):
        """Tests fit with centre='mean' intercept path"""
        r = rtb(n_components_x=5, n_components_y=2,
                centre="mean", scale="std", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        self.assertTrue(np.all(np.isfinite(r.intercept_)))

    def test_centre_none_scale_none(self):
        """Tests fit with centre='None' and scale='None'"""
        r = rtb(n_components_x=5, n_components_y=2,
                centre="None", scale="None", verbose=False)
        r.fit(self.Xt, self.Yt)

        self.assertEqual(r.coef_.shape, (self.p, self.q))
        self.assertEqual(r.intercept_, 0)

    def test_copy_stores_data(self):
        """Tests that copy=True stores X and Y"""
        r = rtb(n_components_x=3, n_components_y=2, copy=True, verbose=False)
        r.fit(self.Xt, self.Yt)
        self.assertTrue(hasattr(r, "X"))
        self.assertTrue(hasattr(r, "Y"))
        self.assertEqual(r.X.shape[0], self.n)
        self.assertEqual(r.Y.shape[0], self.n)

    def test_caseweights_product(self):
        """Tests that combined caseweights = x_caseweights * y_caseweights"""
        r = rtb(n_components_x=5, n_components_y=2, verbose=False)
        r.fit(self.Xt, self.Yt)

        np.testing.assert_array_almost_equal(
            r.caseweights_, r.x_caseweights_ * r.y_caseweights_
        )


if __name__ == "__main__":
    unittest.main()
