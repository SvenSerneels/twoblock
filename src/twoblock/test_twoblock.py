# -*- coding: utf-8 -*-
"""
Created on Wed Aug 7 18:17:46 2020

@author: Sven Serneels
"""

import unittest
from .twoblock import twoblock
import pandas as ps
import numpy as np
from sklearn.metrics import r2_score


class TestTwoBlock(unittest.TestCase):
    """Test  methods in the twoblock class"""

    @classmethod
    def setUpClass(cls):
        print("...setupClass")

    @classmethod
    def tearDownClass(cls):
        print("...teardownClass")

    @classmethod
    def setUp(self):
        self.Yt = ps.read_csv("./data/cookie_lab_train.csv", index_col=0).T
        self.Xt = ps.read_csv("./data/cookie_nir_train.csv", index_col=0).T
        self.Yv = ps.read_csv("./data/cookie_lab_test.csv", index_col=0).T
        self.Xv = ps.read_csv("./data/cookie_nir_test.csv", index_col=0).T
        self.p = self.Xt.shape[1]
        self.q = self.Yt.shape[1]

    @classmethod
    def tearDown(self):
        del self.Xt
        del self.Yt
        del self.Xv
        del self.Yv
        del self.p
        del self.q

    def test_assert(self):

        tb = twoblock(n_components_x=7, n_components_y=200, scale="None")
        self.assertRaises(AssertionError, tb.fit, self.Xt, self.Yt)

        tb = twoblock(n_components_x=700, n_components_y=2, scale="None")
        self.assertRaises(AssertionError, tb.fit, self.Xt, self.Yt)

    def test_fit(self):
        """Tests fit function"""

        tb = twoblock(n_components_x=7, n_components_y=2, scale="None")
        tb.fit(self.Xt, self.Yt)

        self.assertEqual(tb.coef_.shape, (self.p, self.q))  # coefficients

        ypttb = tb.predict(self.Xv)

        self.assertEqual(
            ypttb.shape,
            self.Yv.shape,
        )  # predictions

        r2tbt = [
            r2_score(self.Yv.iloc[:, i], ypttb[:, i]) for i in range(self.q)
        ]

        self.assertGreaterEqual(r2tbt, [0.8 for i in range(self.q)])


if __name__ == "__main__":
    unittest.main()
