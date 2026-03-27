# -*- coding: utf-8 -*-
"""
GPU tests for twoblock and rtb classes.
Skipped automatically when CuPy is not available.
"""

import unittest
import numpy as np
import pandas as ps
import pytest

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestTwoBlockGPU(unittest.TestCase):
    """Test twoblock with gpu=True and compare to CPU results."""

    @classmethod
    def setUpClass(cls):
        cls.Yt = ps.read_csv("./data/cookie_lab_train.csv", index_col=0).T
        cls.Xt = ps.read_csv("./data/cookie_nir_train.csv", index_col=0).T
        cls.Yv = ps.read_csv("./data/cookie_lab_test.csv", index_col=0).T
        cls.Xv = ps.read_csv("./data/cookie_nir_test.csv", index_col=0).T
        cls.p = cls.Xt.shape[1]
        cls.q = cls.Yt.shape[1]

    def test_fit_dense_gpu(self):
        """GPU dense fit produces numpy outputs matching CPU."""
        from .twoblock import twoblock

        tb_gpu = twoblock(n_components_x=7, n_components_y=2, scale="None", gpu=True)
        tb_gpu.fit(self.Xt, self.Yt)

        tb_cpu = twoblock(n_components_x=7, n_components_y=2, scale="None", gpu=False)
        tb_cpu.fit(self.Xt, self.Yt)

        # All fitted attributes should be numpy arrays
        self.assertIsInstance(tb_gpu.coef_, np.ndarray)
        self.assertIsInstance(tb_gpu.x_weights_, np.ndarray)
        self.assertIsInstance(tb_gpu.x_scores_, np.ndarray)

        # Results should match CPU within tolerance
        np.testing.assert_allclose(tb_gpu.coef_, tb_cpu.coef_, rtol=1e-5)
        np.testing.assert_allclose(tb_gpu.x_weights_, tb_cpu.x_weights_, rtol=1e-5)

    def test_fit_dense_gpu_with_scaling(self):
        """GPU dense fit with centering and scaling."""
        from .twoblock import twoblock

        tb_gpu = twoblock(n_components_x=7, n_components_y=2, scale="std", gpu=True)
        tb_gpu.fit(self.Xt, self.Yt)

        tb_cpu = twoblock(n_components_x=7, n_components_y=2, scale="std", gpu=False)
        tb_cpu.fit(self.Xt, self.Yt)

        self.assertIsInstance(tb_gpu.coef_, np.ndarray)
        np.testing.assert_allclose(tb_gpu.coef_, tb_cpu.coef_, rtol=1e-5)

    def test_fit_sparse_gpu(self):
        """GPU sparse fit produces numpy outputs matching CPU."""
        from .twoblock import twoblock

        tb_gpu = twoblock(
            n_components_x=7, n_components_y=2,
            eta_x=.8, scale="std", sparse=True, gpu=True
        )
        tb_gpu.fit(self.Xt, self.Yt)

        tb_cpu = twoblock(
            n_components_x=7, n_components_y=2,
            eta_x=.8, scale="std", sparse=True, gpu=False
        )
        tb_cpu.fit(self.Xt, self.Yt)

        self.assertIsInstance(tb_gpu.coef_, np.ndarray)
        self.assertIsInstance(tb_gpu.x_indret_, np.ndarray)
        np.testing.assert_allclose(tb_gpu.coef_, tb_cpu.coef_, rtol=1e-5)
        np.testing.assert_array_equal(tb_gpu.x_indret_, tb_cpu.x_indret_)

    def test_predict_gpu(self):
        """Predict works after GPU fit (uses numpy internally)."""
        from .twoblock import twoblock

        tb = twoblock(n_components_x=7, n_components_y=2, scale="None", gpu=True)
        tb.fit(self.Xt, self.Yt)
        Yp = tb.predict(self.Xv)

        self.assertIsInstance(Yp, np.ndarray)
        self.assertEqual(Yp.shape, self.Yv.shape)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestRTBGPU(unittest.TestCase):
    """Test rtb with gpu=True and compare to CPU results."""

    @classmethod
    def setUpClass(cls):
        cls.Yt = ps.read_csv("./data/cookie_lab_train.csv", index_col=0).T
        cls.Xt = ps.read_csv("./data/cookie_nir_train.csv", index_col=0).T
        cls.Yv = ps.read_csv("./data/cookie_lab_test.csv", index_col=0).T
        cls.Xv = ps.read_csv("./data/cookie_nir_test.csv", index_col=0).T

    def test_rtb_dense_gpu(self):
        """GPU RTB dense fit produces numpy outputs matching CPU."""
        from .rtb import rtb

        r_gpu = rtb(n_components_x=5, n_components_y=2, verbose=False, gpu=True)
        r_gpu.fit(self.Xt, self.Yt)

        r_cpu = rtb(n_components_x=5, n_components_y=2, verbose=False, gpu=False)
        r_cpu.fit(self.Xt, self.Yt)

        self.assertIsInstance(r_gpu.coef_, np.ndarray)
        self.assertIsInstance(r_gpu.caseweights_, np.ndarray)
        np.testing.assert_allclose(r_gpu.coef_, r_cpu.coef_, rtol=1e-4)
        np.testing.assert_allclose(
            r_gpu.caseweights_, r_cpu.caseweights_, rtol=1e-4
        )

    def test_rtb_sparse_gpu(self):
        """GPU RTB sparse fit produces numpy outputs matching CPU."""
        from .rtb import rtb

        r_gpu = rtb(
            n_components_x=5, n_components_y=2,
            sparse=True, eta_x=0.5, eta_y=0,
            verbose=False, gpu=True
        )
        r_gpu.fit(self.Xt, self.Yt)

        r_cpu = rtb(
            n_components_x=5, n_components_y=2,
            sparse=True, eta_x=0.5, eta_y=0,
            verbose=False, gpu=False
        )
        r_cpu.fit(self.Xt, self.Yt)

        self.assertIsInstance(r_gpu.coef_, np.ndarray)
        np.testing.assert_allclose(r_gpu.coef_, r_cpu.coef_, rtol=1e-4)

    def test_rtb_predict_gpu(self):
        """Predict works after GPU RTB fit."""
        from .rtb import rtb

        r = rtb(n_components_x=5, n_components_y=2, verbose=False, gpu=True)
        r.fit(self.Xt, self.Yt)
        Yp = r.predict(self.Xv)

        self.assertIsInstance(Yp, np.ndarray)
        self.assertEqual(Yp.shape, self.Yv.shape)


if __name__ == "__main__":
    unittest.main()
