# -*- coding: utf-8 -*-
"""
Tests for utils.py, _preproc_utilities.py, and prepro.py
to bring coverage above 80%.
"""

import unittest
import numpy as np
import pandas as ps

from .utils import (
    convert_X_input, convert_y_input,
    const_xscale, const_zscale,
    _predict_check_input, _check_input,
    Fair, Huber, Hampel, brokenstick,
)
from ._preproc_utilities import (
    _handle_zeros_in_scale, _check_trimming,
    mad, median, mean, std,
    _euclidnorm, _diffmat_objective, _l1m_objective, _l1m_jacobian,
    l1median, kstepLTS, scaleTau2, scale_data,
)
from .prepro import VersatileScaler, versatile_scale


class TestUtils(unittest.TestCase):

    def test_convert_X_input_dataframe(self):
        df = ps.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        result = convert_X_input(df)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[1.0, 3.0], [2.0, 4.0]]))

    def test_convert_X_input_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = convert_X_input(arr)
        np.testing.assert_array_equal(result, arr)

    def test_convert_y_input_dataframe(self):
        df = ps.DataFrame({'a': [1.0, 2.0]})
        result = convert_y_input(df)
        self.assertIsInstance(result, np.ndarray)

    def test_convert_y_input_series(self):
        s = ps.Series([1.0, 2.0, 3.0])
        result = convert_y_input(s)
        self.assertIsInstance(result, np.ndarray)

    def test_convert_y_input_array(self):
        arr = np.array([1, 2, 3])
        result = convert_y_input(arr)
        np.testing.assert_array_equal(result, arr)

    def test_const_xscale(self):
        X = np.random.randn(50, 3)
        beta = np.eye(3, 2).ravel(order='F')
        result = const_xscale(beta, X, 2, 0, 0)
        self.assertIsInstance(result, float)

    def test_const_zscale(self):
        X = np.random.randn(50, 3)
        beta = np.eye(3, 2).ravel(order='F')
        result = const_zscale(beta, X, 2, 0, 1)
        self.assertIsInstance(result, float)

    def test_predict_check_input_series(self):
        s = ps.Series([1.0, 2.0, 3.0])
        n, p, Xn = _predict_check_input(s)
        self.assertEqual(n, 1)
        self.assertEqual(p, 3)

    def test_predict_check_input_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        n, p, Xn = _predict_check_input(arr)
        self.assertEqual(n, 1)
        self.assertEqual(p, 3)

    def test_predict_check_input_dataframe(self):
        df = ps.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        n, p, Xn = _predict_check_input(df)
        self.assertEqual(n, 2)
        self.assertEqual(p, 2)
        self.assertIsInstance(Xn, np.ndarray)

    def test_predict_check_input_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        n, p, Xn = _predict_check_input(arr)
        self.assertEqual(n, 2)
        self.assertEqual(p, 2)

    def test_check_input_dataframe(self):
        df = ps.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        result = _check_input(df)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))

    def test_check_input_object_dtype(self):
        arr = np.array([[1, 2], [3, 4]], dtype=object)
        result = _check_input(arr)
        self.assertEqual(result.dtype, np.float64)

    def test_check_input_1d_vector(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _check_input(arr)
        self.assertEqual(result.shape, (3, 1))

    def test_check_input_matrix(self):
        m = np.matrix([[1, 2], [3, 4]])
        result = _check_input(m)
        self.assertIsInstance(result, np.ndarray)

    def test_fair(self):
        x = np.array([0.5, 1.0, 2.0, 5.0])
        w = Fair(x, 1.0)
        self.assertEqual(len(w), 4)
        self.assertTrue(np.all(w > 0))
        self.assertTrue(np.all(w <= 1))

    def test_huber(self):
        x = np.array([0.5, 1.0, 2.0, 5.0])
        w = Huber(x.copy(), 1.5)
        self.assertEqual(w[0], 1.0)
        self.assertTrue(w[-1] < 1.0)

    def test_hampel(self):
        x = np.array([0.5, 1.5, 2.5, 4.0])
        w = Hampel(x, 1.0, 2.0, 3.0)
        self.assertEqual(w[0], 1.0)  # below probct
        self.assertTrue(w[1] < 1.0)  # between probct and hampelb
        self.assertTrue(w[2] < 1.0)  # between hampelb and hampelr
        self.assertEqual(w[3], 0.0)  # above hampelr

    def test_brokenstick(self):
        bs = brokenstick(5)
        self.assertEqual(bs.shape, (5, 1))
        self.assertTrue(bs[0, 0] > bs[-1, 0])


class TestPreprocUtilities(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 5)

    def test_handle_zeros_in_scale_scalar_zero(self):
        self.assertEqual(_handle_zeros_in_scale(0.0), 1.0)

    def test_handle_zeros_in_scale_scalar_nonzero(self):
        self.assertEqual(_handle_zeros_in_scale(2.5), 2.5)

    def test_handle_zeros_in_scale_array(self):
        s = np.array([1.0, 0.0, 3.0])
        result = _handle_zeros_in_scale(s)
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0])

    def test_handle_zeros_in_scale_array_no_copy(self):
        s = np.array([1.0, 0.0, 3.0])
        result = _handle_zeros_in_scale(s, copy=False)
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0])
        # in-place modification
        self.assertEqual(s[1], 1.0)

    def test_check_trimming_valid(self):
        _check_trimming(0.1)  # should not raise

    def test_check_trimming_invalid(self):
        with self.assertRaises(ValueError):
            _check_trimming(1.0)
        with self.assertRaises(ValueError):
            _check_trimming(-0.1)

    def test_mad(self):
        s = mad(self.X)
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_median(self):
        m = median(self.X)
        self.assertEqual(len(m), 5)

    def test_median_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        m = median(X)
        self.assertEqual(len(m), 5)
        self.assertFalse(np.isnan(m).any())

    def test_mean_no_trimming(self):
        m = mean(self.X)
        self.assertEqual(len(m), 5)

    def test_mean_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        m = mean(X)
        self.assertFalse(np.all(np.isnan(m)))

    def test_mean_trimmed(self):
        m = mean(self.X, trimming=0.1)
        self.assertEqual(len(m), 5)

    def test_std_no_trimming(self):
        s = std(self.X)
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_std_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        s = std(X)
        self.assertEqual(len(s), 5)

    def test_std_trimmed(self):
        s = std(self.X, trimming=0.1)
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_euclidnorm(self):
        x = np.array([3.0, 4.0])
        self.assertAlmostEqual(_euclidnorm(x), 5.0)

    def test_euclidnorm_with_nan(self):
        x = np.array([3.0, np.nan, 4.0])
        self.assertAlmostEqual(_euclidnorm(x), 5.0)

    def test_diffmat_objective(self):
        a = np.zeros(5)
        result = _diffmat_objective(a, self.X)
        np.testing.assert_array_equal(result, self.X)

    def test_l1m_objective(self):
        a = median(self.X)
        val = _l1m_objective(a, self.X)
        self.assertIsInstance(val, float)
        self.assertTrue(val > 0)

    def test_l1m_objective_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        a = median(X)
        val = _l1m_objective(a, X)
        self.assertTrue(val > 0)

    def test_l1m_jacobian(self):
        a = median(self.X)
        jac = _l1m_jacobian(a, self.X)
        self.assertEqual(len(jac), 5)

    def test_l1m_jacobian_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        a = median(X)
        jac = _l1m_jacobian(a, X)
        self.assertEqual(len(jac), 5)

    def test_l1median(self):
        m = l1median(self.X)
        self.assertEqual(len(m), 5)

    def test_l1median_1d(self):
        x = self.X[:, 0:1]
        m = l1median(x)
        self.assertEqual(len(m), 1)

    def test_l1median_matrix_input(self):
        m = l1median(np.matrix(self.X))
        self.assertEqual(len(m), 5)

    def test_kstepLTS(self):
        m = kstepLTS(self.X)
        self.assertEqual(len(m), 5)

    def test_kstepLTS_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        m = kstepLTS(X)
        self.assertEqual(len(m), 5)

    def test_scaleTau2(self):
        s = scaleTau2(self.X)
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_scaleTau2_no_consistency(self):
        s = scaleTau2(self.X, consistency=False)
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_scaleTau2_finiteSample(self):
        s = scaleTau2(self.X, consistency="finiteSample")
        self.assertEqual(len(s), 5)
        self.assertTrue(np.all(s > 0))

    def test_scaleTau2_c1_zero(self):
        s = scaleTau2(self.X, c1=0)
        self.assertEqual(len(s), 5)

    def test_scaleTau2_with_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        s = scaleTau2(X)
        self.assertEqual(len(s), 5)

    def test_scale_data_multivariate(self):
        m = np.mean(self.X, axis=0)
        s = np.std(self.X, axis=0)
        Xs = scale_data(self.X, m, s)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_scale_data_univariate(self):
        x = self.X[:, 0:1]
        m = np.array([np.mean(x)])
        s = np.array([np.std(x)])
        Xs = scale_data(x, m, s)
        self.assertEqual(Xs.shape, x.shape)

    def test_scale_data_1d(self):
        x = self.X[:, 0]
        m = np.mean(x)
        s = np.std(x)
        Xs = scale_data(x, m, s)
        self.assertEqual(len(Xs), len(x))


class TestPrepro(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(30, 5)

    def test_versatile_scaler_mean_std(self):
        vs = VersatileScaler(center='mean', scale='std')
        Xs = vs.fit_transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)
        np.testing.assert_array_almost_equal(np.mean(Xs, axis=0), 0, decimal=10)

    def test_versatile_scaler_median_mad(self):
        vs = VersatileScaler(center='median', scale='mad')
        Xs = vs.fit_transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_none_none(self):
        vs = VersatileScaler(center='None', scale='None')
        Xs = vs.fit_transform(self.X)
        np.testing.assert_array_almost_equal(Xs, self.X)

    def test_versatile_scaler_l1median_scaleTau2(self):
        vs = VersatileScaler(center='l1median', scale='scaleTau2')
        Xs = vs.fit_transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_kstepLTS_mad(self):
        vs = VersatileScaler(center='kstepLTS', scale='mad')
        Xs = vs.fit_transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_callable(self):
        from ._preproc_utilities import mean as _mean, std as _std
        vs = VersatileScaler(center=_mean, scale=_std)
        vs.fit(self.X)
        Xs = vs.transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_transform(self):
        vs = VersatileScaler(center='mean', scale='std')
        vs.fit(self.X)
        Xs = vs.transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_predict(self):
        vs = VersatileScaler(center='mean', scale='std')
        vs.fit(self.X)
        vs.transform(self.X)
        Xn = np.random.randn(5, 5)
        Xns = vs.predict(Xn)
        self.assertEqual(Xns.shape, (5, 5))

    def test_versatile_scaler_inverse_transform(self):
        vs = VersatileScaler(center='mean', scale='std')
        Xs = vs.fit_transform(self.X)
        Xr = vs.inverse_transform(Xs)
        np.testing.assert_array_almost_equal(Xr, self.X, decimal=10)

    def test_versatile_scaler_inverse_transform_no_arg(self):
        vs = VersatileScaler(center='mean', scale='std')
        vs.fit_transform(self.X)
        Xr = vs.inverse_transform()
        np.testing.assert_array_almost_equal(Xr, self.X, decimal=10)

    def test_versatile_scaler_1d_input(self):
        vs = VersatileScaler(center='mean', scale='std')
        x = self.X[:, 0]
        Xs = vs.fit_transform(x)
        self.assertEqual(Xs.shape[0], len(x))

    def test_versatile_scale_function(self):
        Xs = versatile_scale(self.X, center='l1median', scale='mad')
        self.assertEqual(Xs.shape, self.X.shape)

    def test_versatile_scaler_trimmed(self):
        vs = VersatileScaler(center='mean', scale='std', trimming=0.1)
        Xs = vs.fit_transform(self.X)
        self.assertEqual(Xs.shape, self.X.shape)


if __name__ == "__main__":
    unittest.main()
