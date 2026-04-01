# -*- coding: utf-8 -*-
"""
Unit tests for CRM - Cellwise Robust M-regression.

@author: Sven Serneels
"""

import unittest
import numpy as np
import pandas as ps
import pytest

from .crm import crm


class TestCRM(unittest.TestCase):
    """Test methods in the crm class"""

    @classmethod
    def setUpClass(cls):
        print("...setupClass TestCRM")

    @classmethod
    def tearDownClass(cls):
        print("...teardownClass TestCRM")

    def setUp(self):
        # Generate clean synthetic data
        np.random.seed(42)
        self.n = 100
        self.p = 5
        self.X = np.random.randn(self.n, self.p)
        self.beta_true = np.array([1, 2, 0, -1, 0.5])
        self.y = self.X @ self.beta_true + 0.1 * np.random.randn(self.n)

        # Create contaminated version
        self.X_cont = self.X.copy()
        self.y_cont = self.y.copy()

        # Inject cellwise outliers
        self.X_cont[0, 0] = 10  # Cell (0, 0) is outlier
        self.X_cont[5, 2] = -8  # Cell (5, 2) is outlier
        self.X_cont[10, 1] = 15  # Cell (10, 1) is outlier

    def tearDown(self):
        pass

    def test_fit_clean_data(self):
        """Test basic fit on clean data"""
        model = crm(verbose=False)
        model.fit(self.X, self.y)

        # Check fitted attributes exist
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        self.assertTrue(hasattr(model, 'fitted_'))
        self.assertTrue(hasattr(model, 'residuals_'))
        self.assertTrue(hasattr(model, 'caseweights_'))
        self.assertTrue(hasattr(model, 'cellweights_'))
        self.assertTrue(hasattr(model, 'cellwise_outliers_'))
        self.assertTrue(hasattr(model, 'casewise_outliers_'))
        self.assertTrue(hasattr(model, 'X_imputed_'))
        self.assertTrue(hasattr(model, 'n_iter_'))

        # Check shapes
        self.assertEqual(model.coef_.shape, (self.p,))
        self.assertEqual(model.fitted_.shape, (self.n,))
        self.assertEqual(model.residuals_.shape, (self.n,))
        self.assertEqual(model.caseweights_.shape, (self.n,))
        self.assertEqual(model.cellweights_.shape, (self.n, self.p))
        self.assertEqual(model.cellwise_outliers_.shape, (self.n, self.p))
        self.assertEqual(model.casewise_outliers_.shape, (self.n,))
        self.assertEqual(model.X_imputed_.shape, (self.n, self.p))

        # Check coefficients are reasonable (close to true)
        # On clean data, should recover true coefficients fairly well
        coef_error = np.sqrt(np.sum((model.coef_ - self.beta_true) ** 2))
        self.assertLess(coef_error, 1.0, "Coefficients should be close to true values")

    def test_fit_contaminated_data(self):
        """Test fit on contaminated data detects outliers"""
        model = crm(verbose=False, crit_cellwise=0.99)
        model.fit(self.X_cont, self.y_cont)

        # Should detect at least some cellwise outliers
        n_cellwise = np.sum(model.cellwise_outliers_)
        self.assertGreater(n_cellwise, 0, "Should detect cellwise outliers")

        # Coefficients should still be reasonable
        # (may not be as close as clean data, but not wildly wrong)
        coef_error = np.sqrt(np.sum((model.coef_ - self.beta_true) ** 2))
        self.assertLess(coef_error, 3.0, "Coefficients should still be reasonable")

    def test_predict(self):
        """Test predict function"""
        model = crm(verbose=False)
        model.fit(self.X, self.y)

        # Predict on training data
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (self.n,))

        # Predict on new data
        X_new = np.random.randn(10, self.p)
        y_pred_new = model.predict(X_new)
        self.assertEqual(y_pred_new.shape, (10,))

    def test_predict_wrong_dimension(self):
        """Test predict raises error for wrong dimensions"""
        model = crm(verbose=False)
        model.fit(self.X, self.y)

        X_wrong = np.random.randn(10, self.p + 2)
        with self.assertRaises(ValueError):
            model.predict(X_wrong)

    def test_transform(self):
        """Test transform (imputation) function"""
        model = crm(verbose=False)
        model.fit(self.X_cont, self.y_cont)

        X_imputed = model.transform(self.X_cont)
        self.assertEqual(X_imputed.shape, self.X_cont.shape)

    def test_regtype_lts(self):
        """Test with LTS regression type"""
        model = crm(regtype='LTS', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        coef_error = np.sqrt(np.sum((model.coef_ - self.beta_true) ** 2))
        self.assertLess(coef_error, 1.5, "LTS should recover coefficients")

    def test_regtype_mm(self):
        """Test with MM regression type (default)"""
        model = crm(regtype='MM', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        coef_error = np.sqrt(np.sum((model.coef_ - self.beta_true) ** 2))
        self.assertLess(coef_error, 1.5, "MM should recover coefficients")

    def test_fun_huber(self):
        """Test with Huber weighting function"""
        model = crm(fun='Huber', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'caseweights_'))

    def test_fun_fair(self):
        """Test with Fair weighting function"""
        model = crm(fun='Fair', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'caseweights_'))

    def test_fun_hampel(self):
        """Test with Hampel weighting function (default)"""
        model = crm(fun='Hampel', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'caseweights_'))

    def test_scale_mad(self):
        """Test with MAD scale estimator"""
        model = crm(scale='mad', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))

    def test_scale_scaletau2(self):
        """Test with scaleTau2 scale estimator"""
        model = crm(scale='scaleTau2', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))

    def test_center_mean(self):
        """Test with mean centering"""
        model = crm(center='mean', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))

    def test_center_l1median(self):
        """Test with L1 median centering"""
        model = crm(center='l1median', verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))

    def test_pandas_input(self):
        """Test with pandas DataFrame input"""
        X_df = ps.DataFrame(
            self.X, columns=[f'var_{i}' for i in range(self.p)]
        )
        y_series = ps.Series(self.y, name='response')

        model = crm(verbose=False)
        model.fit(X_df, y_series)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'colnames_'))
        self.assertEqual(len(model.colnames_), self.p)

    def test_summary(self):
        """Test summary method"""
        model = crm(verbose=False)
        model.fit(self.X, self.y)

        # Should not raise
        model.summary()

    def test_get_cellwise_outliers(self):
        """Test get_cellwise_outliers method"""
        model = crm(verbose=False)
        model.fit(self.X_cont, self.y_cont)

        # Get all
        outliers = model.get_cellwise_outliers()
        self.assertEqual(outliers.shape, (self.n, self.p))

        # Get for specific row
        row_outliers = model.get_cellwise_outliers(row=0)
        self.assertIsInstance(row_outliers, np.ndarray)

    def test_get_casewise_outliers(self):
        """Test get_casewise_outliers method"""
        model = crm(verbose=False)
        model.fit(self.X_cont, self.y_cont)

        outliers = model.get_casewise_outliers()
        self.assertIsInstance(outliers, np.ndarray)

    def test_invalid_fun(self):
        """Test error for invalid weighting function"""
        model = crm(fun='invalid', verbose=False)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)

    def test_invalid_hampel_params(self):
        """Test error for invalid Hampel parameters"""
        # probp1 > probp2
        model = crm(fun='Hampel', probp1=0.99, probp2=0.98, verbose=False)
        with self.assertRaises(ValueError):
            model.fit(self.X, self.y)

    def test_x_y_length_mismatch(self):
        """Test error for X and y length mismatch"""
        model = crm(verbose=False)
        y_wrong = np.random.randn(self.n + 5)
        with self.assertRaises(ValueError):
            model.fit(self.X, y_wrong)

    def test_convergence(self):
        """Test that model converges"""
        model = crm(maxiter=100, tolerance=0.01, verbose=False)
        model.fit(self.X, self.y)

        # Should converge in reasonable iterations for clean data
        self.assertLess(model.n_iter_, 100, "Should converge before maxiter")

    def test_imputation_correctness(self):
        """Test that imputation replaces outlying cells"""
        model = crm(verbose=False, crit_cellwise=0.99)
        model.fit(self.X_cont, self.y_cont)

        # For cells marked as outliers, X_imputed should differ from X
        if np.any(model.cellwise_outliers_):
            outlier_rows, outlier_cols = np.where(model.cellwise_outliers_)
            for i, j in zip(outlier_rows, outlier_cols):
                # Imputed value should differ from original (which was outlier)
                self.assertNotAlmostEqual(
                    model.X_imputed_[i, j],
                    self.X_cont[i, j],
                    places=5,
                    msg=f"Cell ({i}, {j}) should be imputed"
                )

    def test_single_outlier(self):
        """Test with single cellwise outlier"""
        X = self.X.copy()
        y = self.y.copy()
        X[50, 2] = 20  # Single large outlier

        model = crm(verbose=False, crit_cellwise=0.99)
        model.fit(X, y)

        # Should detect the outlier or at least not crash
        self.assertTrue(hasattr(model, 'coef_'))

    def test_many_outliers(self):
        """Test with many cellwise outliers"""
        X = self.X.copy()

        # Contaminate 10% of cells
        n_outliers = int(0.1 * self.n * self.p)
        rows = np.random.choice(self.n, n_outliers)
        cols = np.random.choice(self.p, n_outliers)
        X[rows, cols] += 10 * np.random.randn(n_outliers)

        model = crm(verbose=False)
        model.fit(X, self.y)

        # Should still produce reasonable coefficients
        self.assertTrue(hasattr(model, 'coef_'))

    def test_small_n(self):
        """Test with small sample size"""
        X = np.random.randn(20, 3)
        beta = np.array([1, -1, 0.5])
        y = X @ beta + 0.1 * np.random.randn(20)

        model = crm(verbose=False)
        model.fit(X, y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertEqual(model.coef_.shape, (3,))

    def test_start_cellwise_clean_data(self):
        """Test start_cellwise=True on clean data"""
        try:
            from robpy.outliers.ddc import DDC  # noqa: F401
        except ImportError:
            self.skipTest("robpy not available")

        model = crm(start_cellwise=True, verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'ddc_outliers_'))
        # ddc_outliers_ should be set when start_cellwise=True
        self.assertIsNotNone(model.ddc_outliers_)
        self.assertEqual(model.ddc_outliers_.shape, (self.n, self.p))

    def test_start_cellwise_contaminated_data(self):
        """Test start_cellwise=True on contaminated data"""
        try:
            from robpy.outliers.ddc import DDC  # noqa: F401
        except ImportError:
            self.skipTest("robpy not available")

        model = crm(start_cellwise=True, verbose=False)
        model.fit(self.X_cont, self.y_cont)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'ddc_outliers_'))
        # DDC should detect some outliers in contaminated data
        self.assertIsNotNone(model.ddc_outliers_)

        # Coefficients should still be reasonable
        coef_error = np.sqrt(np.sum((model.coef_ - self.beta_true) ** 2))
        self.assertLess(coef_error, 3.0, "Coefficients should be reasonable")

    def test_start_cellwise_false_no_ddc(self):
        """Test that start_cellwise=False does not use DDC"""
        model = crm(start_cellwise=False, verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        # ddc_outliers_ should be None when start_cellwise=False
        self.assertIsNone(model.ddc_outliers_)

    def test_start_cellwise_stores_original_data(self):
        """Test that original X and y are stored when using DDC"""
        try:
            from robpy.outliers.ddc import DDC  # noqa: F401
        except ImportError:
            self.skipTest("robpy not available")

        model = crm(start_cellwise=True, verbose=False)
        model.fit(self.X_cont, self.y_cont)

        # X_ and y_ should be the original (not DDC-imputed) data
        np.testing.assert_array_equal(model.X_, self.X_cont)
        np.testing.assert_array_equal(model.y_, self.y_cont)


@pytest.mark.gpu
class TestCRMGPU(unittest.TestCase):
    """GPU tests for crm class"""

    @classmethod
    def setUpClass(cls):
        try:
            import cupy  # noqa: F401
            cls.gpu_available = True
        except ImportError:
            cls.gpu_available = False
        print("...setupClass TestCRMGPU")

    def setUp(self):
        if not self.gpu_available:
            self.skipTest("CuPy not available")

        np.random.seed(42)
        self.n = 50
        self.p = 3
        self.X = np.random.randn(self.n, self.p)
        self.beta_true = np.array([1, -0.5, 0.5])
        self.y = self.X @ self.beta_true + 0.1 * np.random.randn(self.n)

    def test_gpu_fit(self):
        """Test fit with GPU enabled"""
        model = crm(gpu=True, verbose=False)
        model.fit(self.X, self.y)

        self.assertTrue(hasattr(model, 'coef_'))
        self.assertEqual(model.coef_.shape, (self.p,))

    def test_gpu_predict(self):
        """Test predict with GPU"""
        model = crm(gpu=True, verbose=False)
        model.fit(self.X, self.y)

        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (self.n,))


if __name__ == "__main__":
    unittest.main()
