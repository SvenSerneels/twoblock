# -*- coding: utf-8 -*-
"""
Tests for SPADIMO - SPArse DIrections of Maximal Outlyingness

@author: Sven Serneels
"""

import numpy as np
import pandas as ps
import pytest
from numpy.testing import assert_array_equal

from .spadimo import spadimo
from ._preproc_utilities import Qn, mad


class TestQn:
    """Tests for Qn scale estimator."""

    def test_qn_basic(self):
        """Test Qn on simple data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        s = Qn(X)
        assert s.shape == (3,)
        # Qn should be close to 1 for standard normal
        assert all(s > 0.5) and all(s < 2.0)

    def test_qn_constant_column(self):
        """Test Qn handles constant columns."""
        X = np.ones((50, 2))
        X[:, 1] = np.random.randn(50)
        s = Qn(X)
        assert s[0] == 0.0  # constant column
        assert s[1] > 0  # non-constant column

    def test_qn_vs_mad(self):
        """Test Qn is similar to MAD for normal data."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        s_qn = Qn(X)
        s_mad = mad(X)
        # Both should be close to 1 for normal data
        np.testing.assert_allclose(s_qn, s_mad, rtol=0.5)


class TestSpadimo:
    """Tests for SPADIMO class."""

    @pytest.fixture
    def synthetic_data_with_outlier(self):
        """Create synthetic data with a known outlier."""
        np.random.seed(42)
        n, p = 100, 10

        # Generate normal data
        X = np.random.randn(n, p)

        # Create weights (all ones except outlier)
        weights = np.ones(n)

        # Make observation 0 an outlier in variables 0, 1, 2
        X[0, 0] = 10.0
        X[0, 1] = 8.0
        X[0, 2] = 9.0
        weights[0] = 0.1  # Low weight for outlier

        return X, weights, 0

    def test_fit_basic(self, synthetic_data_with_outlier):
        """Test basic fitting."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo(scale='mad')
        sp.fit(X, weights, obs)

        # Check attributes exist
        assert hasattr(sp, 'outlvars_')
        assert hasattr(sp, 'direction_')
        assert hasattr(sp, 'eta_')
        assert hasattr(sp, 'outlyingness_before_')
        assert hasattr(sp, 'outlyingness_after_')
        assert hasattr(sp, 'contributions_')

    def test_outlier_detection(self, synthetic_data_with_outlier):
        """Test that SPADIMO identifies the outlying variables."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo(scale='mad', stop_early=False)
        sp.fit(X, weights, obs)

        # The first 3 variables should be flagged
        # (they have the extreme values)
        flagged = set(sp.outlvars_)
        assert 0 in flagged, "Variable 0 should be flagged"
        assert 1 in flagged, "Variable 1 should be flagged"
        assert 2 in flagged, "Variable 2 should be flagged"

    def test_outlyingness_decreases(self, synthetic_data_with_outlier):
        """Test that outlyingness decreases after removing flagged variables."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo(scale='mad')
        sp.fit(X, weights, obs)

        # After removing flagged variables, outlyingness should decrease
        # (unless no variables were flagged)
        if len(sp.outlvars_) > 0:
            assert sp.outlyingness_after_ <= sp.outlyingness_before_

    def test_different_scales(self, synthetic_data_with_outlier):
        """Test with different scale estimators."""
        X, weights, obs = synthetic_data_with_outlier

        for scale in ['Qn', 'mad', 'scaleTau2']:
            sp = spadimo(scale=scale)
            sp.fit(X, weights, obs)
            assert hasattr(sp, 'outlvars_')

    def test_stop_early(self, synthetic_data_with_outlier):
        """Test early stopping behavior."""
        X, weights, obs = synthetic_data_with_outlier

        sp_early = spadimo(stop_early=True)
        sp_full = spadimo(stop_early=False)

        sp_early.fit(X, weights, obs)
        sp_full.fit(X, weights, obs)

        # Early stopping should stop at first non-outlying eta
        # This might result in fewer iterations
        assert sp_early.eta_ >= sp_full.eta_

    def test_custom_etas(self, synthetic_data_with_outlier):
        """Test with custom eta values."""
        X, weights, obs = synthetic_data_with_outlier

        custom_etas = [0.8, 0.5, 0.2]
        sp = spadimo(etas=custom_etas)
        sp.fit(X, weights, obs)

        assert sp.eta_ in custom_etas

    def test_pandas_input(self, synthetic_data_with_outlier):
        """Test with pandas DataFrame input."""
        X, weights, obs = synthetic_data_with_outlier

        df = ps.DataFrame(X, columns=[f'var_{i}' for i in range(X.shape[1])])
        sp = spadimo()
        sp.fit(df, weights, obs)

        assert sp.colnames_ is not None
        assert len(sp.colnames_) == X.shape[1]

    def test_get_outlying_variables(self, synthetic_data_with_outlier):
        """Test get_outlying_variables method."""
        X, weights, obs = synthetic_data_with_outlier

        df = ps.DataFrame(X, columns=[f'var_{i}' for i in range(X.shape[1])])
        sp = spadimo()
        sp.fit(df, weights, obs)

        indices = sp.get_outlying_variables(names=False)
        names = sp.get_outlying_variables(names=True)

        assert len(indices) == len(names)
        if len(indices) > 0:
            assert all(isinstance(n, str) for n in names)

    def test_direction_normalized(self, synthetic_data_with_outlier):
        """Test that direction vector is normalized."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo()
        sp.fit(X, weights, obs)

        # Direction should be approximately unit norm
        norm = np.sqrt(np.sum(sp.direction_ ** 2))
        np.testing.assert_allclose(norm, 1.0, rtol=0.1)

    def test_invalid_obs_index(self, synthetic_data_with_outlier):
        """Test error handling for invalid observation index."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo()

        with pytest.raises(ValueError, match="obs must be"):
            sp.fit(X, weights, 999)

        with pytest.raises(ValueError, match="obs must be"):
            sp.fit(X, weights, -1)

    def test_invalid_weights_length(self, synthetic_data_with_outlier):
        """Test error handling for invalid weights length."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo()

        with pytest.raises(ValueError, match="weights must have length"):
            sp.fit(X, weights[:50], obs)

    def test_invalid_scale(self, synthetic_data_with_outlier):
        """Test error handling for invalid scale."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo(scale='invalid_scale')

        with pytest.raises(ValueError, match="scale must be one of"):
            sp.fit(X, weights, obs)

    def test_contributions_positive(self, synthetic_data_with_outlier):
        """Test that contributions are non-negative."""
        X, weights, obs = synthetic_data_with_outlier

        sp = spadimo()
        sp.fit(X, weights, obs)

        assert all(sp.contributions_ >= 0)

    def test_n_greater_than_p(self):
        """Test standard n > p case."""
        np.random.seed(42)
        n, p = 200, 10
        X = np.random.randn(n, p)
        X[0, :3] = 5.0  # outlier
        weights = np.ones(n)
        weights[0] = 0.1

        sp = spadimo()
        sp.fit(X, weights, 0)

        assert hasattr(sp, 'outlvars_')

    def test_n_close_to_p(self):
        """Test case where n is close to p."""
        np.random.seed(42)
        n, p = 30, 25
        X = np.random.randn(n, p)
        X[0, :5] = 4.0  # outlier
        weights = np.ones(n)
        weights[0] = 0.1

        sp = spadimo()
        sp.fit(X, weights, 0)

        assert hasattr(sp, 'outlvars_')


class TestSpadimoGPU:
    """GPU-specific tests for SPADIMO."""

    @pytest.fixture
    def check_cupy(self):
        """Check if CuPy is available."""
        try:
            import cupy
            return True
        except ImportError:
            pytest.skip("CuPy not available")

    @pytest.mark.gpu
    def test_gpu_basic(self, check_cupy):
        """Test basic GPU functionality."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        X[0, :3] = 5.0
        weights = np.ones(n)
        weights[0] = 0.1

        sp = spadimo(gpu=True)
        sp.fit(X, weights, 0)

        assert hasattr(sp, 'outlvars_')
        # Results should be numpy arrays (converted back from GPU)
        assert isinstance(sp.direction_, np.ndarray)

    @pytest.mark.gpu
    def test_gpu_cpu_consistency(self, check_cupy):
        """Test that GPU and CPU give consistent results."""
        np.random.seed(42)
        n, p = 100, 10
        X = np.random.randn(n, p)
        X[0, :3] = 5.0
        weights = np.ones(n)
        weights[0] = 0.1

        sp_cpu = spadimo(gpu=False)
        sp_gpu = spadimo(gpu=True)

        sp_cpu.fit(X.copy(), weights.copy(), 0)
        sp_gpu.fit(X.copy(), weights.copy(), 0)

        # Results should be similar
        np.testing.assert_allclose(
            sp_cpu.outlyingness_before_,
            sp_gpu.outlyingness_before_,
            rtol=1e-5
        )
        np.testing.assert_array_equal(sp_cpu.outlvars_, sp_gpu.outlvars_)
