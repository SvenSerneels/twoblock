# -*- coding: utf-8 -*-
"""
SPADIMO - SPArse DIrections of Maximal Outlyingness

Identifies which variables contribute most to making an observation an outlier.
Python port of the R package's SPADIMO algorithm.

@author: Sven Serneels
"""

import numpy as np
import pandas as ps
from sklearn.base import BaseEstimator
from scipy.stats import chi2
import warnings

from .utils import _check_input
from ._preproc_utilities import mad, scaleTau2, kstepLTS, Qn, mean
from ._gpu_utils import get_array_module, to_xp, to_numpy


class spadimo(BaseEstimator):
    """
    SPADIMO - SPArse DIrections of Maximal Outlyingness

    Identifies which variables contribute most to making an observation an
    outlier. Given a data matrix X, case weights from a robust estimator,
    and an observation index to analyze, it returns the sparse direction of
    maximal outlyingness and the contributing variables.

    Parameters
    ----------
    scale : str, default 'Qn'
        Scale function for robust standardization.
        Options: 'Qn', 'mad', 'scaleTau2', 'kstepLTS'

    n_latent : int, default 1
        Number of latent variables for sparse direction.

    etas : array-like or None, default None
        Sparsity parameters. If None, uses sequence from 0.9 to 0.1 (step -0.05).

    csq_critv : float, default 0.975
        Chi-squared critical value quantile for determining outlyingness.

    stop_early : bool, default False
        If True, stop at first eta value where observation becomes non-outlying.

    gpu : bool, default False
        Enable GPU acceleration via CuPy.

    copy : bool, default True
        Copy input data.

    Attributes (post-fit)
    ---------------------
    outlvars_ : ndarray
        Variable indices contributing to outlyingness.

    direction_ : ndarray
        Sparse direction vector (p,).

    eta_ : float
        Final eta value where observation becomes non-outlying.

    outlyingness_before_ : float
        Outlyingness metric before variable removal.

    outlyingness_after_ : float
        Outlyingness metric after variable removal.

    contributions_ : ndarray
        Per-variable contribution scores.

    flagged_vars_by_eta_ : dict
        Dictionary mapping eta values to flagged variable indices.

    outlying_by_eta_ : dict
        Dictionary mapping eta values to boolean indicating if still outlying.
    """

    def __init__(
        self,
        scale='Qn',
        n_latent=1,
        etas=None,
        csq_critv=0.975,
        stop_early=False,
        gpu=False,
        copy=True,
    ):
        self.scale = scale
        self.n_latent = n_latent
        self.etas = etas
        self.csq_critv = csq_critv
        self.stop_early = stop_early
        self.gpu = gpu
        self.copy = copy

    def fit(self, X, weights, obs):
        """
        Analyze observation for outlyingness and identify contributing variables.

        Parameters
        ----------
        X : array-like of shape (n, p)
            Data matrix.

        weights : array-like of shape (n,)
            Case weights from a robust estimator (e.g., from rtb.caseweights_).
            Weights should be in [0, 1] with lower weights for outliers.

        obs : int
            Index of the observation to analyze (0-based).

        Returns
        -------
        self : spadimo
            Fitted estimator.
        """
        # Get array module for GPU support
        xp, xp_linalg = get_array_module(self.gpu)

        # Input validation
        if type(X) == ps.core.frame.DataFrame:
            self.colnames_ = list(X.columns)
            X = X.to_numpy().astype('float64')
        else:
            self.colnames_ = None

        X = _check_input(X)
        n, p = X.shape

        if self.copy:
            X = X.copy()

        weights = np.asarray(weights).ravel()
        if len(weights) != n:
            raise ValueError(
                f"weights must have length {n}, got {len(weights)}"
            )

        if obs < 0 or obs >= n:
            raise ValueError(
                f"obs must be in [0, {n-1}], got {obs}"
            )

        # Set default etas if not provided
        if self.etas is None:
            etas = np.arange(0.9, 0.05, -0.05)
        else:
            etas = np.asarray(self.etas)

        # Robust standardization using weighted mean for center
        # and selected robust scale estimator
        w = weights.copy()
        w_sum = np.sum(w)
        if w_sum < 1e-10:
            w_sum = 1.0

        # Weighted mean for centering
        mu = np.sum(X * w[:, np.newaxis], axis=0) / w_sum

        # Select scale function
        scale_funcs = {
            'Qn': Qn,
            'mad': mad,
            'scaleTau2': scaleTau2,
            'kstepLTS': lambda x, **kw: np.std(x, axis=0),  # fallback
        }
        if self.scale not in scale_funcs:
            raise ValueError(
                f"scale must be one of {list(scale_funcs.keys())}, got {self.scale}"
            )
        scale_func = scale_funcs[self.scale]
        sigma = scale_func(X)

        # Handle zero scales
        sigma[sigma < 1e-10] = 1.0

        # Standardized data
        Z = (X - mu) / sigma

        # Move to GPU if requested
        if self.gpu:
            Z = to_xp(Z, xp)
            w = to_xp(w, xp)

        # Compute initial outlyingness
        outlyingness_before = self._compute_outlyingness(Z, w, obs, xp, xp_linalg)

        # Chi-squared threshold
        threshold = np.sqrt(chi2.ppf(self.csq_critv, p))

        # Store results for each eta
        flagged_vars_by_eta = {}
        outlying_by_eta = {}
        direction_by_eta = {}

        # Track cumulative flagged variables
        all_flagged = set()
        final_eta = etas[0]
        final_direction = None

        for eta in etas:
            # Compute sparse direction
            a = self._spadimo_exs(Z, w, obs, eta, xp, xp_linalg)

            if self.gpu:
                a_np = to_numpy(a)
            else:
                a_np = a

            direction_by_eta[eta] = a_np

            # Identify contributing variables (soft-thresholding)
            abs_a = np.abs(a_np)
            max_abs_a = np.max(abs_a)
            if max_abs_a > 1e-10:
                flagged_mask = (abs_a - eta * max_abs_a) >= 0
                flagged_indices = np.where(flagged_mask)[0]
            else:
                flagged_indices = np.array([], dtype=int)

            flagged_vars_by_eta[eta] = flagged_indices
            all_flagged.update(flagged_indices.tolist())

            # Compute outlyingness on reduced data (excluding flagged vars)
            remaining_vars = [i for i in range(p) if i not in all_flagged]

            if len(remaining_vars) > 0:
                Z_reduced = Z[:, remaining_vars]
                if len(remaining_vars) < p:
                    # Recompute threshold for reduced dimension
                    threshold_reduced = np.sqrt(chi2.ppf(self.csq_critv, len(remaining_vars)))
                else:
                    threshold_reduced = threshold

                outlyingness_after = self._compute_outlyingness(
                    Z_reduced, w, obs, xp, xp_linalg
                )
                is_outlying = outlyingness_after > threshold_reduced
            else:
                outlyingness_after = 0.0
                is_outlying = False

            outlying_by_eta[eta] = is_outlying
            final_eta = eta
            final_direction = a_np

            if self.stop_early and not is_outlying:
                break

        # Compute per-variable contributions from final direction
        if final_direction is not None:
            contributions = np.abs(final_direction) * np.abs(to_numpy(Z[obs, :]) if self.gpu else Z[obs, :])
        else:
            contributions = np.zeros(p)

        # Store fitted attributes
        setattr(self, 'outlvars_', np.array(sorted(all_flagged), dtype=int))
        setattr(self, 'direction_', final_direction)
        setattr(self, 'eta_', final_eta)
        setattr(self, 'outlyingness_before_', float(to_numpy(outlyingness_before) if self.gpu else outlyingness_before))
        setattr(self, 'outlyingness_after_', float(to_numpy(outlyingness_after) if self.gpu else outlyingness_after))
        setattr(self, 'contributions_', contributions)
        setattr(self, 'flagged_vars_by_eta_', flagged_vars_by_eta)
        setattr(self, 'outlying_by_eta_', outlying_by_eta)
        setattr(self, 'threshold_', threshold)
        setattr(self, 'X_standardized_', to_numpy(Z) if self.gpu else Z)
        setattr(self, 'center_', mu)
        setattr(self, 'scale_', sigma)

        return self

    def _spadimo_exs(self, Z, w, obs, eta, xp, xp_linalg):
        """
        Compute sparse direction of maximal outlyingness.

        Parameters
        ----------
        Z : array
            Standardized data matrix (n x p).
        w : array
            Case weights (n,).
        obs : int
            Observation index.
        eta : float
            Sparsity parameter.
        xp : module
            Array module (numpy or cupy).
        xp_linalg : module
            Linear algebra module.

        Returns
        -------
        a : array
            Sparse direction vector (p,).
        """
        n, p = Z.shape

        # Create weighted data: zw = sqrt(w) * z
        sqrt_w = xp.sqrt(w)
        Zw = Z * sqrt_w[:, None]

        # Create indicator: yw[obs] = 1, zeros elsewhere
        yw = xp.zeros(n)
        yw[obs] = 1.0

        # Compute direction: a = Zw.T @ yw
        a = Zw.T @ yw

        # Normalize
        norm_a = xp.sqrt(xp.sum(a ** 2))
        if norm_a > 1e-10:
            a = a / norm_a

        # Soft-threshold: keep where |a| - eta*max(|a|) >= 0
        abs_a = xp.abs(a)
        max_abs_a = xp.max(abs_a)
        if max_abs_a > 1e-10:
            threshold = eta * max_abs_a
            mask = abs_a >= threshold
            a = a * mask
        else:
            a = xp.zeros(p)

        # Re-normalize
        norm_a = xp.sqrt(xp.sum(a ** 2))
        if norm_a > 1e-10:
            a = a / norm_a

        return a

    def _compute_outlyingness(self, Z, w, obs, xp, xp_linalg):
        """
        Compute weighted Mahalanobis distance for outlyingness.

        Parameters
        ----------
        Z : array
            Standardized data matrix (n x p).
        w : array
            Case weights (n,).
        obs : int
            Observation index.
        xp : module
            Array module (numpy or cupy).
        xp_linalg : module
            Linear algebra module.

        Returns
        -------
        outlyingness : float
            Weighted Mahalanobis distance.
        """
        n, p = Z.shape

        # Check if n > p (standard case)
        if n > p:
            # Weighted covariance: Sigma_w = (Z.T @ diag(w) @ Z) / (sum(w) - 1)
            w_sum = xp.sum(w)
            if w_sum <= 1:
                w_sum = 1.0

            # Weighted mean (for re-centering)
            Zw = Z * w[:, None]
            mu_w = xp.sum(Zw, axis=0) / w_sum

            # Center data
            Zc = Z - mu_w

            # Weighted covariance
            Zcw = Zc * xp.sqrt(w)[:, None]
            Sigma_w = (Zcw.T @ Zcw) / (w_sum - 1)

            # Add small regularization for numerical stability
            Sigma_w = Sigma_w + xp.eye(p) * 1e-8

            # Mahalanobis distance: sqrt(z[obs].T @ inv(Sigma_w) @ z[obs])
            z_obs = Zc[obs, :]
            try:
                Sigma_inv = xp_linalg.inv(Sigma_w)
                outlyingness = xp.sqrt(z_obs @ Sigma_inv @ z_obs)
            except Exception:
                # Fallback to pseudo-inverse
                if xp.__name__ == 'cupy':
                    Sigma_inv = xp.linalg.pinv(Sigma_w)
                else:
                    Sigma_inv = xp_linalg.pinv(Sigma_w)
                outlyingness = xp.sqrt(z_obs @ Sigma_inv @ z_obs)
        else:
            # n <= p case: use PCA-based approach
            # Try to use robpy if available for true robust PCA
            try:
                from robpy.pca.base import RobustPCA
                # For high-dimensional case, use robust PCA distances
                Z_np = to_numpy(Z) if xp.__name__ == 'cupy' else Z
                rpca = RobustPCA(n_components=min(n-1, p))
                rpca.fit(Z_np)
                scores = rpca.transform(Z_np)
                # Compute distance in score space
                outlyingness = xp.sqrt(xp.sum(scores[obs, :] ** 2))
            except ImportError:
                # Fallback: use standard SVD-based approach
                warnings.warn(
                    "For n <= p, robpy is recommended for robust outlyingness. "
                    "Install with: pip install robpy. "
                    "Using SVD-based fallback."
                )
                # SVD-based Mahalanobis in reduced space
                U, S, Vt = xp_linalg.svd(Z, full_matrices=False)
                # Keep components with non-zero singular values
                tol = 1e-10 * S[0] if len(S) > 0 else 1e-10
                k = xp.sum(S > tol)
                if hasattr(k, 'item'):
                    k = k.item()
                k = max(1, min(k, n - 1))

                # Scores and scale by singular values
                scores = U[:, :k] * S[:k]
                score_cov = xp.cov(scores.T)
                if score_cov.ndim == 0:
                    score_cov = score_cov.reshape(1, 1)

                # Regularize
                score_cov = score_cov + xp.eye(k) * 1e-8

                try:
                    cov_inv = xp_linalg.inv(score_cov)
                except Exception:
                    if xp.__name__ == 'cupy':
                        cov_inv = xp.linalg.pinv(score_cov)
                    else:
                        cov_inv = xp_linalg.pinv(score_cov)

                z_scores = scores[obs, :]
                outlyingness = xp.sqrt(z_scores @ cov_inv @ z_scores)

        return outlyingness

    def get_outlying_variables(self, names=False):
        """
        Get the variables contributing to outlyingness.

        Parameters
        ----------
        names : bool, default False
            If True and column names are available, return names instead of indices.

        Returns
        -------
        outlvars : array or list
            Indices or names of outlying variables.
        """
        if not hasattr(self, 'outlvars_'):
            raise ValueError("Model not fitted. Call fit() first.")

        if names and self.colnames_ is not None:
            return [self.colnames_[i] for i in self.outlvars_]
        return self.outlvars_

    def summary(self):
        """
        Print a summary of the SPADIMO analysis.
        """
        if not hasattr(self, 'outlvars_'):
            raise ValueError("Model not fitted. Call fit() first.")

        print("SPADIMO Analysis Summary")
        print("=" * 50)
        print(f"Scale estimator: {self.scale}")
        print(f"Final eta: {self.eta_:.3f}")
        print(f"Outlyingness before: {self.outlyingness_before_:.4f}")
        print(f"Outlyingness after: {self.outlyingness_after_:.4f}")
        print(f"Threshold (chi-sq {self.csq_critv}): {self.threshold_:.4f}")
        print(f"Number of flagged variables: {len(self.outlvars_)}")

        if len(self.outlvars_) > 0:
            print("\nFlagged variables:")
            if self.colnames_ is not None:
                for idx in self.outlvars_:
                    print(f"  [{idx}] {self.colnames_[idx]}: "
                          f"contribution = {self.contributions_[idx]:.4f}")
            else:
                for idx in self.outlvars_:
                    print(f"  [{idx}]: contribution = {self.contributions_[idx]:.4f}")
