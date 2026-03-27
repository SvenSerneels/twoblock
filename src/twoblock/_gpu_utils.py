# -*- coding: utf-8 -*-
"""
GPU backend utilities for optional CuPy acceleration.
"""

import numpy as np


def get_array_module(gpu=False):
    """Return (xp, xp_linalg) tuple for array operations."""
    if gpu:
        try:
            import cupy as cp
            return cp, cp.linalg
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU support. "
                "Install with: pip install cupy-cuda12x"
            )
    import scipy.linalg as sp_linalg
    return np, sp_linalg


def to_xp(arr, xp):
    """Convert array to target array module."""
    return xp.asarray(arr)


def to_numpy(arr):
    """Ensure array is numpy (handles both numpy and cupy input)."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)
