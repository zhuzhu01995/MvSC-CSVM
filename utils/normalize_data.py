"""Data normalization - GPU-accelerated version."""
import sys
sys.path.append('..')
from gpu_utils import xp
import numpy as np


def normalize_data(X):
    """
    Column-wise normalization (GPU-accelerated).

    Args:
        X: Array of shape (nFea, nSmp) (NumPy or CuPy array).

    Returns:
        ProcessData: Normalized data (same type as input).
    """
    # Detect input type
    is_numpy = isinstance(X, np.ndarray)

    # NumPy arrays stay on CPU
    if is_numpy:
        X_proc = np.asarray(X)
        norms = np.linalg.norm(X_proc, axis=0)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        ProcessData = X_proc / norms[np.newaxis, :]
    else:
        # CuPy arrays processed on GPU
        norms = xp.linalg.norm(X, axis=0)
        norms = xp.maximum(norms, 1e-12)  # Avoid division by zero
        ProcessData = X / norms[xp.newaxis, :]

    return ProcessData
