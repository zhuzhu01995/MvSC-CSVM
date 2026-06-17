"""Contingency table - GPU-accelerated version."""
import numpy as np


def contingency(Mem1, Mem2):
    """
    Build a contingency matrix for two label vectors.

    Args:
        Mem1: First membership vector (NumPy array).
        Mem2: Second membership vector (NumPy array).

    Returns:
        Cont: Contingency matrix (NumPy array).

    Note: Runs on CPU due to indexing operations.
    """
    if Mem1.ndim > 1 or Mem2.ndim > 1:
        raise ValueError('Contingency: Requires two vector arguments')

    if len(Mem1) != len(Mem2):
        raise ValueError('Contingency: Vectors must have the same length')

    Cont = np.zeros((int(np.max(Mem1)), int(np.max(Mem2))))

    for i in range(len(Mem1)):
        Cont[int(Mem1[i])-1, int(Mem2[i])-1] += 1

    return Cont
