"""F-score, precision, and recall - CPU version."""
import numpy as np


def compute_f(T, H):
    """
    Compute F-score, precision, and recall.

    Args:
        T: Ground-truth labels (NumPy array).
        H: Cluster assignments (NumPy array).

    Returns:
        f: F-score.
        p: Precision.
        r: Recall.

    Note: Runs on CPU.
    """
    if len(T) != len(H):
        raise ValueError('T and H must have the same length')

    N = len(T)
    numT = 0
    numH = 0
    numI = 0

    for n in range(N):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(Tn & Hn)

    p = 1
    r = 1
    f = 1

    if numH > 0:
        p = numI / numH

    if numT > 0:
        r = numI / numT

    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return f, p, r
