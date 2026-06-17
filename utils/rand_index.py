"""Rand index - CPU version."""
import numpy as np
from scipy.special import comb
from .contingency import contingency


def rand_index(c1, c2):
    """
    Compare two partitions with the Rand index.

    Args:
        c1: First cluster membership vector (NumPy array).
        c2: Second cluster membership vector (NumPy array).

    Returns:
        AR: Adjusted Rand index (Hubert & Arabie).
        RI: Unadjusted Rand index.
        MI: Mirkin index.
        HI: Hubert index.

    Note: Runs on CPU.
    """
    if c1.ndim > 1 or c2.ndim > 1:
        raise ValueError('RandIndex: Requires two vector arguments')

    if len(c1) != len(c2):
        raise ValueError('RandIndex: Vectors must have the same length')

    # Build contingency matrix
    C = contingency(c1, c2)

    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1) ** 2)  # Sum of squared row sums
    njs = np.sum(np.sum(C, axis=0) ** 2)  # Sum of squared column sums

    t1 = comb(int(n), 2, exact=True)  # Total number of entity pairs
    t2 = np.sum(C ** 2)  # Sum of n_ij^2
    t3 = 0.5 * (nis + njs)

    # Expected index (for adjustment)
    nc = (n * (n**2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))

    A = t1 + t2 - t3  # Agreements
    D = -t2 + t3  # Disagreements

    if t1 == nc:
        AR = 0  # Avoid division by zero; if k=1, define Rand = 0
    else:
        AR = (A - nc) / (t1 - nc)  # Adjusted Rand - Hubert & Arabie 1985

    RI = A / t1  # Rand 1971 - agreement probability
    MI = D / t1  # Mirkin 1970 - disagreement probability
    HI = (A - D) / t1  # Hubert 1977 - p(agree) - p(disagree)

    return AR, RI, MI, HI
