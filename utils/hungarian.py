"""Hungarian algorithm (scipy) - CPU version."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(A):
    """
    Solve the assignment problem with the Hungarian method.

    Args:
        A: Square cost matrix (NumPy array).

    Returns:
        C: Optimal assignment (NumPy array).
        T: Cost of the optimal assignment.

    Note: Runs on CPU (uses scipy).
    """
    row_ind, col_ind = linear_sum_assignment(A)
    C = col_ind
    T = A[row_ind, col_ind].sum()

    return C, T
