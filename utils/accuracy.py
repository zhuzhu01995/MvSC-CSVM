"""Clustering accuracy - CPU version."""
import numpy as np
from .best_map import best_map


def accuracy(C, gt):
    """
    Compute clustering accuracy.

    Args:
        C: Cluster assignments (NumPy array).
        gt: Ground-truth labels (NumPy array).

    Returns:
        ACC: Accuracy.

    Note: Runs on CPU.
    """
    C = best_map(gt, C)
    ACC = np.sum(gt == C) / len(gt)
    return ACC
