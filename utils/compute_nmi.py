"""Normalized mutual information (NMI) - CPU version."""
import numpy as np


def compute_nmi(T, H):
    """
    Compute normalized mutual information.

    Args:
        T: Ground-truth labels (NumPy array).
        H: Cluster assignments (NumPy array).

    Returns:
        A: Confusion matrix.
        nmi: Normalized mutual information.
        avgent: Average entropy.

    Note: Runs on CPU.
    """
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)

    # Points per class
    D = np.zeros(num_class)
    for j in range(num_class):
        index_class = (T == classes[j])
        D[j] = np.sum(index_class)

    # Mutual information
    mi = 0
    A = np.zeros((num_clust, num_class))
    avgent = 0
    B = np.zeros(num_clust)
    miarr = np.zeros((num_clust, num_class))

    for i in range(num_clust):
        # Points in cluster i
        index_clust = (H == clusters[i])
        B[i] = np.sum(index_clust)

        for j in range(num_class):
            index_class = (T == classes[j])
            # Points from class j assigned to cluster i
            A[i, j] = np.sum(index_class & index_clust)

            if A[i, j] != 0:
                miarr[i, j] = A[i, j] / N * np.log2(N * A[i, j] / (B[i] * D[j]))
                # Average entropy term
                avgent = avgent - (B[i] / N) * (A[i, j] / B[i]) * np.log2(A[i, j] / B[i])
            else:
                miarr[i, j] = 0

            mi = mi + miarr[i, j]

    # Class entropy
    class_ent = 0
    for i in range(num_class):
        if D[i] > 0:
            class_ent = class_ent + D[i] / N * np.log2(N / D[i])

    # Cluster entropy
    clust_ent = 0
    for i in range(num_clust):
        if B[i] > 0:
            clust_ent = clust_ent + B[i] / N * np.log2(N / B[i])

    # Normalized mutual information
    if (clust_ent + class_ent) > 0:
        nmi = 2 * mi / (clust_ent + class_ent)
    else:
        nmi = 0

    return A, nmi, avgent
