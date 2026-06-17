"""Spectral clustering - GPU-accelerated version."""
import numpy as np
from sklearn.cluster import KMeans
import warnings
import sys
sys.path.append('..')
from gpu_utils import to_cpu, to_gpu, xp

warnings.filterwarnings('ignore')


def spectral_clustering(CKSym, n):
    """
    Spectral clustering (Ng, Jordan, and Weiss) on graph nodes (GPU-accelerated).

    Args:
        CKSym: N×N adjacency matrix (NumPy or CuPy array).
        n: Number of clusters.

    Returns:
        groups: Length-N vector of cluster memberships (NumPy array).
    """
    # Move to GPU if not already there
    CKSym_gpu = to_gpu(CKSym) if isinstance(CKSym, np.ndarray) else CKSym

    N = CKSym_gpu.shape[0]
    MAXiter = 1000  # KMeans max iterations
    REPlic = 20  # KMeans n_init repetitions

    # Normalized symmetric Laplacian L = I - D^{-1/2} W D^{-1/2} (on GPU)
    DN = xp.diag(1.0 / xp.sqrt(xp.sum(CKSym_gpu, axis=1) + xp.finfo(float).eps))
    LapN = xp.eye(N) - DN @ CKSym_gpu @ DN

    # eigh for symmetric matrices; eigenvalues in ascending order
    # Use the n smallest eigenvectors for clustering (on GPU)
    eigvals, eigvecs = xp.linalg.eigh(LapN)
    kerN = eigvecs[:, :n]  # First n eigenvectors (smallest eigenvalues)

    # Row-normalize eigenvectors (vectorized on GPU)
    norms = xp.linalg.norm(kerN, axis=1, keepdims=True)
    kerNS = kerN / (norms + xp.finfo(float).eps)

    # Transfer back to CPU for KMeans (scikit-learn runs on CPU)
    kerNS_cpu = to_cpu(kerNS)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n, max_iter=MAXiter, n_init=REPlic, random_state=42)
    groups = kmeans.fit_predict(kerNS_cpu) + 1

    return groups
