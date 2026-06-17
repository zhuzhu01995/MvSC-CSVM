"""Best label mapping - CPU version."""
import numpy as np
from .hungarian import hungarian


def best_map(L1, L2):
    """
    Permute labels in L2 to best match L1.

    Args:
        L1: First label vector (NumPy array).
        L2: Second label vector (NumPy array).

    Returns:
        newL2: Remapped L2 labels (NumPy array).

    Note: Runs on CPU.
    """
    L1 = L1.flatten()
    L2 = L2.flatten()

    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))

    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))

    c, _ = hungarian(-G)
    newL2 = np.zeros_like(L2)

    # c[i] is the cluster index assigned to true class i;
    # map cluster Label2[c[i]] to true class Label1[i]
    for i in range(nClass1):
        if i < len(c):  # Guard against index out of bounds
            newL2[L2 == Label2[c[i]]] = Label1[i]

    return newL2
