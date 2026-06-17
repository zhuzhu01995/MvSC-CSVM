"""Fractional shrinkage operator - GPU-accelerated version."""
import sys
sys.path.append('..')
from gpu_utils import xp
from .frac_update_sigma import frac_update_sigma


def frac_shrink(X, lambda_val, mode, a):
    """
    Fractional shrinkage operator (GPU-accelerated).

    Args:
        X: Input tensor (GPU array).
        lambda_val: Lambda parameter.
        mode: Mode (3 for tensor mode).
        a: Alpha parameter.

    Returns:
        X: Processed tensor.
        objV: Objective value.
    """
    sX = X.shape

    if mode == 3:
        Y = xp.moveaxis(X, 0, -1)
        Y = xp.moveaxis(Y, 0, 1)
    else:
        Y = X.copy()

    # FFT transform (on GPU)
    Yhat = xp.fft.fft(Y, axis=2)

    objV = 0
    if mode == 3:
        n3 = sX[0]
        m = min(sX[1], sX[2])
    else:
        n3 = sX[2]
        m = min(sX[0], sX[1])

    # SVD on each frontal slice (on GPU)
    for i in range(n3):
        uhat, shat, vhat = xp.linalg.svd(Yhat[:, :, i], full_matrices=False)

        # Update singular values
        for j in range(m):
            shat[j] = frac_update_sigma(shat[j], lambda_val, a)

        # Compute objective value
        objV += xp.sum((a * xp.abs(shat)) / (1 + a * xp.abs(shat)))

        # Reconstruct matrix
        Yhat[:, :, i] = uhat @ xp.diag(shat) @ vhat

    # Inverse FFT transform (on GPU)
    Y = xp.fft.ifft(Yhat, axis=2).real

    if mode == 3:
        Y = xp.moveaxis(Y, 1, 0)
        X = xp.moveaxis(Y, -1, 0)
    else:
        X = Y

    # Convert scalar to Python type
    if hasattr(objV, 'item'):
        objV = objV.item()

    return X, objV
