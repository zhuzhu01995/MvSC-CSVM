"""Fractional sigma update function - GPU-accelerated version."""
import sys
sys.path.append('..')
from gpu_utils import xp
from .glu import glu


def frac_update_sigma(sigma, lambda_val, a):
    """
    Update singular values (GPU-accelerated).

    Args:
        sigma: Input singular value.
        lambda_val: Lambda parameter.
        a: Alpha parameter.

    Returns:
        s_sigma: Updated singular value.
    """
    lambda_mu = lambda_val

    # Compute threshold
    a_sq = a * a
    if lambda_mu <= 1 / a_sq:
        t_star = (lambda_mu * a) / 2
    else:
        t_star = xp.sqrt(lambda_mu) - 1 / (2 * a)

    # Apply thresholding rule
    if xp.abs(sigma) <= t_star:
        s_sigma = 0
    else:
        # Call glu for the non-zero solution
        s_sigma = glu(lambda_mu, a, sigma)

    return s_sigma
