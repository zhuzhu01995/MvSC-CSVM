"""Fractional threshold function - GPU-accelerated version."""
import sys
sys.path.append('..')
from gpu_utils import xp


def glu(la, a, x):
    """
    Fractional threshold function (GPU-accelerated).

    Args:
        la: Lambda parameter.
        a: Alpha parameter.
        x: Input value.

    Returns:
        x0: Output value.
    """
    # Closed-form expression for the fractional threshold function
    f = xp.arccos(-1 + (27 * la * a * a) / (4.0 * (1 + a * xp.abs(x)) ** 3.0))
    x0 = xp.sign(x) * (((1 + a * xp.abs(x)) * (1 + 2 * xp.cos(f / 3 - xp.pi / 3)) - 3) / (3.0 * a))

    return x0
