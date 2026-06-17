import sys
sys.path.append('..')
from gpu_utils import xp


def update_C_Fusion_fro(Z, M, gamma, beta, remove_diag=True, eps=1e-12):
    """
    C subproblem (Frobenius regularization, closed-form):

        min_{C = C^T}  sum_k γ ||Z^(k) - M^(k) ⊙ C||_F^2  +  β ||C||_F^2

    Hadamard product. Under symmetry, each pair (i,j) and (j,i) (i≠j) is merged into one
    variable c = c_ij = c_ji. The quadratic term from F^2 is β c_ij^2 + β c_ji^2 = 2β c^2;
    after merging the fit terms from both positions, the first-order condition yields a
    closed form. Diagonal entries (i,i) fit the same vector form.

    Let
        A = sum_k γ M^(k) ⊙ Z^(k),    B = sum_k γ M^(k) ⊙ M^(k)   (element-wise)

    Then the global optimum (with symmetric merging) is

        C = (A + A^T) ⊘ (B + B^T + 2β)

    where ⊘ denotes element-wise division (denominator includes eps to avoid division by zero).

    Note: Solving W_ij = A_ij/(B_ij+β) independently per (i,j) without symmetry, then
    projecting with argmin_{C=C^T} ||C-W||_F^2 via (W+W^T)/2, is generally not equivalent
    to the symmetric merged solution above; use (A+A^T)/(B+B^T+2β) here.

    Args:
        Z, M: Lists of length V with N×N arrays, using gpu_utils.xp (CuPy or NumPy).
        gamma, beta: Scalars.
        remove_diag: If True, zero diag(C) after the closed-form step and symmetrize again.
    """
    if len(M) != len(Z):
        raise ValueError("M and Z must have the same length")

    A = xp.zeros_like(Z[0])
    B = xp.zeros_like(Z[0])
    for k in range(len(M)):
        A = A + gamma * (M[k] * Z[k])
        B = B + gamma * (M[k] * M[k])

    den = B + B.T + 2.0 * beta
    C = (A + A.T) / (den + eps)

    if remove_diag:
        C = C - xp.diag(xp.diag(C))
    C = 0.5 * (C + C.T)
    return C
