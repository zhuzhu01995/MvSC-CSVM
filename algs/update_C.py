import sys
sys.path.append('..')
from gpu_utils import xp

# See also update_C_fro.py: closed-form when β||C||_* is replaced by β||C||_F^2 in the same C subproblem
#   min_{C=C^T} Σ_k γ||Z^k - M^k⊙C||_F^2 + β||C||_F^2

def prox_nuclear_symmetric(W, tau):
    # W is symmetrized; eigh avoids unnecessary complex results
    eigvals, Q = xp.linalg.eigh(W)
    eigvals_shrink = xp.sign(eigvals) * xp.maximum(xp.abs(eigvals) - tau, 0.0)

    return Q @ xp.diag(eigvals_shrink) @ Q.T

def update_C_Fusion_nuclear( C, Z, M, gamma, beta, inner_iter, tol, remove_diag):
    """
    (Nuclear-norm regularization version; for the weaker Frobenius closed-form, see update_C_fro.update_C_Fusion_fro)

    Solve:
       min_C sum_k gamma * ||Z[k]-M[k]*C_Fusion||_F^2 + beta *||C||_*
         s.t.   C_Fusion = C_Fusion^T, diag(C_Fusion) = 0
    where * denotes the Hadamard (element-wise) product.
    """
    eps = 1e-8
    # Lipschitz constant estimate
    MM_sum = xp.zeros_like(C)
    for k in range(len(M)):
        MM_sum += M[k] * M[k]

    mm_max = float(xp.max(MM_sum))
    # When M is very small, L becomes too small -> tau_C explodes -> C zeroed in one step,
    # spectral clustering input degenerates and metrics do not vary with hyperparameters
    L = 2.0 * gamma * max(mm_max, 1.0) + eps
    tau_C = 1.0 / L

    for _ in range(inner_iter):
        C_prev =C.copy()
        grad = xp.zeros_like(C)

        for k in range(len(M)):
            grad += M[k] * (M[k] * C -Z[k])
        grad = 2.0 * gamma *grad

        # gradient step
        W = C - tau_C * grad

        # Symmetrize
        W = 0.5 * (W + W.T)

        # Symmetric nuclear-norm proximal step
        C = prox_nuclear_symmetric(W, tau_C*beta)

        # Remove diagonal
        if remove_diag:
            C = C - xp.diag(xp.diag(C))

        # Symmetrize again to reduce numerical error
        C = 0.5 * (C + C.T)

        # Convergence check
        diff = float(xp.max(xp.abs(C - C_prev)))
        if diff < tol:
            break

    return C
