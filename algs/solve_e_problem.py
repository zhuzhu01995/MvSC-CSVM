"""FISTA-accelerated proximal gradient method for the E subproblem - GPU version."""
import sys
sys.path.append('..')
from gpu_utils import xp


def solve_e_problem(C_cell, D_cell, lambda_val, mu, max_iter, epsilon, E_init=None):
    """
    Solve the multi-view E subproblem with FISTA-accelerated proximal gradient (GPU).

    Args:
        C_cell: List {C^{(1)}, C^{(2)}, ..., C^{(V)}}, each C{v} of size N × N (GPU array).
        D_cell: List {D^{(1)}, D^{(2)}, ..., D^{(V)}}, each D{v} of size d_v × N (GPU array).
        lambda_val: Regularization parameter λ.
        mu: Penalty parameter μ (used in ADMM).
        max_iter: Maximum iterations.
        epsilon: Convergence tolerance.
        E_init: Initial stacked matrix (optional).

    Returns:
        E_stacked: Stacked matrix [E^{(1)}; E^{(2)}; ...; E^{(V)}] (GPU array).
    """
    # View count and dimension info
    V = len(C_cell)  # Number of views
    dims = [D.shape[0] for D in D_cell]  # Dimension per view
    N = D_cell[0].shape[1]  # Number of samples
    total_dim = sum(dims)  # Total dimension

    # Precompute starting row index for each view
    row_start = [0] + list(xp.cumsum(xp.array(dims[:-1])).tolist())
    row_end = list(xp.cumsum(xp.array(dims)).tolist())

    # Compute Lipschitz constant (on GPU)
    L_val = 0
    for v in range(V):
        CCT = C_cell[v] @ C_cell[v].T
        norm_CCT = xp.linalg.norm(CCT, 2)

        if norm_CCT > L_val:
            L_val = norm_CCT

    L_val = mu * L_val  # Final Lipschitz constant
    eta = 1 / L_val  # Step size

    # Initialize variables (on GPU)
    if E_init is not None:
        E_k = E_init.copy()
    else:
        E_k = xp.zeros((total_dim, N))

    W_k = E_k.copy()  # Auxiliary variable (for FISTA acceleration)
    t_k = 1  # FISTA momentum parameter

    # Main iteration loop
    for iter_num in range(max_iter):
        # Store previous iterate
        E_prev = E_k.copy()

        # Compute gradient (per view, on GPU)
        G_cell = []  # Gradient per view

        for v in range(V):
            # Extract W_k block for the current view
            W_v = W_k[row_start[v]:row_end[v], :]

            # Gradient: G(v) = μ * (W_v * C{v} + D{v}) * C{v}'
            term = W_v @ C_cell[v] + D_cell[v]
            G_v = mu * (term @ C_cell[v].T)
            G_cell.append(G_v)

        # Stack gradients from all views (on GPU)
        G_stacked = xp.vstack(G_cell)

        # Gradient step: U = W_k - η * G_stacked
        U = W_k - eta * G_stacked

        # Proximal operator: column-wise shrinkage (vectorized on GPU)
        threshold = eta * lambda_val

        # Vectorized: compute norm of each column
        norms = xp.linalg.norm(U, axis=0)  # L2 norm per column

        # Shrinkage factors
        shrink_factors = xp.maximum(0, 1 - threshold / (norms + 1e-12))

        # Apply shrinkage
        E_next = U * shrink_factors[xp.newaxis, :]

        # FISTA acceleration
        t_next = (1 + xp.sqrt(1 + 4 * t_k ** 2)) / 2
        momentum = (t_k - 1) / t_next

        # Update auxiliary variable: W_{k+1} = E_{k+1} + momentum * (E_{k+1} - E_k)
        W_next = E_next + momentum * (E_next - E_k)

        # Update iterates
        E_k = E_next
        W_k = W_next
        t_k = t_next

        # Convergence check (norm on GPU)
        diff_norm = xp.linalg.norm(E_k - E_prev, 'fro')
        if diff_norm < epsilon:
            break

    # Return final solution
    return E_k
