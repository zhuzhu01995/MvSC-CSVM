"""CSVM-MSC main algorithm - GPU-accelerated version."""
import sys
sys.path.append('..')
from gpu_utils import xp, to_gpu
from .frac_shrink import frac_shrink
from .solve_e_problem import solve_e_problem
from .update_C_fro import update_C_Fusion_fro
from utils import spectral_clustering, compute_nmi, accuracy, compute_f, rand_index


def alg_scvm_msc(X, cls_num, gt, opts=None):
    """
    CSVM-MSC: Consensus Spectral View Modulation multi-view subspace clustering.

    Args:
        X: Data features (list of views, each a NumPy or CuPy array).
        cls_num: Number of clusters.
        gt: Ground-truth labels (NumPy array).
        opts: Optional algorithm parameters (maxIter, epsilon, lambda, gamma, beta, zeta, etc.).

    Returns:
        C: Cluster assignments (NumPy array).
        C_Fusion: Consensus matrix (NumPy or CuPy array).
        Out: Metrics and convergence history.
    """
    # Transfer data to GPU
    X_gpu = to_gpu(X)

    # Parameter setup
    N = X_gpu[0].shape[1]
    K = len(X_gpu)  # Number of views

    # Default parameters
    maxIter = 200
    epsilon = 1e-7
    lambda_val = 0.2
    gamma = 0.001           # Hyperparameter (tune as needed)
    zeta = 0.001            # Hyperparameter (tune as needed)
    beta = 0.001
    alpha = 1e-5            # Lagrange penalty (corresponds to \xi)
    mu = 1e-5               # Lagrange penalty
    eta = 2
    max_mu = 1e10
    max_alpha = 1e10
    flag_debug = 0
    Frac_alpha = 5000

    # Read parameters from opts
    if opts is not None:
        if 'maxIter' in opts:
            maxIter = opts['maxIter']
        if 'epsilon' in opts:
            epsilon = opts['epsilon']
        if 'lambda' in opts:
            lambda_val = opts['lambda']
        if 'gamma' in opts:
            gamma = opts['gamma']
        if 'beta' in opts:
            beta = opts['beta']
        if 'zeta' in opts:
            zeta = opts['zeta']
        if 'mu' in opts:
            mu = opts['mu']
        if 'alpha' in opts:
            alpha = opts['alpha']
        if 'eta' in opts:
            eta = opts['eta']
        if 'max_mu' in opts:
            max_mu = opts['max_mu']
        if 'max_alpha' in opts:
            max_alpha = opts['max_alpha']
        if 'flag_debug' in opts:
            flag_debug = opts['flag_debug']
        if 'Frac_alpha' in opts:
            Frac_alpha = opts['Frac_alpha']

    # Initialize on GPU
    # If C_Fusion and M are all zeros: M stays 0, C subproblem gradient is 0, C_Fusion stays 0,
    # spectral clustering input degenerates and all hyperparameter settings yield the same metrics.
    # Use a non-zero symmetric initialization.
    dtype = X_gpu[0].dtype
    C_Fusion = xp.ones((N, N), dtype=dtype)
    C_Fusion = C_Fusion - xp.eye(N, dtype=dtype)
    C_Fusion = C_Fusion / max(N - 1, 1)
    M = [xp.ones((N, N), dtype=dtype) - xp.eye(N, dtype=dtype) for _ in range(K)]
    Z = [xp.zeros((N, N)) for _ in range(K)]
    B = [xp.zeros((N, N)) for _ in range(K)]
    G = [xp.zeros((N, N)) for _ in range(K)]
    E = [xp.zeros((X_gpu[k].shape[0], N)) for k in range(K)]
    R = [xp.zeros((X_gpu[k].shape[0], N)) for k in range(K)]


    # Initialize history
    history = {
        'norm_Z_G': [],  # ||Z - G||_∞
        'norm_Z': [],
        'norm_Z_Fusion': [],
        'objval': []
    }

    iter_num = 0
    Isconverg = False

    # Main iteration
    while not Isconverg:
        if flag_debug:
            print(f'----processing iter {iter_num + 1}--------')

        # Update Z^k (matrix ops on GPU)
        for k in range(K):
            tmp = (X_gpu[k] - E[k]).T @ R[k] + alpha * (X_gpu[k] - E[k]).T @ (X_gpu[k] - E[k]) - B[k] + mu * G[k] +2*gamma*M[k]*C_Fusion
            Z[k] = xp.linalg.solve(alpha * (X_gpu[k] - E[k]).T @ (X_gpu[k] - E[k]) + (mu + 2*gamma) * xp.eye(N), tmp)

        # Update E^k
        # Prepare proximal gradient inputs
        C_cell = [Z[k] - xp.eye(N) for k in range(K)]
        D_cell = [X_gpu[k] - X_gpu[k] @ Z[k] + R[k] / alpha for k in range(K)]

        # Proximal gradient solver settings
        max_iter_inner = 50  # Max inner iterations
        tol_inner = 1e-5  # Inner tolerance

        # Build initial E_stacked from current E
        E_stacked = xp.vstack(E)

        # Proximal gradient solve (on GPU)
        E_stacked = solve_e_problem(C_cell, D_cell, lambda_val, mu,
                                     max_iter_inner, tol_inner, E_stacked)

        # Split stacked E back into per-view matrices
        start_idx = 0
        for k in range(K):
            d_k = X_gpu[k].shape[0]
            E[k] = E_stacked[start_idx:start_idx + d_k, :]
            start_idx = start_idx + d_k

        # Update G (tensor FFT and SVD on GPU)
        Z_tensor = xp.stack(Z, axis=2)
        B_tensor = xp.stack(B, axis=2)

        G_tensor, objV = frac_shrink(Z_tensor + B_tensor / mu, 6 / mu, 3, Frac_alpha)

        # Update auxiliary variables (on GPU)
        B_tensor = B_tensor + mu * (Z_tensor - G_tensor)
        for k in range(K):
            R[k] = R[k] + alpha * ((X_gpu[k] - E[k]) - (X_gpu[k] - E[k]) @ Z[k])
            G[k] = G_tensor[:, :, k]
            B[k] = B_tensor[:, :, k]

        # Update M^k
        eps = 1e-8
        for k in range(K):
            denom_M = gamma * (C_Fusion * C_Fusion) + zeta +eps
            M[k] = gamma * C_Fusion * Z[k] / denom_M
            M[k] = M[k] - xp.diag(xp.diag(M[k]))


        # Update C
        C_Fusion = update_C_Fusion_fro(Z, M, gamma, beta, remove_diag=True)
        nf = float(xp.linalg.norm(C_Fusion, "fro"))
        if not (nf == nf) or nf < 1e-14:
            C_Fusion = xp.ones((N, N), dtype=dtype) - xp.eye(N, dtype=dtype)
            C_Fusion = C_Fusion / max(N - 1, 1)

        # Record iteration info
        history['objval'].append(objV)

        # Convergence criteria (computed on GPU)
        Isconverg = True

        # Check convergence
        residual_list = [(X_gpu[k] - E[k]) - (X_gpu[k] - E[k]) @ Z[k] for k in range(K)]
        norm_Z = float(max([xp.max(xp.abs(residual)) for residual in residual_list]))
        history['norm_Z'].append(norm_Z)

        if norm_Z > epsilon:
            if flag_debug:
                print(f'norm_Z   {norm_Z:.10f}')
            Isconverg = False

        residual_list = [Z[k] - M[k] * C_Fusion for k in range(K)]
        norm_Z_Fusion = float(max([xp.max(xp.abs(residual)) for residual in residual_list]))
        history['norm_Z_Fusion'].append(norm_Z_Fusion)

        if norm_Z_Fusion > epsilon:
            if flag_debug:
                print(f'norm_Z_Fusion   {norm_Z_Fusion:.10f}')
            Isconverg = False

        norm_Z_G = float(max([xp.max(xp.abs(Z[k] - G[k])) for k in range(K)]))
        history['norm_Z_G'].append(norm_Z_G)

        if norm_Z_G > epsilon:
            if flag_debug:
                print(f'norm_Z_G   {norm_Z_G:.10f}')
            Isconverg = False

        # Check max iterations
        if iter_num >= maxIter:
            Isconverg = True

        # Update penalty parameters
        mu = min(mu * eta, max_mu)
        alpha = min(alpha * eta, max_alpha)

        iter_num += 1

    C = spectral_clustering(C_Fusion, cls_num)
    # Evaluation metrics (on CPU)
    _, nmi, _ = compute_nmi(gt, C)
    ACC = accuracy(C, gt)
    f, p, r = compute_f(gt, C)
    AR, _, _, _ = rand_index(gt, C)

    # Build output
    Out = {
        'NMI': nmi,
        'AR': AR,
        'ACC': ACC,
        'recall': r,
        'precision': p,
        'fscore': f,
        'history': history
    }

    return C, C_Fusion, Out
