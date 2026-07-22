# CSVM-MSC GPU-Accelerated Implementation

**CSVM-MSC** (Consensus Spectral View Modulation Multi-view Subspace Clustering) — a **GPU-accelerated Python implementation** of the multi-view subspace clustering algorithm. Core numerical computation is powered by [CuPy](https://cupy.dev/); when no GPU is available or CuPy is not installed, it automatically falls back to NumPy CPU mode.

## Features

- **GPU acceleration**: Matrix operations, FFT/IFFT, SVD, etc. run on the GPU (CuPy / cuFFT / cuSOLVER)
- **Automatic fallback**: Uses NumPy when CuPy is unavailable — no code changes required
- **Clustering test script**: `test_CSVM.py` supports multiple datasets, multiple runs, and statistical summaries
- **Evaluation metrics**: ACC, NMI, AR, Precision, Recall, F-score

## Requirements

| Component | Description |
|-----------|-------------|
| Python | 3.8+ |
| Required | NumPy 1.x, SciPy, scikit-learn |
| Optional (recommended) | NVIDIA GPU + CUDA 12.x + CuPy 13.x |

> **Note**: Keep `numpy<2.0`. CuPy 14.x forces an upgrade to NumPy 2.x, which may cause crashes in this project’s environment.

## Installation

```bash
# Clone or enter the project directory
cd MvSC_CSVM_GPU

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install base dependencies
pip install "numpy<2.0" scipy scikit-learn
```

### GPU Support (Optional)

```bash
# Pin NumPy 1.x before installing CuPy (CUDA 12.x example)
pip install "numpy==1.26.4"
pip install "cupy-cuda12x==13.6.0" fastrlock

# Verify
python -c "import cupy; print(cupy.__version__)"
```

For other CUDA versions, see the [CuPy installation docs](https://docs.cupy.dev/en/stable/install.html).

## Data Preparation

Place `.mat` datasets in the `data/` folder under the project root (create it if needed):

```
MvSC_CSVM_GPU/
└── data/
    ├── yale.mat
    ├── yaleB.mat
    ├── ORL.mat
    ├── NH.mat
    ├── COIL20MV.mat
    └── UCI_Digits_X_gt_LogDetFormat.mat
```

Each data file should contain multi-view features `X1, X2, ...` and ground-truth labels `gt` (or `gnd` / `y`).

`test_CSVM.py` loads datasets from the local `data/` directory (or `../MvSC_CSVM_python_GPU/data/` if present).

## Project Structure

```
MvSC_CSVM_GPU/
├── algs/                          # Core algorithms
│   ├── alg_scvm_msc.py            # Main algorithm entry point
│   ├── frac_shrink.py             # Fractional-order tensor shrinkage
│   ├── frac_update_sigma.py       # Singular value update
│   ├── glu.py                     # Fractional threshold function
│   ├── solve_e_problem.py         # Noise matrix E subproblem (FISTA)
│   ├── update_C.py                # Consensus matrix C (nuclear norm)
│   └── update_C_fro.py            # Consensus matrix C (Frobenius)
├── utils/                         # Utility functions
│   ├── normalize_data.py          # Data normalization
│   ├── spectral_clustering.py     # Spectral clustering
│   ├── accuracy.py                # ACC
│   ├── compute_nmi.py             # NMI
│   ├── compute_f.py               # F-score / Precision / Recall
│   ├── rand_index.py              # AR
│   ├── best_map.py                # Label mapping
│   ├── hungarian.py               # Hungarian algorithm
│   └── contingency.py             # Contingency table
├── gpu_utils.py                   # GPU detection and array conversion
└── test_CSVM.py                   # Main test script (multi-dataset, multi-run)
```

## Quick Start

### 1. Run Tests

Edit the configuration block at the top of `test_CSVM.py` (`test_list`, `num_runs`), then run:

```bash
python test_CSVM.py
```

Dataset indices: `0`=Yale, `1`=Extended YaleB, `2`=ORL, `3`=Notting Hill, `4`=COIL-20, `5`=UCI Digits.

### 2. Code Example

```python
import numpy as np
from scipy.io import loadmat

from utils import normalize_data
from algs import alg_scvm_msc
from gpu_utils import to_gpu, get_gpu_info

print(get_gpu_info())

data = loadmat("data/yale.mat")
X = [data["X1"], data["X2"], data["X3"]]
gt = data["gt"].flatten()

Y = [to_gpu(normalize_data(x)) for x in X]

opts = {
    "lambda": 0.3,       # Noise sparsity regularization
    "gamma": 0.0001,     # View modulation regularization
    "beta": 0.0001,      # Consensus matrix C regularization
    "zeta": 0.1,         # M^(v) regularization
    "Frac_alpha": 5000,
    "maxIter": 60,
    "epsilon": 1e-4,
    "mu": 1e-5,
    "eta": 2,
    "max_mu": 1e10,
}

cls_num = len(np.unique(gt))
C, C_Fusion, Out = alg_scvm_msc(Y, cls_num, gt, opts)

print(f"ACC: {Out['ACC']:.4f}")
print(f"NMI: {Out['NMI']:.4f}")
print(f"F-score: {Out['fscore']:.4f}")
```

## Datasets and Default Hyperparameters

The following parameters are configured in `test_CSVM.py` and can be used directly to reproduce experiments.

| Index | Dataset | Samples | Views | Classes | λ | γ | β | ζ |
|:-----:|---------|--------:|------:|------:|----:|----:|----:|----:|
| 0 | Yale | 165 | 3 | 15 | 0.3 | 1e-4 | 1e-4 | 0.1 |
| 1 | Extended YaleB | 2414 | 3 | 38 | 0.002 | 2.5e-4 | 1e-4 | 2.5e-4 |
| 2 | ORL | 400 | 3 | 40 | 0.01 | 1e-4 | 1e-4 | 0.01 |
| 3 | Notting Hill | 4660 | 3 | 5 | 0.00625 | 0.1 | 0.000125 | 1e-4 |
| 4 | COIL-20 | 1440 | 3 | 20 | 0.1 | 0.1 | 1.0 | 0.01 |
| 5 | UCI Digits | 2000 | 6 | 10 | 2e-4 | 2e-4 | 1e-4 | 0.001 |

## GPU Utility API

```python
from gpu_utils import xp, to_gpu, to_cpu, GPU_AVAILABLE, get_gpu_info

# xp automatically points to cupy or numpy
arr = xp.zeros((100, 100))

gpu_arr = to_gpu(cpu_arr)   # CPU → GPU
cpu_arr = to_cpu(gpu_arr)   # GPU → CPU

if GPU_AVAILABLE:
    print(get_gpu_info())
```

## Algorithm Parameters

Common fields in `opts` for `alg_scvm_msc(X, cls_num, gt, opts)`:

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `lambda` | Sparsity regularization for noise matrix E | 0.2 |
| `gamma` | Weight for view modulation term | 0.001 |
| `beta` | Frobenius regularization for consensus matrix C | 0.001 |
| `zeta` | Regularization for auxiliary variable M^(v) | 0.001 |
| `Frac_alpha` | Fractional shrinkage parameter | 5000 |
| `maxIter` | Maximum iterations | 60 |
| `epsilon` | Convergence threshold | 1e-7 |
| `mu`, `eta`, `max_mu` | ADMM penalty parameters and update strategy | — |
| `alpha`, `max_alpha` | Additional Lagrange penalty parameters | 1e-5 / 1e10 |

Returns: `C` is the cluster labels, `C_Fusion` is the consensus matrix, and `Out` contains evaluation metrics and convergence history.

## Troubleshooting

**CuPy installation fails**

```bash
nvidia-smi          # Check driver and CUDA version
pip install "numpy==1.26.4"
pip install "cupy-cuda12x==13.6.0"
```

**Out of GPU memory**

- Validate with small datasets such as Yale first
- Close other processes using the GPU
- Reduce `maxIter` or run in CPU mode

**Data files not found**

- Ensure the `data/` directory exists and contains the corresponding `.mat` files

**CPU fallback message**

If you see `CuPy unavailable, running with NumPy (CPU mode)` at startup, GPU acceleration was not detected. The algorithm still runs correctly, but more slowly.

## License

MIT License

---

On systems with an NVIDIA GPU, this implementation typically achieves **10–50×** speedup over a pure CPU implementation (depending on dataset size and GPU model). The full test script also runs without a GPU.
