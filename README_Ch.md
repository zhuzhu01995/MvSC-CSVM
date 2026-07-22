# CSVM-MSC GPU 加速实现

**CSVM-MSC**（Consensus Spectral View Modulation Multi-view Subspace Clustering）多视图子空间聚类算法的 **GPU 加速 Python 实现**。核心数值计算基于 [CuPy](https://cupy.dev/)，在无 GPU 或未安装 CuPy 时自动回退到 NumPy CPU 模式。

## 特性

- **GPU 加速**：矩阵运算、FFT/IFFT、SVD 等均在 GPU 上执行（CuPy / cuFFT / cuSOLVER）
- **自动回退**：检测不到 CuPy 时使用 NumPy，无需改代码
- **聚类测试脚本**：`test_CSVM.py` 支持多数据集、多次运行与统计汇总
- **评估指标**：ACC、NMI、AR、Precision、Recall、F-score

## 环境要求

| 组件 | 说明 |
|------|------|
| Python | 3.8+ |
| 必需 | NumPy 1.x、SciPy、scikit-learn |
| 可选（推荐） | NVIDIA GPU + CUDA 12.x + CuPy 13.x |

> **注意**：请保持 `numpy<2.0`。CuPy 14.x 会强制升级 NumPy 2.x，在本项目环境下可能导致崩溃。

## 安装

```bash
# 克隆或进入项目目录
cd MvSC_CSVM_GPU

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 安装基础依赖
pip install "numpy<2.0" scipy scikit-learn
```

### GPU 支持（可选）

```bash
# 固定 NumPy 1.x 后再安装 CuPy（CUDA 12.x 示例）
pip install "numpy==1.26.4"
pip install "cupy-cuda12x==13.6.0" fastrlock

# 验证
python -c "import cupy; print(cupy.__version__)"
```

其他 CUDA 版本请参考 [CuPy 安装文档](https://docs.cupy.dev/en/stable/install.html)。

## 数据准备

将 `.mat` 格式数据集放入项目根目录下的 `data/` 文件夹（需自行创建）：

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

每个数据文件应包含多视图特征 `X1, X2, ...` 及真实标签 `gt`（或 `gnd` / `y`）。

`test_CSVM.py` 从本地 `data/` 目录加载数据（若存在 `../MvSC_CSVM_python_GPU/data/` 则优先使用该路径）。

## 项目结构

```
MvSC_CSVM_GPU/
├── algs/                          # 算法核心
│   ├── alg_scvm_msc.py            # 主算法入口
│   ├── frac_shrink.py             # 分数阶张量收缩
│   ├── frac_update_sigma.py       # 奇异值更新
│   ├── glu.py                     # 分数阈值函数
│   ├── solve_e_problem.py         # 噪声矩阵 E 子问题（FISTA）
│   ├── update_C.py                # 共识矩阵 C（核范数）
│   └── update_C_fro.py            # 共识矩阵 C（Frobenius）
├── utils/                         # 工具函数
│   ├── normalize_data.py          # 数据归一化
│   ├── spectral_clustering.py     # 谱聚类
│   ├── accuracy.py                # ACC
│   ├── compute_nmi.py             # NMI
│   ├── compute_f.py               # F-score / Precision / Recall
│   ├── rand_index.py              # AR
│   ├── best_map.py                # 标签映射
│   ├── hungarian.py               # 匈牙利算法
│   └── contingency.py             # 列联表
├── gpu_utils.py                   # GPU 检测与数组转换
└── test_CSVM.py                   # 主测试脚本（多数据集、多次运行）
```

## 快速开始

### 1. 运行测试

编辑 `test_CSVM.py` 顶部的配置区域（`test_list`、`num_runs`），然后运行：

```bash
python test_CSVM.py
```

数据集索引：`0`=Yale，`1`=Extended YaleB，`2`=ORL，`3`=Notting Hill，`4`=COIL-20，`5`=UCI Digits。

### 2. 代码调用示例

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
    "lambda": 0.3,       # 噪声稀疏正则
    "gamma": 0.0001,     # 视图调制正则
    "beta": 0.0001,      # 共识矩阵 C 正则
    "zeta": 0.1,         # M^(v) 正则
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

## 数据集与默认超参数

以下参数已在 `test_CSVM.py` 中配置，可直接用于复现实验。

| 索引 | 数据集 | 样本数 | 视图数 | 类别数 | λ | γ | β | ζ |
|:----:|--------|-------:|-------:|-------:|----:|----:|----:|----:|
| 0 | Yale | 165 | 3 | 15 | 0.3 | 1e-4 | 1e-4 | 0.1 |
| 1 | Extended YaleB | 2414 | 3 | 38 | 0.002 | 2.5e-4 | 1e-4 | 2.5e-4 |
| 2 | ORL | 400 | 3 | 40 | 0.01 | 1e-4 | 1e-4 | 0.01 |
| 3 | Notting Hill | 4660 | 3 | 5 | 0.01 | 0.1 | 1e-4 | 1e-4 |
| 4 | COIL-20 | 1440 | 3 | 20 | 0.1 | 0.1 | 1.0 | 0.01 |
| 5 | UCI Digits | 2000 | 6 | 10 | 2e-4 | 2e-4 | 1e-4 | 0.001 |

## GPU 工具 API

```python
from gpu_utils import xp, to_gpu, to_cpu, GPU_AVAILABLE, get_gpu_info

# xp 自动指向 cupy 或 numpy
arr = xp.zeros((100, 100))

gpu_arr = to_gpu(cpu_arr)   # CPU → GPU
cpu_arr = to_cpu(gpu_arr)   # GPU → CPU

if GPU_AVAILABLE:
    print(get_gpu_info())
```

## 算法参数说明

`alg_scvm_msc(X, cls_num, gt, opts)` 中 `opts` 常用字段：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `lambda` | 噪声矩阵 E 的稀疏正则 | 0.2 |
| `gamma` | 视图调制项权重 | 0.001 |
| `beta` | 共识矩阵 C 的 Frobenius 正则 | 0.001 |
| `zeta` | 辅助变量 M^(v) 的正则 | 0.001 |
| `Frac_alpha` | 分数阶收缩参数 | 5000 |
| `maxIter` | 最大迭代次数 | 200 |
| `epsilon` | 收敛阈值 | 1e-7 |
| `mu`, `eta`, `max_mu` | ADMM 惩罚参数及其更新策略 | — |
| `alpha`, `max_alpha` | 额外 Lagrange 罚参数 | 1e-5 / 1e10 |

返回值：`C` 为聚类标签，`C_Fusion` 为共识矩阵，`Out` 包含评估指标与收敛历史。

## 故障排除

**CuPy 安装失败**

```bash
nvidia-smi          # 确认驱动与 CUDA 版本
pip install "numpy==1.26.4"
pip install "cupy-cuda12x==13.6.0"
```

**GPU 内存不足**

- 先用 Yale 等小数据集验证
- 关闭其他占用 GPU 的进程
- 减小 `maxIter` 或在 CPU 模式下运行

**数据文件未找到**

- 确认 `data/` 目录存在且包含对应 `.mat` 文件

**CPU 回退提示**

启动时若看到 `CuPy unavailable, running with NumPy (CPU mode)`，说明未检测到 GPU 加速，算法仍可正常运行，但速度较慢。

## 许可证

MIT License

---

在配备 NVIDIA GPU 的系统上，相对纯 CPU 实现通常可获得 **10–50 倍**加速（取决于数据集规模与 GPU 型号）。无 GPU 时亦可完整运行测试脚本。
