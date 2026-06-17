"""CSVM-MSC GPU-accelerated package."""
from .gpu_utils import GPU_AVAILABLE, get_gpu_info

__version__ = '1.0.0-gpu'
__all__ = ['GPU_AVAILABLE', 'get_gpu_info']
