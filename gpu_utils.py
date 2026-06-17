"""GPU utility module - auto-detect GPU with CPU fallback."""
import warnings

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU detected, using CuPy for GPU acceleration")
except Exception:
    cp = np
    GPU_AVAILABLE = False
    warnings.warn("⚠ CuPy unavailable, running with NumPy (CPU mode)")


def get_array_module(x=None):
    """
    Get the array module (cupy or numpy).

    Args:
        x: Optional array object for auto-detection.

    Returns:
        The appropriate array module (cupy or numpy).
    """
    if GPU_AVAILABLE:
        if x is None:
            return cp
        return cp.get_array_module(x)
    return np


def to_gpu(x):
    """
    Transfer data to GPU.

    Args:
        x: NumPy array or list.

    Returns:
        GPU array (if available) or original NumPy array.
    """
    if not GPU_AVAILABLE:
        return x

    if isinstance(x, list):
        return [cp.asarray(item) for item in x]
    else:
        return cp.asarray(x)


def to_cpu(x):
    """
    Transfer data to CPU.

    Args:
        x: CuPy array, NumPy array, or list.

    Returns:
        NumPy array.
    """
    if isinstance(x, list):
        result = []
        for item in x:
            if GPU_AVAILABLE and isinstance(item, cp.ndarray):
                result.append(cp.asnumpy(item))
            else:
                result.append(np.asarray(item))
        return result
    else:
        if GPU_AVAILABLE and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        else:
            return np.asarray(x)


def get_gpu_info():
    """Get GPU information."""
    if not GPU_AVAILABLE:
        return "GPU unavailable - using CPU mode"

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        memory_info = device.mem_info
        free_mem = memory_info[0] / 1024**3  # GB
        total_mem = memory_info[1] / 1024**3  # GB

        info = f"""GPU info:
        Device: {props['name'].decode()}
        Compute capability: {props['major']}.{props['minor']}
        Total memory: {total_mem:.2f} GB
        Free memory: {free_mem:.2f} GB
        Multiprocessors: {props['multiProcessorCount']}
        """
        return info
    except Exception as e:
        return f"Unable to get GPU info: {str(e)}"


# Export common array creation helpers (GPU-first)
xp = cp  # Primary array module

__all__ = [
    'xp', 'cp', 'np',
    'GPU_AVAILABLE',
    'get_array_module',
    'to_gpu',
    'to_cpu',
    'get_gpu_info'
]
