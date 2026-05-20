"""
后端插件模块
"""

from .base import BackendPlugin, LatencyResult
from .cuda_backend import CUDABackend

__all__ = [
    "BackendPlugin",
    "LatencyResult",
    "CUDABackend",
]