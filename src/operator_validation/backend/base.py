"""
后端插件抽象基类
"""

from __future__ import annotations

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from ..config import BackendInfo


@dataclass
class LatencyResult:
    """延迟测试结果"""
    latency_ms: float
    std_ms: float
    throughput_gflops: float


class BackendPlugin(ABC):
    """
    后端插件抽象基类
    
    所有后端（CUDA、MLU、 Triton 等）必须实现此接口。
    
    实现要点：
    1. __init__ 中完成设备初始化
    2. exec_operator 中实现算子调用逻辑
    3. synchronize 确保测试同步
    
    Example:
        class MyBackend(BackendPlugin):
            def _init_device(self) -> torch.device:
                return torch.device("mlu:0")
            
            def exec_operator(self, aten_name: str, inputs, kwargs) -> torch.Tensor:
                if aten_name == "matmul":
                    return torch.bmm(inputs[0], inputs[1])
                ...
            
            def synchronize(self):
                torch.mlu.synchronize()
    """

    def __init__(self, info: BackendInfo):
        """
        Args:
            info: 后端配置信息
        """
        self.info = info
        self._device = self._init_device()

    @abstractmethod
    def _init_device(self) -> torch.device:
        """
        初始化并返回 torch.device
        
        Returns:
            torch.device 对象
        """
        pass

    @abstractmethod
    def exec_operator(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        执行算子
        
        Args:
            aten_name: ATen 算子名称（如 "matmul", "relu", "softmax"）
            inputs: 输入 tensors 列表
            kwargs: 额外参数（如 {"padding": 1, "dim": -1}）
            
        Returns:
            输出 tensor
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """同步等待，确保算子执行完成"""
        pass

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 tensor 迁移到本后端设备"""
        return tensor.to(self._device)

    def create_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        创建随机 tensor
        
        Args:
            shape: 张量形状
            dtype: 数据类型，默认使用后端配置的类型
        """
        if dtype is None:
            dtype = self.info.dtype
        return torch.randn(shape, dtype=dtype, device=self._device)

    @property
    def device(self) -> torch.device:
        """返回本后端的设备"""
        return self._device

    @property
    def name(self) -> str:
        """返回后端名称"""
        return self.info.name

    @property
    def is_available(self) -> bool:
        """检查后端是否可用"""
        try:
            t = self.create_tensor((2, 2))
            return True
        except Exception:
            return False
