"""
配置加载与数据结构定义
"""

from __future__ import annotations

import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from enum import Enum


class DeviceType(Enum):
    """设备类型枚举"""
    CUDA = "cuda"
    MLU = "mlu"
    NPU = "npu"
    CPU = "cpu"
    CUSTOM = "custom"


class PrecisionMetric(Enum):
    """精度指标类型"""
    MSE = "mse"
    MAX_ABS_ERR = "max_abs_err"
    MAX_REL_ERR = "max_rel_err"
    COSINE_SIM = "cosine_sim"


@dataclass
class BackendInfo:
    """
    后端配置信息
    
    Attributes:
        name: 后端唯一标识符
        device_id: 设备 ID，如 "0"、"1"
        device_type: 设备类型 (CUDA/MLU/NPU/CPU)
        dtype: 计算精度
        vendor: 厂商名称
        priority: 优先级，0 = reference backend（所有对比的基准）
        config: 后端特定配置字典
        enabled: 是否启用
    """
    name: str
    device_id: str
    device_type: DeviceType
    dtype: torch.dtype = torch.float32
    vendor: str = "unknown"
    priority: int = 99
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, d: Dict) -> "BackendInfo":
        """从字典构造 BackendInfo"""
        device_str = d.get("device", f"{name}:0")
        if ":" in device_str:
            dtype_part, device_id = device_str.split(":")
            device_type = DeviceType(dtype_part)
        else:
            device_type = DeviceType.CUDA
            device_id = device_str

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype_str = d.get("dtype", "float32")
        
        return cls(
            name=name,
            device_id=device_id,
            device_type=device_type,
            dtype=dtype_map.get(dtype_str, torch.float32),
            vendor=d.get("vendor", "unknown"),
            priority=d.get("priority", 99),
            config=d.get("config", {}),
            enabled=d.get("enabled", True),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OperatorDef:
    """
    算子定义
    
    Attributes:
        name: 算子名称（唯一标识）
        aten_alias: ATen 函数名（映射到 torch.xxx）
        shapes: 不同规模的输入 shapes
        kwargs: 额外参数（如 padding、dim 等）
    """
    name: str
    aten_alias: str
    shapes: Dict[str, List[List[int]]] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> "OperatorDef":
        return cls(
            name=d["name"],
            aten_alias=d.get("aten_alias", d["name"]),
            shapes=d.get("shapes", {}),
            kwargs=d.get("kwargs", {}),
        )


@dataclass
class ValidationConfig:
    """
    验证配置
    
    Attributes:
        rtol: 相对误差容限 (relative tolerance)
        atol: 绝对误差容限 (absolute tolerance)  
        cosine_threshold: Cosine similarity 阈值
        warmup_iter: 预热迭代次数
        bench_iter: 正式测试迭代次数
        skip_large: 是否跳过大规模测试（省时间）
    """
    rtol: float = 1e-3
    atol: float = 1e-4
    cosine_threshold: float = 0.9999
    warmup_iter: int = 20
    bench_iter: int = 100
    skip_large: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "ValidationConfig":
        return cls(
            rtol=d.get("rtol", 1e-3),
            atol=d.get("atol", 1e-4),
            cosine_threshold=d.get("cosine_threshold", 0.9999),
            warmup_iter=d.get("warmup_iter", 20),
            bench_iter=d.get("bench_iter", 100),
            skip_large=d.get("skip_large", False),
        )


@dataclass
class OutputConfig:
    """输出配置"""
    report_path: str = "validation_report.json"
    log_level: str = "INFO"
    verbose: bool = True


@dataclass
class FullConfig:
    """完整配置"""
    backends: List[BackendInfo] = field(default_factory=list)
    operators: List[OperatorDef] = field(default_factory=list)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "FullConfig":
        """从 YAML 文件加载配置"""
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, d: Dict) -> "FullConfig":
        """从字典加载配置"""
        backend_list = []
        for name, info in d.get("backends", {}).items():
            if info.get("enabled", True):
                backend_list.append(BackendInfo.from_dict(name, info))

        operator_list = [
            OperatorDef.from_dict(op) for op in d.get("operators", [])
        ]

        validation = ValidationConfig.from_dict(d.get("validation", {}))
        output = OutputConfig(**d.get("output", {}))

        return cls(
            backends=backend_list,
            operators=operator_list,
            validation=validation,
            output=output,
        )


def load_config(path: str | Path) -> FullConfig:
    """
    加载配置文件
    
    Args:
        path: YAML 配置文件路径
        
    Returns:
        FullConfig 对象
        
    Example:
        >>> config = load_config("configs/h100_vs_mlu590.yaml")
        >>> print(config.backends)
    """
    return FullConfig.from_file(path)
