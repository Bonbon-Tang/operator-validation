"""
operator_validation
~~~~~~~~~~~~~~~~~~~
Multi-backend operator validation framework for AI accelerators.

以 NVIDIA H100 为基线，对比寒武纪 MLU590、FlagOS Triton 等后端的
算子精度与性能。

Usage:
    from operator_validation import ValidatorEngine, load_config
    
    engine = ValidatorEngine.from_config("configs/h100_vs_mlu590.yaml")
    results = engine.run_all()
    engine.generate_report("report.json")
"""

__version__ = "0.1.0"
__author__ = "Your Team"

from .config import ValidationConfig, BackendInfo, OperatorDef, load_config
from .engine.validator import ValidatorEngine
from .engine.reporter import ReportGenerator
from .backend.registry import PluginRegistry
from .operators.registry import OperatorRegistry

__all__ = [
    "ValidatorEngine",
    "ValidationConfig", 
    "BackendInfo",
    "OperatorDef",
    "load_config",
    "ReportGenerator",
    "PluginRegistry",
    "OperatorRegistry",
]
