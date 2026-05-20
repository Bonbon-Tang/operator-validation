"""验证引擎 - 核心调度层"""
from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from ..config import FullConfig, ValidationConfig, BackendInfo, OperatorDef
from ..backend.base import BackendPlugin
from ..backend.registry import PluginRegistry
from .executor import OperatorExecutor
from ..utils.metrics import compute_accuracy

@dataclass
class TestCaseResult:
    operator: str = ""
    shape: List[int] = field(default_factory=list)
    backend: str = ""
    reference_output: Optional[np.ndarray] = None
    output: Optional[np.ndarray] = None
    mse: float = 0.0
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    cosine_sim: float = 0.0
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput_gflops: float = 0.0
    status: str = "UNKNOWN"
    error: str = ""
    latency_ratio_vs_ref: float = 0.0

@dataclass
class OperatorResult:
    operator: str = ""
    cases: List[TestCaseResult] = field(default_factory=list)

class ValidatorEngine:
    def __init__(self, config: FullConfig):
        self.config = config
        self.backends: Dict[str, BackendPlugin] = {}
        self.executors: Dict[str, OperatorExecutor] = {}
        self.results: List[TestCaseResult] = []
        self._results_by_op: Dict[str, OperatorResult] = {}
        self._init_backends()
        self.reference_backend: Optional[BackendPlugin] = None
        for name, backend in self.backends.items():
            if backend.info.priority == 0:
                self.reference_backend = backend
                break
        if self.reference_backend is None:
            raise ValueError("配置中必须有一个 priority=0 的 reference backend")

    def _init_backends(self):
        for info in self.config.backends:
            if not info.enabled:
                continue
            try:
                backend = PluginRegistry.create(info.name, info)
                executor = OperatorExecutor(backend, self.config.validation.warmup_iter, self.config.validation.bench_iter)
                self.backends[info.name] = backend
                self.executors[info.name] = executor
            except Exception as e:
                print(f"  ⚠️ 后端 {info.name} 初始化失败: {e}")

    @classmethod
    def from_config(cls, config_path: str) -> "ValidatorEngine":
        config = FullConfig.from_file(config_path)
        return cls(config)

    def _judge_status(self, mse: float, max_rel_err: float, max_abs_err: float, cosine_sim: float) -> str:
        rtol = self.config.validation.rtol
        atol = self.config.validation.atol
        cos_thresh = self.config.validation.cosine_threshold
        if (max_rel_err < rtol and max_abs_err < atol and cosine_sim > cos_thresh):
            return "PASS"
        elif (max_rel_err < 10 * rtol and max_abs_err < 10 * atol):
            return "WARNING"
        else:
            return "FAIL"

    def validate_operator(self, op_def: OperatorDef, shape: List[int]) -> OperatorResult:
        result = OperatorResult(operator=op_def.name)
        ref_executor = self.executors.get(self.reference_backend.name)
        if ref_executor is None:
            return result
        ref_output = None
        ref_latency = None
        try:
            ref_output, ref_latency = ref_executor.execute(op_def.aten_alias, [tuple(shape)], op_def.kwargs)
            ref_case = TestCaseResult(
                operator=op_def.name, shape=shape, backend=self.reference_backend.name,
                reference_output=ref_output, output=ref_output,
                latency_ms=ref_latency.latency_ms, latency_std_ms=ref_latency.std_ms,
                throughput_gflops=ref_latency.throughput_gflops, status="REFERENCE"
            )
            result.cases.append(ref_case)
        except Exception as e:
            ref_case = TestCaseResult(operator=op_def.name, shape=shape, backend=self.reference_backend.name, status="SKIP", error=f"Reference backend 失败: {e}")
            result.cases.append(ref_case)
            return result
        for backend_name, executor in self.executors.items():
            if backend_name == self.reference_backend.name:
                continue
            try:
                output, latency = executor.execute(op_def.aten_alias, [tuple(shape)], op_def.kwargs)
                metrics = compute_accuracy(output, ref_output)
                status = self._judge_status(metrics["mse"], metrics["max_rel_err"], metrics["max_abs_err"], metrics["cosine_sim"])
                latency_ratio = latency.latency_ms / ref_latency.latency_ms if ref_latency and ref_latency.latency_ms > 0 else 0
                case = TestCaseResult(
                    operator=op_def.name, shape=shape, backend=backend_name,
                    reference_output=ref_output, output=output,
                    mse=metrics["mse"], max_abs_err=metrics["max_abs_err"],
                    max_rel_err=metrics["max_rel_err"], cosine_sim=metrics["cosine_sim"],
                    latency_ms=latency.latency_ms, latency_std_ms=latency.std_ms,
                    throughput_gflops=latency.throughput_gflops,
                    status=status, latency_ratio_vs_ref=latency_ratio
                )
                result.cases.append(case)
            except Exception as e:
                case = TestCaseResult(operator=op_def.name, shape=shape, backend=backend_name, status="FAIL", error=str(e))
                result.cases.append(case)
        return result

    def run_all(self, operators: Optional[List[str]] = None, shapes: Optional[List[str]] = None) -> Dict[str, OperatorResult]:
        all_results = {}
        for i, op_def in enumerate(self.config.operators):
            if operators and op_def.name not in operators:
                continue
            print(f"\n[{i+1}/{len(self.config.operators)}] 算子: {op_def.name}")
            target_shapes = op_def.shapes.get("medium", op_def.shapes.get("small", [[64, 128]]))
            if not target_shapes:
                target_shapes = [[64, 128]]
            op_result = OperatorResult(operator=op_def.name)
            for shape in target_shapes:
                print(f"    shape: {shape}")
                case_result = self.validate_operator(op_def, shape)
                op_result.cases.extend(case_result.cases)
                for case in case_result.cases:
                    if case.status == "REFERENCE":
                        print(f"      📌 {case.backend}: {case.latency_ms:.3f}ms (REF)")
                    elif case.status == "FAIL":
                        print(f"      ❌ {case.backend}: FAIL - {case.error or '精度超标'}")
                    else:
                        icon = "✅" if case.status == "PASS" else "⚠️"
                        ratio = case.latency_ratio_vs_ref
                        print(f"      {icon} {case.backend}: MSE={case.mse:.2e} | RelErr={case.max_rel_err:.2e} | ×{ratio:.2f}x | {case.status}")
            all_results[op_def.name] = op_result
        self._results_by_op = all_results
        return all_results

    def get_summary(self) -> Dict[str, Any]:
        total = passed = failed = warning = skip = 0
        by_op = {}
        for op_name, op_result in self._results_by_op.items():
            statuses = [c.status for c in op_result.cases if c.status != "REFERENCE"]
            op_pass = sum(1 for s in statuses if s == "PASS")
            op_fail = sum(1 for s in statuses if s == "FAIL")
            op_warn = sum(1 for s in statuses if s == "WARNING")
            total += len(statuses)
            passed += op_pass
            failed += op_fail
            warning += op_warn
            by_op[op_name] = {"pass": op_pass, "fail": op_fail, "warning": op_warn, "overall": "PASS" if op_fail == 0 and op_warn == 0 else "FAIL" if op_fail > 0 else "WARNING"}
        return {"total": total, "passed": passed, "failed": failed, "warning": warning, "by_operator": by_op}
