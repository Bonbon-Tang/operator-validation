"""算子执行器 - 负责在指定后端上执行算子并测量性能。"""
import time
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ..backend.base import BackendPlugin, LatencyResult

class OperatorExecutor:
    def __init__(self, backend: BackendPlugin, warmup: int = 20, bench_iter: int = 100):
        self.backend = backend
        self.warmup = warmup
        self.bench_iter = bench_iter

    def execute(self, aten_name: str, input_shapes: List[Tuple], kwargs: Optional[Dict] = None):
        kwargs = kwargs or {}
        inputs = [self.backend.create_tensor(shape) for shape in input_shapes]
        inputs_on_device = [self.backend.to_device(t) for t in inputs]
        for _ in range(self.warmup):
            _ = self.backend.exec_operator(aten_name, inputs_on_device, kwargs)
        self.backend.synchronize()
        times = []
        output = None
        for _ in range(self.bench_iter):
            start = time.perf_counter()
            output = self.backend.exec_operator(aten_name, inputs_on_device, kwargs)
            self.backend.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        times_valid = times[5:] if len(times) > 5 else times
        avg_latency = np.mean(times_valid)
        std_latency = np.std(times_valid)
        output_elements = output.numel() if output is not None else 1
        throughput = (output_elements * 2) / (avg_latency * 1e-3) / 1e9
        latency_result = LatencyResult(latency_ms=avg_latency, std_ms=std_latency, throughput_gflops=throughput)
        output_np = output.cpu().numpy() if output is not None else np.array([])
        return output_np, latency_result

    def dry_run(self, aten_name: str, input_shapes: List[Tuple], kwargs: Optional[Dict] = None) -> bool:
        try:
            kwargs = kwargs or {}
            inputs = [self.backend.create_tensor(shape) for shape in input_shapes]
            inputs_on_device = [self.backend.to_device(t) for t in inputs]
            _ = self.backend.exec_operator(aten_name, inputs_on_device, kwargs)
            self.backend.synchronize()
            return True
        except Exception:
            return False
