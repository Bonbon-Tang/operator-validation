"""
MLU 后端插件实现
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from .base import BackendPlugin, LatencyResult
from ..config import BackendInfo


class MLUBackend(BackendPlugin):
    """
    寒武纪 MLU 后端插件
    
    实现要点：
    1. 使用 torch.mlu 进行算子执行
    2. 支持常见的 ATen 算子映射
    3. 通过 torch.mlu.synchronize() 确保同步
    
    支持的算子：
    - matmul, bmm, mm
    - relu, gelu, silu
    - softmax, log_softmax
    - layer_norm, rms_norm
    - attention, scaled_dot_product_attention
    - convolution ops
    - pooling ops
    - element-wise ops
    """

    # 算子映射表：将 ATen 名称映射到 torch 函数
    OPERATOR_MAP: Dict[str, str] = {
        # 矩阵运算
        "matmul": "torch.matmul",
        "bmm": "torch.bmm",
        "mm": "torch.mm",
        "addmm": "torch.addmm",
        "mv": "torch.mv",
        # 激活函数
        "relu": "torch.relu",
        "gelu": "torch.nn.functional.gelu",
        "silu": "torch.nn.functional.silu",
        "sigmoid": "torch.sigmoid",
        "tanh": "torch.tanh",
        # 归一化
        "softmax": "torch.nn.functional.softmax",
        "log_softmax": "torch.nn.functional.log_softmax",
        "layer_norm": "torch.nn.functional.layer_norm",
        "rms_norm": "torch.nn.functional.rms_norm",
        # 注意力
        "scaled_dot_product_attention": "torch.nn.functional.scaled_dot_product_attention",
        # 卷积
        "conv2d": "torch.nn.functional.conv2d",
        "conv3d": "torch.nn.functional.conv3d",
        "conv_transpose2d": "torch.nn.functional.conv_transpose2d",
        "conv_transpose3d": "torch.nn.functional.conv_transpose3d",
        # 池化
        "max_pool2d": "torch.nn.functional.max_pool2d",
        "avg_pool2d": "torch.nn.functional.avg_pool2d",
        "adaptive_avg_pool2d": "torch.nn.functional.adaptive_avg_pool2d",
        "adaptive_max_pool2d": "torch.nn.functional.adaptive_max_pool2d",
        # 元素级运算
        "add": "torch.add",
        "sub": "torch.sub",
        "mul": "torch.mul",
        "div": "torch.div",
        "pow": "torch.pow",
        "sqrt": "torch.sqrt",
        "rsqrt": "torch.rsqrt",
        "exp": "torch.exp",
        "log": "torch.log",
        "sum": "torch.sum",
        "mean": "torch.mean",
        "prod": "torch.prod",
        # 形状操作
        "reshape": "torch.reshape",
        "view": "torch.Tensor.view",
        "transpose": "torch.transpose",
        "permute": "torch.permute",
        "squeeze": "torch.squeeze",
        "unsqueeze": "torch.unsqueeze",
        "cat": "torch.cat",
        "stack": "torch.stack",
        "split": "torch.split",
        # 采样操作
        " dropout": "torch.nn.functional.dropout",
        # 线性算子
        "linear": "torch.nn.functional.linear",
    }

    def _init_device(self) -> torch.device:
        """
        初始化 MLU 设备
        
        Returns:
            torch.device: MLU 设备对象，格式为 "mlu:{device_id}"
        """
        device_id = self.info.device_id
        return torch.device(f"mlu:{device_id}")

    def exec_operator(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        执行算子
        
        Args:
            aten_name: ATen 算子名称
            inputs: 输入 tensors 列表
            kwargs: 额外参数
            
        Returns:
            输出 tensor
            
        Raises:
            ValueError: 不支持的算子时抛出
            RuntimeError: MLU 执行出错时抛出
        """
        # 确保输入都在 MLU 上
        inputs = [self.to_device(inp) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        
        # 处理 kwargs 中的 tensor
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                processed_kwargs[k] = self.to_device(v)
            else:
                processed_kwargs[k] = v
        
        try:
            # 使用通用分发逻辑
            result = self._dispatch_operator(aten_name, inputs, processed_kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"MLU backend execution failed for operator '{aten_name}': {str(e)}") from e

    def _dispatch_operator(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        分发算子到对应的 PyTorch MLU 实现
        
        Args:
            aten_name: ATen 算子名称
            inputs: 输入 tensors 列表
            kwargs: 处理后的参数
            
        Returns:
            输出 tensor
        """
        # 直接映射的算子（无额外处理）
        direct_dispatch = {
            "matmul": lambda: torch.matmul(inputs[0], inputs[1], **kwargs),
            "bmm": lambda: torch.bmm(inputs[0], inputs[1]),
            "mm": lambda: torch.mm(inputs[0], inputs[1]),
            "addmm": lambda: torch.addmm(inputs[0], inputs[1], inputs[2], **kwargs),
            "mv": lambda: torch.mv(inputs[0], inputs[1]),
            "relu": lambda: torch.relu(inputs[0]),
            "sigmoid": lambda: torch.sigmoid(inputs[0]),
            "tanh": lambda: torch.tanh(inputs[0]),
            "sum": lambda: torch.sum(inputs[0], **kwargs),
            "mean": lambda: torch.mean(inputs[0], **kwargs),
            "prod": lambda: torch.prod(inputs[0], **kwargs),
            "exp": lambda: torch.exp(inputs[0]),
            "log": lambda: torch.log(inputs[0]),
            "sqrt": lambda: torch.sqrt(inputs[0]),
            "rsqrt": lambda: torch.rsqrt(inputs[0]),
            "pow": lambda: torch.pow(inputs[0], inputs[1] if len(inputs) > 1 else kwargs.get("exponent", 2)),
            "add": lambda: torch.add(inputs[0], inputs[1] if len(inputs) > 1 else kwargs.get("other", 1)),
            "sub": lambda: torch.sub(inputs[0], inputs[1] if len(inputs) > 1 else kwargs.get("other", 1)),
            "mul": lambda: torch.mul(inputs[0], inputs[1] if len(inputs) > 1 else kwargs.get("other", 1)),
            "div": lambda: torch.div(inputs[0], inputs[1] if len(inputs) > 1 else kwargs.get("other", 1)),
            "cat": lambda: torch.cat([inp for inp in inputs if isinstance(inp, torch.Tensor)], **kwargs),
            "stack": lambda: torch.stack([inp for inp in inputs if isinstance(inp, torch.Tensor)], **kwargs),
            "split": lambda: torch.split(inputs[0], kwargs.get("split_size_or_sections", 1), **kwargs),
            "squeeze": lambda: torch.squeeze(inputs[0], **kwargs),
            "unsqueeze": lambda: torch.unsqueeze(inputs[0], **kwargs),
            "transpose": lambda: torch.transpose(inputs[0], kwargs.get("dim0", 0), kwargs.get("dim1", 1)),
            "reshape": lambda: torch.reshape(inputs[0], kwargs.get("shape", inputs[1] if len(inputs) > 1 else inputs[0].shape)),
            "view": lambda: inputs[0].view(*[kwargs.get("shape", inputs[0].shape)] if isinstance(kwargs.get("shape"), int) else kwargs.get("shape", inputs[0].shape)),
            "linear": lambda: torch.nn.functional.linear(inputs[0], inputs[1], **kwargs),
            "dropout": lambda: torch.nn.functional.dropout(inputs[0], **kwargs),
        }
        
        # 检查直接映射
        if aten_name in direct_dispatch:
            return direct_dispatch[aten_name]()
        
        # 需要额外参数处理的算子
        if aten_name == "gelu":
            return torch.nn.functional.gelu(inputs[0], **kwargs)
        
        if aten_name == "silu":
            return torch.nn.functional.silu(inputs[0], **kwargs)
        
        if aten_name == "softmax":
            dim = kwargs.get("dim", -1)
            return torch.nn.functional.softmax(inputs[0], dim=dim)
        
        if aten_name == "log_softmax":
            dim = kwargs.get("dim", -1)
            return torch.nn.functional.log_softmax(inputs[0], dim=dim)
        
        if aten_name == "layer_norm":
            normalized_shape = kwargs.get("normalized_shape", inputs[1] if len(inputs) > 1 else None)
            weight = inputs[2] if len(inputs) > 2 else None
            bias = inputs[3] if len(inputs) > 3 else None
            return torch.nn.functional.layer_norm(inputs[0], normalized_shape, weight, bias, **kwargs)
        
        if aten_name == "rms_norm":
            # PyTorch 没有直接支持 rms_norm，使用公式实现
            # rms_norm(x) = x / rms(x) where rms(x) = sqrt(mean(x^2))
            x = inputs[0]
            weight = inputs[1] if len(inputs) > 1 else None
            eps = kwargs.get("eps", 1e-6)
            rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
            output = x * rms
            if weight is not None:
                output = output * weight
            return output
        
        if aten_name == "scaled_dot_product_attention":
            return torch.nn.functional.scaled_dot_product_attention(
                inputs[0], inputs[1], inputs[2],
                attn_mask=kwargs.get("attn_mask"),
                dropout_p=kwargs.get("dropout_p", 0.0),
                is_causal=kwargs.get("is_causal", False),
                scale=kwargs.get("scale"),
            )
        
        if aten_name == "conv2d":
            weight = inputs[1]
            return torch.nn.functional.conv2d(
                inputs[0], weight, bias=inputs[2] if len(inputs) > 2 else None,
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                dilation=kwargs.get("dilation", 1),
                groups=kwargs.get("groups", 1),
            )
        
        if aten_name == "conv3d":
            weight = inputs[1]
            return torch.nn.functional.conv3d(
                inputs[0], weight, bias=inputs[2] if len(inputs) > 2 else None,
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                dilation=kwargs.get("dilation", 1),
                groups=kwargs.get("groups", 1),
            )
        
        if aten_name == "conv_transpose2d":
            weight = inputs[1]
            return torch.nn.functional.conv_transpose2d(
                inputs[0], weight, bias=inputs[2] if len(inputs) > 2 else None,
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                output_padding=kwargs.get("output_padding", 0),
                dilation=kwargs.get("dilation", 1),
                groups=kwargs.get("groups", 1),
            )
        
        if aten_name == "conv_transpose3d":
            weight = inputs[1]
            return torch.nn.functional.conv_transpose3d(
                inputs[0], weight, bias=inputs[2] if len(inputs) > 2 else None,
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                output_padding=kwargs.get("output_padding", 0),
                dilation=kwargs.get("dilation", 1),
                groups=kwargs.get("groups", 1),
            )
        
        if aten_name == "max_pool2d":
            return torch.nn.functional.max_pool2d(
                inputs[0],
                kernel_size=kwargs.get("kernel_size", 2),
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                dilation=kwargs.get("dilation", 1),
                ceil_mode=kwargs.get("ceil_mode", False),
            )
        
        if aten_name == "avg_pool2d":
            return torch.nn.functional.avg_pool2d(
                inputs[0],
                kernel_size=kwargs.get("kernel_size", 2),
                stride=kwargs.get("stride", 1),
                padding=kwargs.get("padding", 0),
                ceil_mode=kwargs.get("ceil_mode", False),
                count_include_pad=kwargs.get("count_include_pad", True),
            )
        
        if aten_name == "adaptive_avg_pool2d":
            return torch.nn.functional.adaptive_avg_pool2d(
                inputs[0],
                output_size=kwargs.get("output_size", (1, 1)),
            )
        
        if aten_name == "adaptive_max_pool2d":
            return torch.nn.functional.adaptive_max_pool2d(
                inputs[0],
                output_size=kwargs.get("output_size", (1, 1)),
            )
        
        if aten_name == "permute":
            dims = kwargs.get("dims", list(range(len(inputs[1])) if len(inputs) > 1 else inputs[0].dim()))
            return torch.permute(inputs[0], dims)
        
        # 不支持的算子
        raise ValueError(f"Unsupported operator for MLU backend: '{aten_name}'")

    def synchronize(self) -> None:
        """
        同步等待，确保算子执行完成
        
        对于 MLU 后端，调用 torch.mlu.synchronize() 等待所有
        在该设备上排队的操作完成。
        """
        torch.mlu.synchronize()

    def benchmark_operator(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any],
        warmup_iter: int = 20,
        bench_iter: int = 100
    ) -> LatencyResult:
        """
        基准测试算子性能
        
        Args:
            aten_name: ATen 算子名称
            inputs: 输入 tensors 列表
            kwargs: 额外参数
            warmup_iter: 预热迭代次数
            bench_iter: 正式测试迭代次数
            
        Returns:
            LatencyResult: 包含延迟和吞吐量的结果
        """
        import time
        
        # 预热
        for _ in range(warmup_iter):
            _ = self.exec_operator(aten_name, inputs, kwargs)
        self.synchronize()
        
        # 正式测试
        start_event = torch.cuda.Event(enable_timing=True) if hasattr(torch, 'cuda') else None
        end_event = torch.cuda.Event(enable_timing=True) if hasattr(torch, 'cuda') else None
        
        # 使用 CPU 计时作为后备
        times = []
        for _ in range(bench_iter):
            if start_event and hasattr(start_event, 'record'):
                start_event.record()
            else:
                t0 = time.perf_counter()
            
            _ = self.exec_operator(aten_name, inputs, kwargs)
            self.synchronize()
            
            if end_event and hasattr(end_event, 'record'):
                end_event.record()
                torch.mlu.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # 转换为毫秒
        
        times_ms = np.array(times)
        latency_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        
        # 计算理论 GFLOPS（简化估算）
        gflops = self._estimate_gflops(aten_name, inputs, kwargs, latency_ms)
        
        return LatencyResult(
            latency_ms=latency_ms,
            std_ms=std_ms,
            throughput_gflops=gflops
        )

    def _estimate_gflops(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any],
        latency_ms: float
    ) -> float:
        """
        估算算子的 GFLOPS 吞吐量
        
        这是一个简化的估算，基于输入形状和算子类型的理论计算量。
        
        Args:
            aten_name: 算子名称
            inputs: 输入 tensors
            kwargs: 参数
            latency_ms: 延迟（毫秒）
            
        Returns:
            估算的 GFLOPS
        """
        if not inputs:
            return 0.0
        
        # 获取第一个输入的元素数量作为基准
        try:
            num_elements = inputs[0].numel()
        except Exception:
            return 0.0
        
        # 不同算子的计算量系数（flops per element）
        flops_per_element = {
            "matmul": lambda ins: ins[0].shape[-2] * ins[1].shape[-1] * ins[0].shape[-1],
            "bmm": lambda ins: ins[0].shape[-2] * ins[1].shape[-2] * ins[0].shape[-1],
            "mm": lambda ins: ins[0].shape[-2] * ins[1].shape[-1] * ins[0].shape[-1],
            "conv2d": lambda ins: self._conv2d_flops(ins, kwargs),
            "relu": lambda ins: ins[0].numel(),
            "gelu": lambda ins: ins[0].numel() * 20,  # 近似
            "softmax": lambda ins: ins[0].numel() * 5,
            "layer_norm": lambda ins: ins[0].numel() * 5,
            "scaled_dot_product_attention": lambda ins: self._attention_flops(ins, kwargs),
        }
        
        try:
            if aten_name in flops_per_element:
                total_flops = flops_per_element[aten_name](inputs)
            else:
                # 默认使用元素数 * 一个系数作为估算
                total_flops = num_elements * 10
        except Exception:
            total_flops = num_elements * 10
        
        # 转换为 GFLOPS
        # latency_ms 是平均延迟，total_flops 是单次计算的 flops
        if latency_ms > 0:
            gflops = (total_flops / 1e9) / (latency_ms / 1000)
        else:
            gflops = 0.0
        
        return gflops

    def _conv2d_flops(self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]) -> int:
        """计算 conv2d 的 flops"""
        x = inputs[0]
        weight = inputs[1]
        out_channels, in_channels, k_h, k_w = weight.shape
        out_h = kwargs.get("out_h", x.shape[2] // kwargs.get("stride", 1))
        out_w = kwargs.get("out_w", x.shape[3] // kwargs.get("stride", 1))
        groups = kwargs.get("groups", 1)
        return out_h * out_w * out_channels * in_channels * k_h * k_w // groups

    def _attention_flops(self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]) -> int:
        """计算 attention 的 flops"""
        # Q, K, V shapes: [batch, seq_len, heads, head_dim]
        q = inputs[0]
        try:
            batch, seq_len, heads, head_dim = q.shape
        except Exception:
            return 0
        # Attention: QK^T (bmm) + softmax + matmul with V
        # QK^T: batch * heads * seq_len * seq_len * head_dim
        # matmul with V: batch * heads * seq_len * seq_len * head_dim
        return 2 * batch * heads * seq_len * seq_len * head_dim