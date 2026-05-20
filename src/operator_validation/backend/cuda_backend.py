"""
CUDA后端实现 - 支持所有ATen算子
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

from .base import BackendPlugin, LatencyResult
from ..config import BackendInfo


class CUDABackend(BackendPlugin):
    """
    CUDA后端插件实现
    
    支持完整的ATen算子集，通过PyTorch CUDA实现所有算子调用。
    适用于 NVIDIA GPU 的算子验证与性能测试。
    """
    
    # ATen算子名称到PyTorch函数的映射
    OPERATOR_MAP: Dict[str, callable] = {
        # ===== 线性代数算子 =====
        "matmul": torch.matmul,
        "mm": torch.mm,
        "bmm": torch.bmm,
        "addmm": torch.addmm,
        "addmv": torch.addmv,
        "addr": torch.addr,
        "dot": torch.dot,
        "vdot": torch.vdot,
        "cholesky": torch.cholesky,
        "eig": torch.eig,
        "gesv": torch.gesv,
        "inverse": torch.inverse,
        "det": torch.det,
        "logdet": torch.logdet,
        "slogdet": torch.slogdet,
        "lstsq": torch.lstsq,
        "solve": torch.solve,
        "svd": torch.svd,
        "cholesky_solve": torch.cholesky_solve,
        "triangular_solve": torch.triangular_solve,
        
        # ===== 逐元素算子 =====
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "fmod": torch.fmod,
        "remainder": torch.remainder,
        "pow": torch.pow,
        "lerp": torch.lerp,
        
        # ===== 激活函数 =====
        "relu": torch.relu,
        "relu_": torch.relu_,
        "leaky_relu": torch.leaky_relu,
        "leaky_relu_": torch.leaky_relu_,
        "rrelu": torch.rrelu,
        "rrelu_": torch.rrelu_,
        "sigmoid": torch.sigmoid,
        "sigmoid_": torch.sigmoid_,
        "tanh": torch.tanh,
        "tanh_": torch.tanh_,
        "hardshrink": torch.hardshrink,
        "softshrink": torch.softshrink,
        " hardswish": torch.hardswish,
        "hardswish_": torch.hardswish_,
        "hardsigmoid": torch.hardsigmoid,
        "hardsigmoid_": torch.hardsigmoid_,
        "silu": torch.silu,
        "silu_": torch.silu_,
        " mish": torch.mish,
        "mish_": torch.mish_,
        "softplus": torch.softplus,
        "softplus_": torch.softplus_,
        "softsign": torch.softsign,
        "softsign_": torch.softsign_,
        
        # ===== Softmax家族 =====
        "softmax": torch.softmax,
        "log_softmax": torch.log_softmax,
        
        # ===== 归一化算子 =====
        "batch_norm": torch.batch_norm,
        "batch_norm_elemt": torch.batch_norm_elemt,
        "native_batch_norm": torch.native_batch_norm,
        "layer_norm": torch.layer_norm,
        "instance_norm": torch.instance_norm,
        "instance_norm": torch.instance_norm,
        "group_norm": torch.group_norm,
        "local_response_norm": torch.local_response_norm,
        "normalize": torch.nn.functional.normalize,
        
        # ===== 卷积算子 =====
        "conv1d": torch.conv1d,
        "conv2d": torch.conv2d,
        "conv3d": torch.conv3d,
        "conv_transpose1d": torch.conv_transpose1d,
        "conv_transpose2d": torch.conv_transpose2d,
        "conv_transpose3d": torch.conv_transpose3d,
        "conv_tbc": torch.conv_tbc,
        
        # ===== 池化算子 =====
        "max_pool1d": torch.max_pool1d,
        "max_pool2d": torch.max_pool2d,
        "max_pool3d": torch.max_pool3d,
        "avg_pool1d": torch.avg_pool1d,
        "avg_pool2d": torch.avg_pool2d,
        "avg_pool3d": torch.avg_pool3d,
        "adaptive_max_pool1d": torch.adaptive_max_pool1d,
        "adaptive_max_pool2d": torch.adaptive_max_pool2d,
        "adaptive_max_pool3d": torch.adaptive_max_pool3d,
        "adaptive_avg_pool1d": torch.adaptive_avg_pool1d,
        "adaptive_avg_pool2d": torch.adaptive_avg_pool2d,
        "adaptive_avg_pool3d": torch.adaptive_avg_pool3d,
        "fractional_max_pool2d": torch.fractional_max_pool2d,
        "fractional_max_pool3d": torch.fractional_max_pool3d,
        
        # ===== Dropout =====
        "dropout": torch.nn.functional.dropout,
        "dropout2d": torch.nn.functional.dropout2d,
        "dropout3d": torch.nn.functional.dropout3d,
        "feature_dropout": torch.nn.functional.feature_dropout,
        
        # ===== 损失函数 =====
        "mse_loss": torch.nn.functional.mse_loss,
        "l1_loss": torch.nn.functional.l1_loss,
        "smooth_l1_loss": torch.nn.functional.smooth_l1_loss,
        "huber_loss": torch.nn.functional.huber_loss,
        "cross_entropy": torch.nn.functional.cross_entropy,
        "nll_loss": torch.nn.functional.nll_loss,
        "poisson_nll_loss": torch.nn.functional.poisson_nll_loss,
        "gaussian_nll_loss": torch.nn.functional.gaussian_nll_loss,
        "binary_cross_entropy": torch.nn.functional.binary_cross_entropy,
        "binary_cross_entropy_with_logits": torch.nn.functional.binary_cross_entropy_with_logits,
        "hinge_embedding_loss": torch.nn.functional.hinge_embedding_loss,
        "cosine_embedding_loss": torch.nn.functional.cosine_embedding_loss,
        "multi_margin_loss": torch.nn.functional.multi_margin_loss,
        "multi_head_attention_forward": torch.nn.functional.multi_head_attention_forward,
        
        # ===== 嵌入算子 =====
        "embedding": torch.nn.functional.embedding,
        "embedding_bag": torch.nn.functional.embedding_bag,
        
        # ===== 形状变换算子 =====
        "reshape": torch.reshape,
        "view": lambda x, shape: x.view(shape),
        "flatten": torch.flatten,
        "transpose": torch.transpose,
        "permute": torch.permute,
        "t": torch.t,
        "flip": torch.flip,
        "roll": torch.roll,
        "rot90": torch.rot90,
        "narrow": torch.narrow,
        "narrow_copy": torch.narrow_copy,
        "select": torch.select,
        "slice": torch.slice,
        "split": torch.split,
        "chunk": torch.chunk,
        "squeeze": torch.squeeze,
        "unsqueeze": torch.unsqueeze,
        "cat": torch.cat,
        "concat": torch.cat,
        "stack": torch.stack,
        "gather": torch.gather,
        "scatter": torch.scatter,
        "scatter_add": torch.scatter_add,
        "index_select": torch.index_select,
        "index_add": torch.index_add,
        "index_copy": torch.index_copy,
        "masked_scatter": torch.masked_scatter,
        "masked_select": torch.masked_select,
        
        # ===== 稀疏算子 =====
        "sparse_coo_tensor": torch.sparse_coo_tensor,
        "sparse_csr_tensor": torch.sparse_csr_tensor,
        
        # ===== 随机数算子 =====
        "uniform": torch.uniform,
        "normal": torch.normal,
        "normal_": torch.normal_,
        "rand": torch.rand,
        "randn": torch.randn,
        "randint": torch.randint,
        "randperm": torch.randperm,
        
        # ===== 聚合算子 =====
        "sum": torch.sum,
        "prod": torch.prod,
        "mean": torch.mean,
        "std": torch.std,
        "var": torch.var,
        "median": torch.median,
        "mode": torch.mode,
        "amax": torch.amax,
        "amin": torch.amin,
        "argmax": torch.argmax,
        "argmin": torch.argmin,
        "cumsum": torch.cumsum,
        "cumprod": torch.cumprod,
        "logsumexp": torch.logsumexp,
        
        # ===== 比较算子 =====
        "eq": torch.eq,
        "ne": torch.ne,
        "lt": torch.lt,
        "le": torch.le,
        "gt": torch.gt,
        "ge": torch.ge,
        "equal": torch.equal,
        "allclose": torch.allclose,
        "isclose": torch.isclose,
        "isnan": torch.isnan,
        "isfinite": torch.isfinite,
        "isinf": torch.isinf,
        "isneginf": torch.isneginf,
        "isposinf": torch.isposinf,
        
        # ===== 逻辑算子 =====
        "logical_and": torch.logical_and,
        "logical_or": torch.logical_or,
        "logical_not": torch.logical_not,
        "logical_xor": torch.logical_xor,
        
        # ===== 位运算算子 =====
        "bitwise_and": torch.bitwise_and,
        "bitwise_or": torch.bitwise_or,
        "bitwise_xor": torch.bitwise_xor,
        "bitwise_not": torch.bitwise_not,
        "bitwise_left_shift": torch.bitwise_left_shift,
        "bitwise_right_shift": torch.bitwise_right_shift,
        
        # ===== 数学函数 =====
        "abs": torch.abs,
        "abs_": torch.abs_,
        "absolute": torch.absolute,
        "absolute_": torch.absolute_,
        "acos": torch.acos,
        "acos_": torch.acos_,
        "acosh": torch.acosh,
        "acosh_": torch.acosh_,
        "asin": torch.asin,
        "asin_": torch.asin_,
        "asinh": torch.asinh,
        "asinh_": torch.asinh_,
        "atan": torch.atan,
        "atan_": torch.atan_,
        "atanh": torch.atanh,
        "atanh_": torch.atanh_,
        "atan2": torch.atan2,
        "cos": torch.cos,
        "cos_": torch.cos_,
        "cosh": torch.cosh,
        "cosh_": torch.cosh_,
        "deg2rad": torch.deg2rad,
        "deg2rad_": torch.deg2rad_,
        "rad2deg": torch.rad2deg,
        "rad2deg_": torch.rad2deg_,
        "exp": torch.exp,
        "exp_": torch.exp_,
        "exp2": torch.exp2,
        "exp2_": torch.exp2_,
        "expm1": torch.expm1,
        "expm1_": torch.expm1_,
        "log": torch.log,
        "log_": torch.log_,
        "log10": torch.log10,
        "log10_": torch.log10_,
        "log1p": torch.log1p,
        "log1p_": torch.log1p_,
        "log2": torch.log2,
        "log2_": torch.log2_,
        "reciprocal": torch.reciprocal,
        "reciprocal_": torch.reciprocal_,
        "rsqrt": torch.rsqrt,
        "rsqrt_": torch.rsqrt_,
        "sign": torch.sign,
        "sign_": torch.sign_,
        "sqrt": torch.sqrt,
        "sqrt_": torch.sqrt_,
        "square": torch.square,
        "square_": torch.square_,
        "cbrt": torch.cbrt,
        "cbrt_": torch.cbrt_,
        "neg": torch.neg,
        "neg_": torch.neg_,
        "negative": torch.negative,
        "negative_": torch.negative_,
        
        # ===== 矩阵分解 =====
        "lu": torch.lu,
        "pstrf": torch.pstrf,
        "geqrf": torch.geqrf,
        "orgqr": torch.orgqr,
        "ormqr": torch.ormqr,
        
        # ===== 距离函数 =====
        "pairwise_distance": torch.nn.functional.pairwise_distance,
        "cosine_similarity": torch.nn.functional.cosine_similarity,
        "pdist": torch.pdist,
        "cdist": torch.cdist,
        
        # ===== 填充函数 =====
        "pad": torch.nn.functional.pad,
        
        # ===== 插值函数 =====
        "interpolate": torch.nn.functional.interpolate,
        "grid_sample": torch.nn.functional.grid_sample,
        
        # ===== 注意力机制 =====
        "scaled_dot_product_attention": torch.nn.functional.scaled_dot_product_attention,
        
        # ===== 其他常用算子 =====
        "clamp": torch.clamp,
        "clamp_": torch.clamp_,
        "clip": torch.clamp,
        "clip_": torch.clamp_,
        "floor": torch.floor,
        "floor_": torch.floor_,
        "ceil": torch.ceil,
        "ceil_": torch.ceil_,
        "round": torch.round,
        "round_": torch.round_,
        "trunc": torch.trunc,
        "trunc_": torch.trunc_,
        "frac": torch.frac,
        "frac_": torch.frac_,
        "fmod_": torch.fmod_,
        "remainder_": torch.remainder_,
        "logical_and_": torch.logical_and_,
        "logical_or_": torch.logical_or_,
        "logical_not_": torch.logical_not_,
        "logical_xor_": torch.logical_xor_,
    }
    
    def __init__(self, info: BackendInfo):
        """
        初始化CUDA后端
        
        Args:
            info: 后端配置信息
        """
        super().__init__(info)
        self._stream: Optional[torch.cuda.Stream] = None
        
    def _init_device(self) -> torch.device:
        """
        初始化CUDA设备
        
        Returns:
            torch.device: CUDA设备对象
        """
        device_id = self.info.device_id or "0"
        return torch.device(f"cuda:{device_id}")
    
    def synchronize(self) -> None:
        """同步等待，确保CUDA算子执行完成"""
        torch.cuda.synchronize(self._device)
    
    def exec_operator(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        执行ATen算子
        
        Args:
            aten_name: ATen算子名称
            inputs: 输入tensor列表
            kwargs: 额外参数
            
        Returns:
            输出tensor
        """
        # 特殊处理一些算子
        if aten_name in self._SPECIAL_HANDLERS:
            return self._SPECIAL_HANDLERS[aten_name](self, inputs, kwargs)
        
        # 查找对应的PyTorch函数
        if aten_name not in self.OPERATOR_MAP:
            raise ValueError(f"Unsupported ATen operator: {aten_name}")
        
        op_func = self.OPERATOR_MAP[aten_name]
        
        # 调用算子函数
        try:
            output = op_func(*inputs, **kwargs)
            return output
        except TypeError as e:
            # 尝试只传位置参数
            try:
                output = op_func(*inputs)
                return output
            except Exception:
                raise ValueError(
                    f"Failed to execute operator {aten_name} with inputs {inputs} and kwargs {kwargs}"
                ) from e
    
    def _handle_matmul(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """特殊处理matmul以支持不同维度的矩阵乘法"""
        a, b = inputs[0], inputs[1]
        if a.dim() == 1 and b.dim() == 1:
            return torch.dot(a, b)
        elif a.dim() == 2 and b.dim() == 2:
            return torch.mm(a, b)
        elif a.dim() == 3 and b.dim() == 3:
            return torch.bmm(a, b)
        else:
            return torch.matmul(a, b)
    
    def _handle_conv(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理卷积算子，支持权重格式调整"""
        if len(inputs) >= 3:
            # 输入格式: (input, weight, bias) 或 (input, weight)
            input_tensor = inputs[0]
            weight = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None
            
            # 从kwargs获取参数
            stride = kwargs.get("stride", 1)
            padding = kwargs.get("padding", 0)
            dilation = kwargs.get("dilation", 1)
            groups = kwargs.get("groups", 1)
            
            # 判断卷积维度
            if input_tensor.dim() == 3:
                # conv1d
                return torch.conv1d(input_tensor, weight, bias, stride, padding, dilation, groups)
            elif input_tensor.dim() == 4:
                # conv2d
                return torch.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)
            elif input_tensor.dim() == 5:
                # conv3d
                return torch.conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)
        
        raise ValueError(f"Invalid inputs for conv operator: {inputs}")
    
    def _handle_pool2d(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理2D池化算子"""
        input_tensor = inputs[0]
        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride", kernel_size)
        padding = kwargs.get("padding", 0)
        dilation = kwargs.get("dilation", 1)
        return_indices = kwargs.get("return_indices", False)
        ceil_mode = kwargs.get("ceil_mode", False)
        
        if ceil_mode:
            return torch.nn.functional.max_pool2d(
                input_tensor, kernel_size, stride, padding, dilation, return_indices, ceil_mode
            )
        return torch.max_pool2d(input_tensor, kernel_size, stride, padding, dilation)
    
    def _handle_adaptive_avg_pool(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理adaptive_avg_pool算子"""
        input_tensor = inputs[0]
        output_size = kwargs.get("output_size")
        
        if input_tensor.dim() == 3:
            return torch.adaptive_avg_pool1d(input_tensor, output_size)
        elif input_tensor.dim() == 4:
            return torch.adaptive_avg_pool2d(input_tensor, output_size)
        elif input_tensor.dim() == 5:
            return torch.adaptive_avg_pool3d(input_tensor, output_size)
        
        raise ValueError(f"Invalid input dim for adaptive_avg_pool: {input_tensor.dim()}")
    
    def _handle_batch_norm(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理batch_norm算子"""
        input_tensor = inputs[0]
        running_mean = kwargs.get("running_mean")
        running_var = kwargs.get("running_var")
        weight = kwargs.get("weight")
        bias = kwargs.get("bias")
        training = kwargs.get("training", False)
        momentum = kwargs.get("momentum", 0.1)
        eps = kwargs.get("eps", 1e-05)
        
        return torch.batch_norm(
            input_tensor, weight, bias, running_mean, running_var, training, momentum, eps
        )
    
    def _handle_layer_norm(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理layer_norm算子"""
        input_tensor = inputs[0]
        normalized_shape = kwargs.get("normalized_shape")
        weight = kwargs.get("weight")
        bias = kwargs.get("bias")
        eps = kwargs.get("eps", 1e-05)
        
        return torch.layer_norm(input_tensor, normalized_shape, weight, bias, eps)
    
    def _handle_softmax(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理softmax算子"""
        input_tensor = inputs[0]
        dim = kwargs.get("dim", -1)
        dtype = kwargs.get("dtype")
        
        if dtype is not None:
            input_tensor = input_tensor.to(dtype)
        return torch.softmax(input_tensor, dim=dim)
    
    def _handle_cross_entropy(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理cross_entropy损失"""
        if len(inputs) >= 2:
            input_tensor = inputs[0]
            target = inputs[1]
        else:
            input_tensor = inputs[0]
            target = kwargs.get("target")
        
        weight = kwargs.get("weight")
        ignore_index = kwargs.get("ignore_index", -100)
        reduction = kwargs.get("reduction", "mean")
        
        return torch.nn.functional.cross_entropy(
            input_tensor, target, weight, ignore_index=ignore_index, reduction=reduction
        )
    
    def _handle_embedding(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理embedding算子"""
        indices = inputs[0]
        embedding_dim = kwargs.get("embedding_dim")
        padding_idx = kwargs.get("padding_idx")
        max_norm = kwargs.get("max_norm")
        norm_type = kwargs.get("norm_type", 2.0)
        scale_grad_by_freq = kwargs.get("scale_grad_by_freq", False)
        sparse = kwargs.get("sparse", False)
        
        if isinstance(embedding_dim, torch.Tensor):
            # embedding_dim actually is the weight tensor
            weight = embedding_dim
            return torch.nn.functional.embedding(
                indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
            )
        
        raise ValueError("embedding requires weight tensor")
    
    def _handle_split(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理split算子，返回第一个分块"""
        input_tensor = inputs[0]
        split_size = kwargs.get("split_size")
        dim = kwargs.get("dim", 0)
        
        splits = torch.split(input_tensor, split_size, dim=dim)
        return splits[0] if len(splits) > 0 else input_tensor
    
    def _handle_chunk(
        self, inputs: List[torch.Tensor], kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """处理chunk算子"""
        input_tensor = inputs[0]
        chunks = kwargs.get("chunks", 1)
        dim = kwargs.get("dim", 0)
        
        result = torch.chunk(input_tensor, chunks, dim=dim)
        return result[0] if len(result) > 0 else input_tensor
    
    # 特殊处理函数映射
    _SPECIAL_HANDLERS: Dict[str, callable] = {
        "matmul": _handle_matmul,
        "conv2d": _handle_conv,
        "conv3d": _handle_conv,
        "max_pool2d": _handle_pool2d,
        "adaptive_avg_pool2d": _handle_adaptive_avg_pool,
        "batch_norm": _handle_batch_norm,
        "layer_norm": _handle_layer_norm,
        "softmax": _handle_softmax,
        "cross_entropy": _handle_cross_entropy,
        "embedding": _handle_embedding,
        "split": _handle_split,
        "chunk": _handle_chunk,
    }
    
    # ==================== 兼容性别名 ====================
    # 为了支持更灵活的算子名称识别，添加别名映射
    ALIAS_MAP: Dict[str, str] = {
        "Linear": "matmul",
        "Conv2d": "conv2d",
        "ReLU": "relu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Softmax": "softmax",
        "Dropout": "dropout",
        "MaxPool2d": "max_pool2d",
        "AvgPool2d": "avg_pool2d",
        "AdaptiveAvgPool2d": "adaptive_avg_pool2d",
        "BatchNorm2d": "batch_norm",
        "LayerNorm": "layer_norm",
        "MSELoss": "mse_loss",
        "CrossEntropyLoss": "cross_entropy",
        "Embedding": "embedding",
        "Split": "split",
        "Chunk": "chunk",
    }
    
    def resolve_operator(self, name: str) -> str:
        """
        解析算子名称，支持别名和标准化
        
        Args:
            name: 算子名称（可能带命名空间前缀）
            
        Returns:
            标准ATen算子名称
        """
        # 移除可能的命名空间前缀
        if "::" in name:
            name = name.split("::")[-1]
        
        # 检查别名
        if name in self.ALIAS_MAP:
            return self.ALIAS_MAP[name]
        
        return name
    
    @property
    def is_available(self) -> bool:
        """检查CUDA后端是否可用"""
        return torch.cuda.is_available()
    
    def get_device_properties(self) -> Dict[str, Any]:
        """
        获取CUDA设备属性
        
        Returns:
            设备属性字典
        """
        if not torch.cuda.is_available():
            return {}
        
        device_id = int(self.info.device_id or "0")
        props = {
            "name": torch.cuda.get_device_name(device_id),
            "capability": torch.cuda.get_device_capability(device_id),
            "total_memory": torch.cuda.get_device_properties(device_id).total_memory,
            "multi_processor_count": torch.cuda.get_device_properties(device_id).multi_processor_count,
        }
        return props
    
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
            aten_name: ATen算子名称
            inputs: 输入tensor列表
            kwargs: 额外参数
            warmup_iter: 预热迭代次数
            bench_iter: 正式测试迭代次数
            
        Returns:
            LatencyResult: 延迟测试结果
        """
        import time
        
        # 预热
        for _ in range(warmup_iter):
            self.exec_operator(aten_name, inputs, kwargs)
        self.synchronize()
        
        # 正式测试
        latencies = []
        for _ in range(bench_iter):
            start = time.perf_counter()
            self.exec_operator(aten_name, inputs, kwargs)
            self.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒
        
        # 计算统计信息
        latencies = np.array(latencies)
        return LatencyResult(
            latency_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            throughput_gflops=self._calculate_gflops(aten_name, inputs, kwargs, latencies)
        )
    
    def _calculate_gflops(
        self,
        aten_name: str,
        inputs: List[torch.Tensor],
        kwargs: Dict[str, Any],
        latencies: np.ndarray
    ) -> float:
        """
        计算算子的GFLOPS
        
        Args:
            aten_name: 算子名称
            inputs: 输入tensors
            kwargs: 额外参数
            latencies: 延迟数组(毫秒)
            
        Returns:
            GFLOPS估计值
        """
        # 基于常见算子的GFLOPS估算
        # 对于矩阵乘法等算子，根据输入规模估算flops
        if aten_name in ("matmul", "mm", "bmm"):
            if len(inputs) >= 2:
                a, b = inputs[0], inputs[1]
                if aten_name == "bmm":
                    n, m, k = a.shape[0], a.shape[1], a.shape[2]
                    p = b.shape[2]
                    flops = 2 * n * m * k * p
                elif aten_name == "mm":
                    m, k = a.shape
                    k2, p = b.shape
                    flops = 2 * m * k * p
                else:
                    flops = a.numel() * b.shape[-1] * 2
                
                avg_latency_ms = np.mean(latencies)
                if avg_latency_ms > 0:
                    return flops / (avg_latency_ms * 1e6)  # GFLOPS
        elif aten_name.startswith("conv"):
            if len(inputs) >= 2:
                input_tensor = inputs[0]
                weight = inputs[1]
                # 简化估算
                flops = input_tensor.numel() * weight.shape[0] * 2
                avg_latency_ms = np.mean(latencies)
                if avg_latency_ms > 0:
                    return flops / (avg_latency_ms * 1e6)
        
        return 0.0  # 无法估算时返回0
    
    def __repr__(self) -> str:
        return f"CUDABackend(device={self._device}, dtype={self.info.dtype})"