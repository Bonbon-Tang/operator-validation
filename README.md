# Operator Validation Framework

Multi-backend operator validation framework for AI accelerators. 以 NVIDIA H100 为基线，对比寒武纪 MLU590、FlagOS Triton 等后端的算子精度与性能。

## Features

- **插件化后端系统**：新增后端只需实现 `BackendPlugin` 接口
- **YAML 配置驱动**：全部配置外部化，无需修改代码
- **ATen 算子映射**：以 PyTorch ATen 接口为统一入口
- **精度 + 性能双重验证**：MSE/相对误差/余弦相似度 + 延迟/吞吐

## Quick Start

```bash
# 安装
pip install -e .

# 运行验证
python -m operator_validation.main --config configs/default.yaml

# 指定算子和规模
python -m operator_validation.main \
    --config configs/default.yaml \
    --operators matmul softmax sdpa \
    --shapes small medium

# 输出报告
python -m operator_validation.main \
    --config configs/default.yaml \
    --report-json result.json \
    --report-md result.md
```

## Architecture

```
operator_validation/
├── backend/          # 后端插件系统
│   ├── base.py       # BackendPlugin 抽象基类
│   ├── cuda_backend.py    # NVIDIA CUDA 实现
│   ├── mlu_backend.py     # 寒武纪 MLU 实现
│   ├── triton_backend.py # FlagOS Triton 实现
│   └── registry.py   # 插件注册表
├── engine/           # 核心验证引擎
│   ├── executor.py   # 算子执行器
│   ├── validator.py  # 验证调度
│   └── reporter.py   # 报告生成
├── config.py         # 配置加载
└── utils/metrics.py  # 精度指标计算
```

## Adding New Backends

继承 `BackendPlugin` 并注册：

```python
from operator_validation.backend import BackendPlugin, PluginRegistry

class MyBackend(BackendPlugin):
    name = "my_backend"
    vendor = "my_vendor"
    
    def _init_device(self):
        return torch.device("my_device:0")
    
    def exec_operator(self, aten_name, inputs, kwargs):
        # 实现算子映射
        ...
    
    def synchronize(self):
        ...

PluginRegistry.register("my_backend", MyBackend)
```

## Supported Operators

- 矩阵运算：`matmul`, `linear`, `mm`, `addmm`
- 卷积：`conv2d`, `conv3d`
- 归一化：`layer_norm`, `batch_norm2d`, `rms_norm`
- 激活函数：`relu`, `gelu`, `silu`, `tanh`, `sigmoid`
- 注意力：`sdpa`, `flash_attention`
- 池化：`max_pool2d`, `avg_pool2d`
- 逐元素：`add`, `sub`, `mul`, `div`, `pow`, `sqrt`

## License

MIT
