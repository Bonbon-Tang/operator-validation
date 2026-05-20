"""
Backend Plugin Registry
~~~~~~~~~~~~~~~~~~~~~~
后端插件注册表，实现自动发现与延迟注册机制。
"""

from __future__ import annotations

from typing import Dict, Type, Optional, List
import logging

from .base import BackendPlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    后端插件注册表
    
    支持注册、注销、查询后端插件。
    
    Example:
        >>> PluginRegistry.register("cuda", CUDABackend)
        >>> PluginRegistry.register("triton", TritonBackend)
        >>> backend_cls = PluginRegistry.get("cuda")
        >>> all_backends = PluginRegistry.list_all()
    """
    
    _plugins: Dict[str, Type[BackendPlugin]] = {}
    _names: List[str] = []
    
    @classmethod
    def register(
        cls,
        name: str,
        plugin_class: Type[BackendPlugin],
        override: bool = False
    ) -> None:
        """
        注册后端插件
        
        Args:
            name: 插件唯一名称
            plugin_class: 插件类（必须是 BackendPlugin 子类）
            override: 若已存在是否覆盖，默认 False
        """
        if name in cls._plugins and not override:
            raise ValueError(
                f"Plugin '{name}' already registered. "
                f"Use override=True to replace."
            )
        
        if not issubclass(plugin_class, BackendPlugin):
            raise TypeError(
                f"Plugin class must be subclass of BackendPlugin, "
                f"got {plugin_class.__name__}"
            )
        
        cls._plugins[name] = plugin_class
        if name not in cls._names:
            cls._names.append(name)
        
        logger.info(f"Registered backend plugin: {name} ({plugin_class.__name__})")
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        注销后端插件
        
        Args:
            name: 插件名称
            
        Returns:
            是否成功注销
        """
        if name in cls._plugins:
            del cls._plugins[name]
            cls._names.remove(name)
            logger.info(f"Unregistered backend plugin: {name}")
            return True
        return False
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BackendPlugin]]:
        """
        获取后端插件类
        
        Args:
            name: 插件名称
            
        Returns:
            插件类，若不存在返回 None
        """
        return cls._plugins.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """返回所有已注册的插件名称"""
        return list(cls._names)
    
    @classmethod
    def create_instance(
        cls,
        name: str,
        backend_info,
        **kwargs
    ) -> Optional[BackendPlugin]:
        """
        创建后端插件实例
        
        Args:
            name: 插件名称
            backend_info: BackendInfo 配置对象
            **kwargs: 传递给插件构造函数的额外参数
            
        Returns:
            插件实例，若创建失败返回 None
        """
        plugin_class = cls.get(name)
        if plugin_class is None:
            logger.error(f"Plugin '{name}' not found in registry")
            return None
        
        try:
            return plugin_class(backend_info, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create plugin '{name}': {e}")
            return None
    
    @classmethod
    def auto_register(cls) -> None:
        """
        自动注册所有已知的内置后端。
        
        按优先级尝试导入并注册：
        - CUDA Backend
        - Triton Backend  
        - MLU Backend
        """
        # 延迟导入避免循环依赖
        from .cuda_backend import CUDABackend
        
        cls.register("cuda", CUDABackend)
        
        # 尝试导入 Triton
        try:
            from .triton_backend import TritonBackend
            cls.register("triton", TritonBackend)
        except ImportError as e:
            logger.warning(f"Triton backend not available: {e}")
        
        # 尝试导入 MLU
        try:
            from .mlu_backend import MLUBackend
            cls.register("mlu", MLUBackend)
        except ImportError:
            logger.debug("MLU backend not available")
    
    @classmethod
    def clear(cls) -> None:
        """清空所有已注册的插件（主要用于测试）"""
        cls._plugins.clear()
        cls._names.clear()


def register_backend(name: str, override: bool = False):
    """
    装饰器：注册后端插件
    
    Example:
        @register_backend("my_backend")
        class MyBackend(BackendPlugin):
            ...
    """
    def decorator(cls: Type[BackendPlugin]) -> Type[BackendPlugin]:
        PluginRegistry.register(name, cls, override=override)
        return cls
    return decorator