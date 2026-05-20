"""算子注册表"""
from typing import Dict, List, Callable, Any

class OperatorRegistry:
    _operators: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, func: Callable):
        cls._operators[name] = func
    
    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._operators.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls._operators.keys())
