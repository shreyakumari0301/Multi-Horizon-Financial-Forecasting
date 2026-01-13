"""Model registry for registering and instantiating models."""
from typing import Dict, Any, Type

_MODELS: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type):
        _MODELS[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs):
    """Get a model instance by name with given hyperparameters."""
    if name not in _MODELS:
        raise ValueError(f"Model '{name}' not found. Available: {list(_MODELS.keys())}")
    return _MODELS[name](**kwargs)


def list_models() -> list:
    """List all registered model names."""
    return list(_MODELS.keys())
