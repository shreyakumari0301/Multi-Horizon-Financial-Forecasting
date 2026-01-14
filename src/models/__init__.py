"""Model package - imports all models to register them."""
from .registry import register_model, get_model, list_models

# Import models to trigger registration
from . import lstm
from . import transformers
from . import tcn
from . import ridge

# Export regressor classes directly
from .lstm import LSTMRegressor
from .transformers import TransformerRegressor
from .tcn import TCNRegressor
from .ridge import RidgeRegressor

__all__ = [
    "get_model", "list_models", "register_model",
    "LSTMRegressor", "TransformerRegressor", "TCNRegressor", "RidgeRegressor",
]
