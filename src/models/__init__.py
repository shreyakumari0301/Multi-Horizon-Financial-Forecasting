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

# Export wrapper utilities
from .wrapper import (
    get_grid_params,
    create_model,
    ModelWrapper,
    create_model_from_experiments,
    MODEL_GRID_MAP,
)

# Export runner utilities
from .runner import (
    load_fold_data,
    compute_metrics,
    run_experiment,
    run_grid_search,
)

__all__ = [
    "get_model", "list_models", "register_model",
    "LSTMRegressor", "TransformerRegressor", "TCNRegressor", "RidgeRegressor",
    "get_grid_params", "create_model", "ModelWrapper", "create_model_from_experiments", "MODEL_GRID_MAP",
    "load_fold_data", "compute_metrics", "run_experiment", "run_grid_search",
]
