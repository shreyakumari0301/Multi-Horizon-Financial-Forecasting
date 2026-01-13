"""Main training script - orchestrates model training pipeline."""
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.experiments as experiments
from src.models.registry import get_model
from src.train.runner import run_experiment, run_grid_search


def get_grid_params(grid: Dict[str, list], index: int = 0) -> Dict[str, Any]:
    """
    Extract hyperparameters from a grid by taking the first value of each list.
    
    Args:
        grid: Dictionary with lists of hyperparameter values
        index: Which index to take from each list (default: 0)
    
    Returns:
        Dictionary of hyperparameters with single values
    """
    return {k: v[index] if isinstance(v, list) else v for k, v in grid.items()}


def create_model(
    model_name: str,
    grid: Dict[str, list],
    grid_index: int = 0,
    **override_kwargs
):
    """
    Create a model instance from hyperparameter grid.
    
    Args:
        model_name: Name of the model ("lstm", "transformer", "tcn", "ridge")
        grid: Hyperparameter grid dictionary
        grid_index: Index to use when extracting from grid (default: 0)
        **override_kwargs: Additional parameters to override
    
    Returns:
        Model instance
    """
    params = get_grid_params(grid, grid_index)
    params.update(override_kwargs)
    return get_model(model_name, **params)


def main():
    """Main training pipeline."""
    
    # Configuration from experiments.py
    MODELS = ["lstm", "transformer", "tcn", "ridge"]  # or select specific models
    FOLDS = experiments.FOLDS
    HORIZONS = experiments.HORIZONS
    
    # Paths
    SPLITS_DIR = "data/splits"
    RESULTS_DIR = "data/experiments"
    
    print("=" * 60)
    print("Training Pipeline")
    print("=" * 60)
    print(f"Models: {MODELS}")
    print(f"Folds: {FOLDS}")
    print(f"Horizons: {HORIZONS}")
    print(f"Splits: {SPLITS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 60)
    
    # Run experiments for each model
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get grid from experiments
        grid_map = {
            "lstm": experiments.LSTM_GRID,
            "transformer": experiments.TRANSFORMER_GRID,
            "tcn": experiments.TCN_GRID,
            "ridge": experiments.RIDGE_GRID,
        }
        
        grid = grid_map.get(model_name)
        if grid is None:
            print(f"Warning: No grid found for {model_name}, skipping...")
            continue
        
        # Run grid search (using first grid combination for now)
        results_df = run_grid_search(
            model_name=model_name,
            grid=grid,
            folds=FOLDS,
            horizons=HORIZONS,
            splits_dir=SPLITS_DIR,
            results_dir=RESULTS_DIR,
            grid_index=0,  # Use first combination from grid
            create_model_fn=create_model,  # Pass the create_model function
        )
        
        print(f"\n{model_name.upper()} completed. Results shape: {results_df.shape}")
    
    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
