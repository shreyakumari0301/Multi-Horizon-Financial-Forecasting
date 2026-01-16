"""Main training script - orchestrates model training pipeline."""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

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


def ensure_news_features_integrated(
    splits_dir: str = "data/splits",
    news_path: Optional[str] = None,
    processed_dir: str = "data/processed"
) -> bool:
    """
    Ensure news features are integrated into train/test splits.
    
    Checks if news features are already integrated, and if not, processes them
    from raw news data if available.
    
    Args:
        splits_dir: Directory with train/test splits
        news_path: Optional path to raw news headlines CSV
        processed_dir: Directory for processed features
    
    Returns:
        True if news features are available, False otherwise
    """
    # Check if news features are already integrated
    # Look for z_news_pc* columns in first fold
    fold_0_dir = os.path.join(splits_dir, "fold_0")
    train_path = os.path.join(fold_0_dir, "train.csv")
    
    if os.path.exists(train_path):
        import pandas as pd
        train = pd.read_csv(train_path, index_col=0, nrows=1)  # Just check columns
        news_cols = [c for c in train.columns if c.startswith("z_news_pc")]
        if len(news_cols) > 0:
            print(f"✓ News features already integrated: {len(news_cols)} features found")
            return True
    
    # News features not integrated - check if we can process them
    news_features_path = os.path.join(processed_dir, "news_features_28d.csv")
    
    # If processed news features exist, integrate them
    if os.path.exists(news_features_path):
        print("\n" + "=" * 60)
        print("Integrating Existing News Features")
        print("=" * 60)
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "integrate_news_features",
                os.path.join(project_root, "scripts", "features", "integrate_news_features.py")
            )
            integrate_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(integrate_module)
            
            integrate_module.integrate_news_features(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                news_features_path=news_features_path
            )
            print("✓ News features integrated successfully")
            return True
        except Exception as e:
            print(f"⚠ Failed to integrate news features: {e}")
            return False
    
    # If raw news data provided, process it
    if news_path and os.path.exists(news_path):
        print("\n" + "=" * 60)
        print("Processing News Features from Raw Data")
        print("=" * 60)
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "process_all_features",
                os.path.join(project_root, "scripts", "features", "process_all_features.py")
            )
            process_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(process_module)
            
            process_module.process_all_features(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                news_path=news_path,
                n_components=28
            )
            print("✓ News features processed and integrated successfully")
            return True
        except Exception as e:
            print(f"⚠ Failed to process news features: {e}")
            return False
    
    # No news data available
    print("⚠ No news features available - using technical features only")
    return False


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--news_path",
        type=str,
        default=None,
        help="Path to raw news headlines CSV (optional, will process if provided)"
    )
    parser.add_argument(
        "--skip_news",
        action="store_true",
        help="Skip news feature processing even if available"
    )
    
    args = parser.parse_args()
    
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
    
    # Ensure news features are integrated (if not skipped)
    has_news = False
    if not args.skip_news:
        has_news = ensure_news_features_integrated(
            splits_dir=SPLITS_DIR,
            news_path=args.news_path
        )
        if has_news:
            print("✓ Training with 38 features (10 technical + 28 news)")
        else:
            print("✓ Training with 10 features (technical only)")
    else:
        print("⚠ Skipping news feature processing (--skip_news flag)")
    
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
