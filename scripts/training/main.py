"""Main training script - orchestrates model training pipeline."""
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.experiments as experiments
from src.models import list_models, get_model
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
    horizon: str = None,
    **override_kwargs
):
    """
    Create a model instance from hyperparameter grid.
    
    Args:
        model_name: Name of the model
        grid: Hyperparameter grid dictionary
        grid_index: Index to use when extracting from grid (default: 0)
        horizon: Target horizon (for horizon-specific configs)
        **override_kwargs: Additional parameters to override
    
    Returns:
        Model instance
    """
    params = get_grid_params(grid, grid_index)
    
    # Apply horizon-specific configurations if available
    if horizon and hasattr(experiments, 'HORIZON_SPECIFIC_CONFIG'):
        horizon_config = experiments.HORIZON_SPECIFIC_CONFIG.get(horizon, {})
        if model_name in horizon_config:
            params.update(horizon_config[model_name])
    
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
    
    # Configuration: all registered models that have a grid in experiments.py
    MODELS = [m for m in list_models() if m in getattr(experiments, "MODEL_GRIDS", {})]
    if not MODELS:
        MODELS = list(getattr(experiments, "MODEL_GRIDS", {}).keys())
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
    
    # Collect fold-0 comparison rows for terminal table
    comparison_rows = []

    def _fmt_float(x, digits=6):
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    def _print_comparison_table(rows, horizon: str):
        if not rows:
            return
        cols = ["model", "fold", "horizon", "RMSE", "MAE", "R2", "DirAcc", "AvgPnL", "Vol", "Sharpe", "Turnover"]
        widths = {c: len(c) for c in cols}
        for r in rows:
            for c in cols:
                widths[c] = max(widths[c], len(str(r.get(c, ""))))

        def line(sep="-"):
            return sep * (sum(widths.values()) + 3 * (len(cols) - 1))

        print("\n" + line("="))
        print(f"Table: Preliminary test metrics (fold 0, {horizon}). Sharpe from toy sign backtest (1 bp cost)")
        print(line("="))
        header = "   ".join([c.ljust(widths[c]) for c in cols])
        print(header)
        print(line("-"))
        for r in rows:
            print("   ".join([str(r.get(c, "")).ljust(widths[c]) for c in cols]))
        print(line("=") + "\n")

    def _print_overfitting_table(rows, horizon: str):
        if not rows:
            return
        cols = ["model", "fold", "horizon", "Train_RMSE", "Test_RMSE", "Train_R2", "Test_R2", "Train_DirAcc", "Test_DirAcc", "Overfit"]
        widths = {c: len(c) for c in cols}
        for r in rows:
            for c in cols:
                widths[c] = max(widths[c], len(str(r.get(c, ""))))

        def line(sep="-"):
            return sep * (sum(widths.values()) + 3 * (len(cols) - 1))

        print(line("="))
        print(f"Table: Train vs Test (overfitting check, fold 0, {horizon}). Overfit=Y if test RMSE>1.15*train RMSE or train DirAcc - test DirAcc > 0.08")
        print(line("="))
        header = "   ".join([c.ljust(widths[c]) for c in cols])
        print(header)
        print(line("-"))
        for r in rows:
            print("   ".join([str(r.get(c, "")).ljust(widths[c]) for c in cols]))
        print(line("=") + "\n")

    def _print_interpretation():
        print("How to read these metrics (how each model is making a prediction):")
        print("  DirAcc (directional accuracy): Fraction of days the model predicted the correct sign of the return (up/down). Higher = better at direction.")
        print("  RMSE/MAE: Average prediction error magnitude. Lower = smaller errors.")
        print("  R2: Variance explained; negative = worse than predicting the mean.")
        print("  Sharpe/Turnover: From a toy strategy that goes long when pred>0, short when pred<0 (1 bp cost). Sharpe = risk-adjusted return; Turnover = trading frequency.")
        print("  Overfit: Y = train performance notably better than test (model may be memorizing). N = train and test more in line.")
        print("")

    # Run experiments for each model
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # Get grid from experiments (MODEL_GRIDS maps every registered model with a grid)
        grid = experiments.MODEL_GRIDS.get(model_name)
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

        # Capture fold-0 rows for each horizon for terminal comparison (test + train for overfitting)
        try:
            for hz in HORIZONS:
                sub = results_df[(results_df["fold"] == 0) & (results_df["horizon"] == hz)]
                if len(sub) == 0:
                    continue
                row = sub.iloc[0]
                tm = row.get("test_metrics", {}) or {}
                trm = row.get("train_metrics", {}) or {}
                train_rmse = float(trm.get("rmse") or 0)
                test_rmse = float(tm.get("rmse") or 0)
                train_da = float(trm.get("dir_acc") or 0)
                test_da = float(tm.get("dir_acc") or 0)
                # Overfit?: test RMSE notably worse than train, or train DirAcc notably higher than test
                overfit = (train_rmse > 1e-9 and test_rmse > train_rmse * 1.15) or (train_da - test_da > 0.08)
                comparison_rows.append({
                    "model": model_name,
                    "fold": str(int(row.get("fold", 0))),
                    "horizon": hz.replace("target_", ""),
                    "RMSE": _fmt_float(tm.get("rmse"), 6),
                    "MAE": _fmt_float(tm.get("mae"), 6),
                    "R2": _fmt_float(tm.get("r2"), 3),
                    "DirAcc": _fmt_float(tm.get("dir_acc"), 3),
                    "AvgPnL": _fmt_float(tm.get("avg_pnl"), 6),
                    "Vol": _fmt_float(tm.get("vol"), 6),
                    "Sharpe": _fmt_float(tm.get("sharpe"), 3),
                    "Turnover": _fmt_float(tm.get("turnover"), 3),
                    "Train_RMSE": _fmt_float(trm.get("rmse"), 6),
                    "Test_RMSE": _fmt_float(tm.get("rmse"), 6),
                    "Train_R2": _fmt_float(trm.get("r2"), 3),
                    "Test_R2": _fmt_float(tm.get("r2"), 3),
                    "Train_DirAcc": _fmt_float(trm.get("dir_acc"), 3),
                    "Test_DirAcc": _fmt_float(tm.get("dir_acc"), 3),
                    "Overfit": "Y" if overfit else "N",
                })
        except Exception:
            pass
        
        print(f"\n{model_name.upper()} completed. Results shape: {results_df.shape}")

    # Print comparison tables per horizon (fold 0) — same order as MODELS for consistency
    try:
        model_order = {m: i for i, m in enumerate(MODELS)}
        for hz in HORIZONS:
            hz_key = hz.replace("target_", "")
            rows = [r for r in comparison_rows if r.get("horizon") == hz_key]
            rows.sort(key=lambda r: model_order.get(r.get("model", ""), 999))
            h_label = f"h={hz_key.replace('h', '') if hz_key.startswith('h') else hz_key}"
            _print_comparison_table(rows, horizon=h_label)
            _print_overfitting_table(rows, horizon=h_label)
        _print_interpretation()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("Training Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
