"""
Train Hybrid models on ALL folds and ALL horizons.

This script:
1. Trains hybrid models on all 9 folds (fold_0 to fold_8)
2. For all 3 horizons (target_h1, target_h5, target_h20)
3. Saves all 27 models to disk
4. Generates comprehensive results
"""
import sys
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.experiments as experiments
from src.models.registry import get_model
from src.train.runner import run_experiment, load_fold_data, compute_metrics

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, will use performance-based weights")


def get_grid_params(grid: Dict[str, list], index: int = 0) -> Dict[str, Any]:
    """Extract hyperparameters from a grid by taking the first value of each list."""
    return {k: v[index] if isinstance(v, list) else v for k, v in grid.items()}


def create_model(model_name: str, grid: Dict[str, list], grid_index: int = 0, horizon: str = None, **override_kwargs):
    """
    Create a model instance from hyperparameter grid.
    
    Args:
        model_name: Name of the model
        grid: Hyperparameter grid
        grid_index: Index to use from grid
        horizon: Target horizon (for horizon-specific configs)
        **override_kwargs: Additional overrides
    """
    params = get_grid_params(grid, grid_index)
    
    # Apply horizon-specific configurations if available
    if horizon and hasattr(experiments, 'HORIZON_SPECIFIC_CONFIG'):
        horizon_config = experiments.HORIZON_SPECIFIC_CONFIG.get(horizon, {})
        if model_name in horizon_config:
            params.update(horizon_config[model_name])
            print(f"    Using horizon-specific config for {model_name}: {horizon_config[model_name]}")
    
    params.update(override_kwargs)
    return get_model(model_name, **params)


class HybridEnsemble:
    """
    Hybrid ensemble model that combines predictions from multiple base models.
    Uses weighted voting with Ridge as baseline voter.
    """
    def __init__(self, base_models: List, weights: List[float] = None):
        """
        Args:
            base_models: List of fitted model instances
            weights: Weights for each model (None = equal weights)
        """
        self.base_models = base_models
        if weights is None:
            weights = [1.0 / len(base_models)] * len(base_models)
        self.weights = np.array(weights, dtype=np.float32)
        self.weights = self.weights / self.weights.sum()  # Normalize
        self._fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of base model predictions."""
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            if pred.ndim == 1:
                pred = pred[:, None]
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)  # (n_models, n_samples, n_outputs)
        weighted_pred = np.tensordot(self.weights, predictions, axes=1)
        
        return weighted_pred.squeeze() if weighted_pred.shape[-1] == 1 else weighted_pred
    
    def __repr__(self):
        model_names = [m.__class__.__name__ for m in self.base_models]
        return f"HybridEnsemble(models={model_names}, weights={self.weights})"


def optimize_ensemble_weights(
    base_models: List,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "dir_acc",
    uses_delta: bool = False
) -> np.ndarray:
    """
    Optimize ensemble weights based on validation performance.
    
    Args:
        base_models: List of fitted base models
        X_val: Validation features
        y_val: Validation targets
        metric: Metric to optimize ("dir_acc", "rmse", "mae")
        uses_delta: Whether models use delta targets
    
    Returns:
        Optimized weights array
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for weight optimization")
    # Get predictions from all models
    predictions = []
    for model in base_models:
        pred = model.predict(X_val)
        if pred.ndim == 1:
            pred = pred[:, None]
        predictions.append(pred)
    predictions = np.array(predictions)  # (n_models, n_samples, n_outputs)
    
    # Handle delta targets
    y_val_true = y_val.copy()
    if uses_delta:
        y_val_delta = np.zeros_like(y_val)
        y_val_delta[1:] = y_val[1:] - y_val[:-1]
        y_val_delta[0] = y_val[0]
        y_val_true = y_val_delta
    
    def objective(weights):
        """Objective function to minimize (negative of metric to maximize)."""
        # Normalize weights
        weights = np.maximum(weights, 0)  # Ensure non-negative
        weights = weights / (weights.sum() + 1e-10)  # Normalize
        
        # Weighted ensemble prediction
        ensemble_pred = np.tensordot(weights, predictions, axes=1).squeeze()
        
        # Compute metric
        if metric == "dir_acc":
            # Maximize directional accuracy
            dir_acc = np.mean(np.sign(ensemble_pred) == np.sign(y_val_true))
            return -dir_acc  # Negative because we minimize
        elif metric == "rmse":
            rmse = np.sqrt(np.mean((ensemble_pred - y_val_true) ** 2))
            return rmse
        elif metric == "mae":
            mae = np.mean(np.abs(ensemble_pred - y_val_true))
            return mae
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Initial weights (equal)
    n_models = len(base_models)
    initial_weights = np.ones(n_models) / n_models
    
    # Constraints: weights sum to 1, all >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    
    # Optimize with multiple methods for robustness
    methods = ['SLSQP', 'L-BFGS-B']
    best_result = None
    best_value = np.inf
    
    for method in methods:
        try:
            result = minimize(
                objective,
                initial_weights,
                method=method,
                bounds=bounds,
                constraints=constraints if method == 'SLSQP' else None,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success and result.fun < best_value:
                best_result = result
                best_value = result.fun
        except Exception as e:
            continue
    
    if best_result is not None and best_result.success:
        optimal_weights = np.maximum(best_result.x, 0)  # Ensure non-negative
        optimal_weights = optimal_weights / (optimal_weights.sum() + 1e-10)  # Normalize
        
        # Check if weights are meaningful (not all equal)
        if np.std(optimal_weights) < 0.01:
            # Weights are too similar, use performance-based instead
            return compute_performance_based_weights(base_models, X_val, y_val, uses_delta)
        
        return optimal_weights
    else:
        # Fallback to performance-based weights
        return compute_performance_based_weights(base_models, X_val, y_val, uses_delta)


def compute_performance_based_weights(
    base_models: List,
    X_val: np.ndarray,
    y_val: np.ndarray,
    uses_delta: bool = False
) -> np.ndarray:
    """
    Compute weights based on individual model performance (inverse RMSE).
    
    Args:
        base_models: List of fitted base models
        X_val: Validation features
        y_val: Validation targets
        uses_delta: Whether models use delta targets
    
    Returns:
        Performance-based weights
    """
    y_val_true = y_val.copy()
    if uses_delta:
        y_val_delta = np.zeros_like(y_val)
        y_val_delta[1:] = y_val[1:] - y_val[:-1]
        y_val_delta[0] = y_val[0]
        y_val_true = y_val_delta
    
    performances = []
    for model in base_models:
        pred = model.predict(X_val)
        # Use directional accuracy as performance metric
        dir_acc = np.mean(np.sign(pred) == np.sign(y_val_true))
        performances.append(dir_acc)
    
    # Convert to weights (higher performance = higher weight)
    performances = np.array(performances)
    # Add small epsilon to avoid zero weights
    performances = performances + 0.01
    weights = performances / performances.sum()
    
    return weights


def save_model(model, save_path: str):
    """Save model to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # For PyTorch models, save state dict
    if hasattr(model, '_model') and model._model is not None:
        # Deep learning model
        torch_model = model._model
        if hasattr(torch_model, 'state_dict'):
            torch.save({
                'model_state_dict': torch_model.state_dict(),
                'model_class': model.__class__.__name__,
                'hyperparams': {k: v for k, v in model.__dict__.items() 
                               if not k.startswith('_') and k != 'device'},
            }, save_path.replace('.pkl', '.pth'))
    
    # For scikit-learn models or ensemble, use pickle
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Warning: Could not pickle model: {e}")
        # Save metadata only
        metadata = {
            'model_class': model.__class__.__name__,
            'hyperparams': {k: str(v) for k, v in model.__dict__.items() 
                          if not k.startswith('_')}
        }
        with open(save_path.replace('.pkl', '_meta.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def train_hybrid_ensemble(
    fold: int,
    horizon: str,
    splits_dir: str = "data/splits",
    results_dir: str = "data/experiments",
    models_dir: str = "data/models",
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Train a hybrid ensemble model for a specific fold and horizon.
    
    Args:
        fold: Fold number
        horizon: Target horizon
        splits_dir: Directory containing fold data
        results_dir: Directory to save results
        models_dir: Directory to save trained models
        save_models: Whether to save models to disk
    
    Returns:
        Dictionary with results
    """
    fold_dir = os.path.join(splits_dir, f"fold_{fold}")
    
    # Load data
    X_train, y_train, X_test, y_test, test_index = load_fold_data(fold_dir, horizon)
    
    # Grid maps
    grid_map = {
        "ridge": experiments.RIDGE_GRID,
        "esn": experiments.ESN_GRID,
        "lstm": experiments.LSTM_GRID,
        "transformer": experiments.TRANSFORMER_GRID,
        "tcn": experiments.TCN_GRID,
    }
    
    # Train all base models
    base_models = []
    model_results = {}
    
    print(f"\n  Training base models for fold {fold}, {horizon}...")
    
    for model_name in ["ridge", "esn", "lstm", "transformer", "tcn"]:
        grid = grid_map.get(model_name)
        if grid is None:
            continue
        
        print(f"    - {model_name.upper()}")
        
        # Create and train model with horizon-specific config
        model = create_model(model_name, grid, grid_index=0, horizon=horizon)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Handle delta targets for metrics
        y_train_true = y_train.copy()
        y_test_true = y_test.copy()
        if hasattr(model, 'use_delta_target') and model.use_delta_target:
            y_train_delta = np.zeros_like(y_train)
            y_train_delta[1:] = y_train[1:] - y_train[:-1]
            y_train_delta[0] = y_train[0]
            y_train_true = y_train_delta
            
            y_test_delta = np.zeros_like(y_test)
            y_test_delta[1:] = y_test[1:] - y_test[:-1]
            y_test_delta[0] = y_test[0]
            y_test_true = y_test_delta
        
        train_metrics = compute_metrics(y_train_true, y_train_pred)
        test_metrics = compute_metrics(y_test_true, y_test_pred)
        
        model_results[model_name] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        
        # Save individual model
        if save_models:
            model_path = os.path.join(models_dir, f"{model_name}", f"fold_{fold}", f"{horizon}.pkl")
            save_model(model, model_path)
        
        base_models.append(model)
    
    # Split training data for weight optimization
    # Use last 20% of training data as validation for weight optimization
    val_frac = 0.2
    n_val = max(1, int(val_frac * len(X_train)))
    n_tr = len(X_train) - n_val
    
    X_tr_opt = X_train[:n_tr]
    y_tr_opt = y_train[:n_tr]
    X_val_opt = X_train[n_tr:]
    y_val_opt = y_train[n_tr:]
    
    # Check if any model uses delta targets
    uses_delta = any(hasattr(m, 'use_delta_target') and m.use_delta_target for m in base_models)
    
    # Optimize ensemble weights based on validation performance
    print(f"\n  Optimizing ensemble weights on validation set...")
    if HAS_SCIPY:
        try:
            optimal_weights = optimize_ensemble_weights(
                base_models=base_models,
                X_val=X_val_opt,
                y_val=y_val_opt,
                metric="dir_acc",  # Optimize for directional accuracy
                uses_delta=uses_delta
            )
            print(f"  Optimized weights: {optimal_weights}")
            print(f"    Ridge: {optimal_weights[0]:.3f}, ESN: {optimal_weights[1]:.3f}, "
                  f"LSTM: {optimal_weights[2]:.3f}, Transformer: {optimal_weights[3]:.3f}, TCN: {optimal_weights[4]:.3f}")
        except Exception as e:
            print(f"  Warning: Weight optimization failed ({e}), using performance-based weights")
            optimal_weights = compute_performance_based_weights(
                base_models, X_val_opt, y_val_opt, uses_delta
            )
    else:
        print(f"  Using performance-based weights (scipy not available)")
        optimal_weights = compute_performance_based_weights(
            base_models, X_val_opt, y_val_opt, uses_delta
        )
        print(f"  Performance-based weights: {optimal_weights}")
        print(f"    Ridge: {optimal_weights[0]:.3f}, ESN: {optimal_weights[1]:.3f}, "
              f"LSTM: {optimal_weights[2]:.3f}, Transformer: {optimal_weights[3]:.3f}, TCN: {optimal_weights[4]:.3f}")
    
    # Create hybrid ensemble with optimized weights
    hybrid = HybridEnsemble(base_models, weights=optimal_weights)
    
    # Evaluate hybrid
    y_train_pred_hybrid = hybrid.predict(X_train)
    y_test_pred_hybrid = hybrid.predict(X_test)
    
    # Handle delta targets
    y_train_true_hybrid = y_train.copy()
    y_test_true_hybrid = y_test.copy()
    # Check if any model uses delta
    uses_delta = any(hasattr(m, 'use_delta_target') and m.use_delta_target for m in base_models)
    if uses_delta:
        y_train_delta = np.zeros_like(y_train)
        y_train_delta[1:] = y_train[1:] - y_train[:-1]
        y_train_delta[0] = y_train[0]
        y_train_true_hybrid = y_train_delta
        
        y_test_delta = np.zeros_like(y_test)
        y_test_delta[1:] = y_test[1:] - y_test[:-1]
        y_test_delta[0] = y_test[0]
        y_test_true_hybrid = y_test_delta
    
    hybrid_train_metrics = compute_metrics(y_train_true_hybrid, y_train_pred_hybrid)
    hybrid_test_metrics = compute_metrics(y_test_true_hybrid, y_test_pred_hybrid)
    
    # Save hybrid model
    if save_models:
        hybrid_path = os.path.join(models_dir, "hybrid", f"fold_{fold}", f"{horizon}.pkl")
        save_model(hybrid, hybrid_path)
    
    # Prepare results
    results = {
        "fold": fold,
        "horizon": horizon,
        "base_models": model_results,
        "hybrid": {
            "train_metrics": hybrid_train_metrics,
            "test_metrics": hybrid_test_metrics,
            "weights": hybrid.weights.tolist(),
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    
    # Save results
    if results_dir:
        exp_dir = os.path.join(results_dir, "hybrid", f"fold_{fold}")
        os.makedirs(exp_dir, exist_ok=True)
        
        results_path = os.path.join(exp_dir, f"{horizon}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred_hybrid": y_test_pred_hybrid,
            "y_pred_ridge": base_models[0].predict(X_test),
            "y_pred_esn": base_models[1].predict(X_test),
            "y_pred_lstm": base_models[2].predict(X_test),
            "y_pred_transformer": base_models[3].predict(X_test),
            "y_pred_tcn": base_models[4].predict(X_test),
        }, index=test_index)
        pred_path = os.path.join(exp_dir, f"{horizon}_predictions.csv")
        pred_df.to_csv(pred_path)
        results["predictions_path"] = pred_path
        results["results_path"] = results_path
    
    return results


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
    fold_0_dir = os.path.join(splits_dir, "fold_0")
    train_path = os.path.join(fold_0_dir, "train.csv")
    
    if os.path.exists(train_path):
        train = pd.read_csv(train_path, index_col=0, nrows=1)  # Just check columns
        news_cols = [c for c in train.columns if c.startswith("z_news_pc")]
        if len(news_cols) > 0:
            print(f"✓ News features already integrated: {len(news_cols)} features found")
            return True
    
    # News features not integrated - check if we can process them
    news_features_path = os.path.join(processed_dir, "news_features_28d.csv")
    
    # If processed news features exist, integrate them
    if os.path.exists(news_features_path):
        print("\n" + "=" * 70)
        print("Integrating Existing News Features")
        print("=" * 70)
        
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
        print("\n" + "=" * 70)
        print("Processing News Features from Raw Data")
        print("=" * 70)
        
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
    """Train hybrid models on all folds and horizons."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train hybrid ensemble models")
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
    
    # All folds and horizons
    ALL_FOLDS = list(range(9))  # fold_0 to fold_8
    ALL_HORIZONS = ["target_h1", "target_h5", "target_h20"]
    
    # Paths
    SPLITS_DIR = "data/splits"
    RESULTS_DIR = "data/experiments"
    MODELS_DIR = "data/models"
    
    print("=" * 70)
    print("Hybrid Model Training - All Folds & Horizons")
    print("=" * 70)
    print(f"Folds: {ALL_FOLDS} ({len(ALL_FOLDS)} folds)")
    print(f"Horizons: {ALL_HORIZONS} ({len(ALL_HORIZONS)} horizons)")
    print(f"Total experiments: {len(ALL_FOLDS) * len(ALL_HORIZONS)}")
    print(f"Splits: {SPLITS_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Models: {MODELS_DIR}")
    print("=" * 70)
    
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
    
    print("=" * 70)
    
    all_results = []
    
    # Train on all combinations
    for fold in ALL_FOLDS:
        for horizon in ALL_HORIZONS:
            print(f"\n{'='*70}")
            print(f"Fold {fold} | {horizon}")
            print(f"{'='*70}")
            
            try:
                results = train_hybrid_ensemble(
                    fold=fold,
                    horizon=horizon,
                    splits_dir=SPLITS_DIR,
                    results_dir=RESULTS_DIR,
                    models_dir=MODELS_DIR,
                    save_models=True
                )
                
                all_results.append(results)
                
                # Print summary
                hybrid_metrics = results["hybrid"]["test_metrics"]
                print(f"\n✓ Hybrid Test Metrics:")
                print(f"  RMSE: {hybrid_metrics['rmse']:.6f}")
                print(f"  MAE: {hybrid_metrics['mae']:.6f}")
                print(f"  DirAcc: {hybrid_metrics['dir_acc']:.3f}")
                
            except Exception as e:
                print(f"✗ Error training fold {fold}, {horizon}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create summary
    if all_results:
        summary_data = []
        for r in all_results:
            summary_data.append({
                "fold": r["fold"],
                "horizon": r["horizon"],
                "hybrid_rmse": r["hybrid"]["test_metrics"]["rmse"],
                "hybrid_mae": r["hybrid"]["test_metrics"]["mae"],
                "hybrid_dir_acc": r["hybrid"]["test_metrics"]["dir_acc"],
                "ridge_rmse": r["base_models"]["ridge"]["test_metrics"]["rmse"],
                "ridge_dir_acc": r["base_models"]["ridge"]["test_metrics"]["dir_acc"],
                "transformer_rmse": r["base_models"]["transformer"]["test_metrics"]["rmse"],
                "transformer_dir_acc": r["base_models"]["transformer"]["test_metrics"]["dir_acc"],
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(RESULTS_DIR, "hybrid", "summary_all_folds_horizons.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total experiments completed: {len(all_results)}")
        print(f"Summary saved to: {summary_path}")
        print(f"\nAverage Hybrid Performance:")
        print(f"  RMSE: {summary_df['hybrid_rmse'].mean():.6f}")
        print(f"  MAE: {summary_df['hybrid_mae'].mean():.6f}")
        print(f"  DirAcc: {summary_df['hybrid_dir_acc'].mean():.3f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
