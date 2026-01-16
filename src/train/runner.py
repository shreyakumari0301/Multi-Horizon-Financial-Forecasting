"""Training runner for model execution and results storage."""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_fold_data(fold_dir: str, target_col: str, include_news: bool = True):
    """
    Load training and test data for a fold.
    
    Args:
        fold_dir: Directory containing train.csv and test.csv
        target_col: Target column name (e.g., "target_h1")
        include_news: Whether to include news features (default: True)
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, test_index)
    """
    train_path = os.path.join(fold_dir, "train.csv")
    test_path = os.path.join(fold_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing data files in {fold_dir}")
    
    train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    # Technical feature columns (starting with "z_" but not news)
    tech_cols = [c for c in train.columns if c.startswith("z_") and not c.startswith("z_news_pc")]
    
    # News feature columns (starting with "z_news_pc" after scaling)
    news_cols = []
    if include_news:
        news_cols = [c for c in train.columns if c.startswith("z_news_pc")]
    
    # Combine all feature columns
    feature_cols = tech_cols + news_cols
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in data")
    
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values
    test_index = test.index
    
    print(f"Loaded features: {len(tech_cols)} technical + {len(news_cols)} news = {len(feature_cols)} total")
    
    return X_train, y_train, X_test, y_test, test_index


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Directional accuracy
    if len(y_true) > 1:
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
        dir_acc = np.mean(true_dir == pred_dir)
    else:
        dir_acc = 0.0
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "dir_acc": float(dir_acc),
    }


def run_experiment(
    model,
    fold: int,
    horizon: str,
    splits_dir: str = "data/splits",
    results_dir: str = "data/experiments",
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experiment: train model and evaluate on test set.
    
    Args:
        model: Model instance with .fit() and .predict() methods
        fold: Fold number
        horizon: Target horizon (e.g., "target_h1")
        splits_dir: Directory containing fold data
        results_dir: Directory to save results
        save_predictions: Whether to save predictions to CSV
    
    Returns:
        Dictionary containing experiment results
    """
    fold_dir = os.path.join(splits_dir, f"fold_{fold}")
    
    # Load data
    X_train, y_train, X_test, y_test, test_index = load_fold_data(fold_dir, horizon)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Handle delta targets: if model uses delta, convert true values to delta for comparison
    y_train_true = y_train.copy()
    y_test_true = y_test.copy()
    if hasattr(model, 'use_delta_target') and model.use_delta_target:
        # Convert to delta for proper comparison
        y_train_delta = np.zeros_like(y_train)
        y_train_delta[1:] = y_train[1:] - y_train[:-1]
        y_train_delta[0] = y_train[0]
        y_train_true = y_train_delta
        
        y_test_delta = np.zeros_like(y_test)
        y_test_delta[1:] = y_test[1:] - y_test[:-1]
        y_test_delta[0] = y_test[0]
        y_test_true = y_test_delta
    
    # Compute metrics
    train_metrics = compute_metrics(y_train_true, y_train_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)
    
    # Prepare results
    results = {
        "fold": fold,
        "horizon": horizon,
        "model_name": model.__class__.__name__,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    }
    
    # Save results
    if results_dir:
        exp_dir = os.path.join(results_dir, f"{results['model_name']}", f"fold_{fold}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save metrics JSON
        results_path = os.path.join(exp_dir, f"{horizon}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save predictions CSV
        if save_predictions:
            # Save original y_test for reference, but predictions are delta if use_delta_target
            pred_df = pd.DataFrame({
                "y_true": y_test,  # Original target
                "y_pred": y_test_pred,  # Delta prediction if use_delta_target
                "y_true_delta": y_test_true if hasattr(model, 'use_delta_target') and model.use_delta_target else y_test,
            }, index=test_index)
            pred_path = os.path.join(exp_dir, f"{horizon}_predictions.csv")
            pred_df.to_csv(pred_path)
            results["predictions_path"] = pred_path
        
        results["results_path"] = results_path
    
    return results


def run_grid_search(
    model_name: str,
    grid: Dict[str, list],
    folds: list,
    horizons: list,
    splits_dir: str = "data/splits",
    results_dir: str = "data/experiments",
    grid_index: int = 0,
    create_model_fn=None,
    **override_kwargs
) -> pd.DataFrame:
    """
    Run grid search across folds and horizons.
    
    Args:
        model_name: Name of the model
        grid: Hyperparameter grid
        folds: List of fold numbers
        horizons: List of target horizons
        splits_dir: Directory containing fold data
        results_dir: Directory to save results
        grid_index: Index to use when extracting from grid
        create_model_fn: Function to create model (from main.py)
        **override_kwargs: Parameters to override
    
    Returns:
        DataFrame with all experiment results
    """
    if create_model_fn is None:
        raise ValueError("create_model_fn must be provided")
    
    all_results = []
    
    for fold in folds:
        for horizon in horizons:
            print(f"\n=== {model_name.upper()} | Fold {fold} | {horizon} ===")
            
            # Create model with grid parameters
            model = create_model_fn(
                model_name,
                grid=grid,
                grid_index=grid_index,
                **override_kwargs
            )
            
            # Run experiment
            results = run_experiment(
                model=model,
                fold=fold,
                horizon=horizon,
                splits_dir=splits_dir,
                results_dir=results_dir,
            )
            
            all_results.append(results)
            
            # Print summary
            test_metrics = results["test_metrics"]
            print(f"Test RMSE: {test_metrics['rmse']:.6f} | "
                  f"MAE: {test_metrics['mae']:.6f} | "
                  f"DirAcc: {test_metrics['dir_acc']:.3f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary
    if results_dir:
        summary_path = os.path.join(results_dir, f"{model_name}_summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary → {summary_path}")
    
    return results_df


__all__ = [
    "load_fold_data",
    "compute_metrics",
    "run_experiment",
    "run_grid_search",
]
