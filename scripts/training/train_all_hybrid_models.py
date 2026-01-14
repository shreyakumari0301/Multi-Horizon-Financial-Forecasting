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
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.experiments as experiments
from src.models.registry import get_model
from src.train.runner import run_experiment, load_fold_data, compute_metrics


def get_grid_params(grid: Dict[str, list], index: int = 0) -> Dict[str, Any]:
    """Extract hyperparameters from a grid by taking the first value of each list."""
    return {k: v[index] if isinstance(v, list) else v for k, v in grid.items()}


def create_model(model_name: str, grid: Dict[str, list], grid_index: int = 0, **override_kwargs):
    """Create a model instance from hyperparameter grid."""
    params = get_grid_params(grid, grid_index)
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
        "lstm": experiments.LSTM_GRID,
        "transformer": experiments.TRANSFORMER_GRID,
        "tcn": experiments.TCN_GRID,
        "ridge": experiments.RIDGE_GRID,
    }
    
    # Train all base models
    base_models = []
    model_results = {}
    
    print(f"\n  Training base models for fold {fold}, {horizon}...")
    
    for model_name in ["ridge", "lstm", "transformer", "tcn"]:
        grid = grid_map.get(model_name)
        if grid is None:
            continue
        
        print(f"    - {model_name.upper()}")
        
        # Create and train model
        model = create_model(model_name, grid, grid_index=0)
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
    
    # Create hybrid ensemble with Ridge as baseline voter (higher weight)
    # Weights: Ridge=0.4, LSTM=0.2, Transformer=0.3, TCN=0.1
    # Order: [Ridge, LSTM, Transformer, TCN] matches base_models order
    weights = [0.4, 0.2, 0.3, 0.1]  # Ridge, LSTM, Transformer, TCN
    hybrid = HybridEnsemble(base_models, weights=weights)
    
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
            "y_pred_lstm": base_models[1].predict(X_test),
            "y_pred_transformer": base_models[2].predict(X_test),
            "y_pred_tcn": base_models[3].predict(X_test),
        }, index=test_index)
        pred_path = os.path.join(exp_dir, f"{horizon}_predictions.csv")
        pred_df.to_csv(pred_path)
        results["predictions_path"] = pred_path
        results["results_path"] = results_path
    
    return results


def main():
    """Train hybrid models on all folds and horizons."""
    
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
