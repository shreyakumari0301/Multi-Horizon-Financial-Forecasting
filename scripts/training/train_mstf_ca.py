"""Train MSTF-CA model on all three horizons (h1, h5, h20).

This script trains the novel Multi-Scale Temporal Fusion Network with Cross-Attention
on all three prediction horizons for comprehensive evaluation.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.experiments as experiments
from src.models.registry import get_model
from src.train.runner import run_experiment, load_fold_data, compute_metrics
import numpy as np
import pandas as pd
import json


def get_grid_params(grid, index: int = 0):
    """Extract hyperparameters from grid."""
    return {k: v[index] if isinstance(v, list) else v for k, v in grid.items()}


def create_model(model_name: str, grid, grid_index: int = 0, horizon: str = None, **override_kwargs):
    """Create model with horizon-specific configs."""
    params = get_grid_params(grid, grid_index)
    
    # Apply horizon-specific configurations
    if horizon and hasattr(experiments, 'HORIZON_SPECIFIC_CONFIG'):
        horizon_config = experiments.HORIZON_SPECIFIC_CONFIG.get(horizon, {})
        if model_name in horizon_config:
            params.update(horizon_config[model_name])
            print(f"    Using horizon-specific config: {horizon_config[model_name]}")
    
    params.update(override_kwargs)
    return get_model(model_name, **params)


def train_mstf_ca_all_horizons(
    folds: list = None,
    splits_dir: str = "data/splits",
    results_dir: str = "data/experiments",
    models_dir: str = "data/models"
):
    """
    Train MSTF-CA on all three horizons.
    
    Args:
        folds: List of fold numbers (default: [0] for quick test, use range(9) for all)
        splits_dir: Directory containing fold data
        results_dir: Directory to save results
        models_dir: Directory to save trained models
    """
    if folds is None:
        folds = [0]  # Default to fold 0 for testing
    
    horizons = ["target_h1", "target_h5", "target_h20"]
    model_name = "mstf_ca"
    grid = experiments.MSTF_CA_GRID
    
    print("=" * 70)
    print("MSTF-CA Training - All Horizons")
    print("=" * 70)
    print(f"Model: {model_name.upper()}")
    print(f"Folds: {folds}")
    print(f"Horizons: {horizons}")
    print(f"Splits: {splits_dir}")
    print(f"Results: {results_dir}")
    print(f"Models: {models_dir}")
    print("=" * 70)
    
    all_results = []
    
    for fold in folds:
        for horizon in horizons:
            print(f"\n{'='*70}")
            print(f"Training {model_name.upper()} | Fold {fold} | {horizon}")
            print(f"{'='*70}")
            
            # Load data
            fold_dir = os.path.join(splits_dir, f"fold_{fold}")
            try:
                X_train, y_train, X_test, y_test, test_index = load_fold_data(
                    fold_dir, horizon, include_news=True
                )
            except FileNotFoundError:
                print(f"⚠ Skipping fold {fold} - data not found")
                continue
            
            # Create model with horizon-specific config
            model = create_model(model_name, grid, grid_index=0, horizon=horizon)
            
            # Train
            print(f"Training on {len(X_train)} samples, {X_train.shape[1]} features...")
            model.fit(X_train, y_train)
            
            # Predict
            print("Generating predictions...")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Handle delta targets for metrics
            y_train_true = y_train.copy()
            y_test_true = y_test.copy()
            if hasattr(model, 'use_delta_target') and model.use_delta_target:
                # Reconstruct from delta predictions
                y_train_recon = np.zeros_like(y_train)
                y_train_recon[0] = y_train[0] if len(y_train) > 0 else 0
                for i in range(1, len(y_train)):
                    y_train_recon[i] = y_train_recon[i-1] + y_train_pred[i]
                
                y_test_recon = np.zeros_like(y_test)
                if len(y_test) > 0:
                    # Use last training value as starting point
                    y_test_recon[0] = y_train[-1] + y_test_pred[0] if len(y_train) > 0 else y_test_pred[0]
                    for i in range(1, len(y_test)):
                        y_test_recon[i] = y_test_recon[i-1] + y_test_pred[i]
                
                # For metrics, compare reconstructed predictions to original targets
                y_train_true = y_train
                y_test_true = y_test
                y_train_pred = y_train_recon
                y_test_pred = y_test_recon
            
            # Compute metrics
            train_metrics = compute_metrics(y_train_true, y_train_pred)
            test_metrics = compute_metrics(y_test_true, y_test_pred)
            
            print(f"\nTrain Metrics:")
            print(f"  RMSE: {train_metrics['rmse']:.6f}")
            print(f"  MAE:  {train_metrics['mae']:.6f}")
            print(f"  DirAcc: {train_metrics['dir_acc']:.3f}")
            
            print(f"\nTest Metrics:")
            print(f"  RMSE: {test_metrics['rmse']:.6f}")
            print(f"  MAE:  {test_metrics['mae']:.6f}")
            print(f"  DirAcc: {test_metrics['dir_acc']:.3f}")
            
            # Save model
            model_dir = os.path.join(models_dir, model_name, f"fold_{fold}")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{horizon}.pkl")
            
            # Save model state (PyTorch model)
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved model → {model_path}")
            
            # Save results
            results = {
                "fold": fold,
                "horizon": horizon,
                "model_name": model_name,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
            }
            
            exp_dir = os.path.join(results_dir, model_name, f"fold_{fold}")
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save metrics JSON
            results_path = os.path.join(exp_dir, f"{horizon}_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Save predictions CSV
            pred_df = pd.DataFrame({
                "y_true": y_test_true,
                "y_pred": y_test_pred,
            }, index=test_index)
            pred_path = os.path.join(exp_dir, f"{horizon}_predictions.csv")
            pred_df.to_csv(pred_path)
            results["predictions_path"] = pred_path
            results["results_path"] = results_path
            
            all_results.append(results)
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(results_dir, f"{model_name}_summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"\nSummary saved → {summary_path}")
        print(f"\nResults Summary:")
        print(results_df[['fold', 'horizon', 'test_metrics']].to_string())
        
        # Print aggregated metrics by horizon
        print(f"\n{'='*70}")
        print("Aggregated Metrics by Horizon:")
        print(f"{'='*70}")
        for horizon in horizons:
            horizon_results = results_df[results_df['horizon'] == horizon]
            if len(horizon_results) > 0:
                test_rmse = [r['rmse'] for r in horizon_results['test_metrics']]
                test_mae = [r['mae'] for r in horizon_results['test_metrics']]
                test_dir = [r['dir_acc'] for r in horizon_results['test_metrics']]
                
                print(f"\n{horizon}:")
                print(f"  RMSE: {np.mean(test_rmse):.6f} ± {np.std(test_rmse):.6f}")
                print(f"  MAE:  {np.mean(test_mae):.6f} ± {np.std(test_mae):.6f}")
                print(f"  DirAcc: {np.mean(test_dir):.3f} ± {np.std(test_dir):.3f}")
    
    return results_df if all_results else None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MSTF-CA on all horizons")
    parser.add_argument(
        "--folds",
        type=str,
        default="0",
        help="Comma-separated list of folds (e.g., '0,1,2' or 'all' for all 9 folds)"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing fold data"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/experiments",
        help="Directory to save results"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="data/models",
        help="Directory to save models"
    )
    
    args = parser.parse_args()
    
    # Parse folds
    if args.folds.lower() == "all":
        folds = list(range(9))
    else:
        folds = [int(f.strip()) for f in args.folds.split(",")]
    
    # Train
    train_mstf_ca_all_horizons(
        folds=folds,
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir
    )


if __name__ == "__main__":
    main()
