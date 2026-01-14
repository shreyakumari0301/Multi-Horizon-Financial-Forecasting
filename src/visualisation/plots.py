"""Visualization functions for comparing models with Ridge baseline."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_results(
    results_dir: str = "data/experiments",
    model_names: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    horizons: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load all experiment results into a DataFrame.
    
    Args:
        results_dir: Directory containing experiment results
        model_names: List of model names to load (None = all)
        folds: List of folds to load (None = all)
        horizons: List of horizons to load (None = all)
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    
    if model_names is None:
        model_names = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d))]
    
    for model_name in model_names:
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # Get all fold directories
        fold_dirs = [d for d in os.listdir(model_dir) 
                    if d.startswith("fold_") and os.path.isdir(os.path.join(model_dir, d))]
        
        for fold_dir in fold_dirs:
            fold_num = int(fold_dir.split("_")[1])
            if folds is not None and fold_num not in folds:
                continue
            
            fold_path = os.path.join(model_dir, fold_dir)
            
            # Get all result JSON files
            json_files = [f for f in os.listdir(fold_path) if f.endswith("_results.json")]
            
            for json_file in json_files:
                horizon = json_file.replace("_results.json", "")
                if horizons is not None and horizon not in horizons:
                    continue
                
                json_path = os.path.join(fold_path, json_file)
                
                try:
                    with open(json_path, "r") as f:
                        result = json.load(f)
                    
                    # Flatten nested metrics
                    flat_result = {
                        "model": model_name,
                        "fold": result["fold"],
                        "horizon": result["horizon"],
                        "n_train": result.get("n_train", 0),
                        "n_test": result.get("n_test", 0),
                        "n_features": result.get("n_features", 0),
                    }
                    
                    # Add train metrics
                    for key, value in result.get("train_metrics", {}).items():
                        flat_result[f"train_{key}"] = value
                    
                    # Add test metrics
                    for key, value in result.get("test_metrics", {}).items():
                        flat_result[f"test_{key}"] = value
                    
                    all_results.append(flat_result)
                except Exception as e:
                    print(f"Warning: Could not load {json_path}: {e}")
    
    return pd.DataFrame(all_results)


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    split: str = "test",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Compare metrics across models with Ridge as baseline.
    
    Args:
        results_df: DataFrame from load_results()
        metric: Metric to compare ("rmse", "mae", "dir_acc")
        split: "train" or "test"
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    metric_col = f"{split}_{metric}"
    
    if metric_col not in results_df.columns:
        raise ValueError(f"Metric {metric_col} not found in results")
    
    # Group by model and horizon
    comparison = results_df.groupby(["model", "horizon"])[metric_col].mean().reset_index()
    
    # Pivot for easier plotting
    pivot_df = comparison.pivot(index="horizon", columns="model", values=metric_col)
    
    # Ensure Ridge is first column for reference
    if "RidgeRegressor" in pivot_df.columns:
        cols = ["RidgeRegressor"] + [c for c in pivot_df.columns if c != "RidgeRegressor"]
        pivot_df = pivot_df[cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    pivot_df.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_title(f"{split.upper()} {metric.upper()} Comparison: Ridge vs Other Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions_comparison(
    results_dir: str = "data/experiments",
    fold: int = 0,
    horizon: str = "target_h1",
    models: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Compare predictions vs actuals for Ridge and other models.
    
    Args:
        results_dir: Directory containing experiment results
        fold: Fold number
        horizon: Target horizon
        models: List of model names (None = all)
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    if models is None:
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models[:4]):  # Max 4 models
        pred_path = os.path.join(results_dir, model_name, f"fold_{fold}", f"{horizon}_predictions.csv")
        
        if not os.path.exists(pred_path):
            continue
        
        pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
        max_val = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(pred_df["y_true"], pred_df["y_pred"])
        
        ax.set_title(f"{model_name}\nR² = {r2:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(models), 4):
        axes[idx].axis("off")
    
    plt.suptitle(f"Predictions vs Actuals Comparison (Fold {fold}, {horizon})", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_time_series_predictions(
    results_dir: str = "data/experiments",
    fold: int = 0,
    horizon: str = "target_h1",
    models: Optional[List[str]] = None,
    n_samples: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Plot time series of predictions vs actuals.
    
    Args:
        results_dir: Directory containing experiment results
        fold: Fold number
        horizon: Target horizon
        models: List of model names (None = all, Ridge first)
        n_samples: Number of samples to plot
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    if models is None:
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
    
    # Ensure Ridge is first
    if "RidgeRegressor" in models:
        models = ["RidgeRegressor"] + [m for m in models if m != "RidgeRegressor"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actuals once
    first_model = models[0]
    pred_path = os.path.join(results_dir, first_model, f"fold_{fold}", f"{horizon}_predictions.csv")
    
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        actuals = pred_df["y_true"].iloc[:n_samples]
        ax.plot(actuals.index, actuals.values, "k-", lw=2, label="Actual", alpha=0.7)
    
    # Plot predictions for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for model_name, color in zip(models, colors):
        pred_path = os.path.join(results_dir, model_name, f"fold_{fold}", f"{horizon}_predictions.csv")
        
        if not os.path.exists(pred_path):
            continue
        
        pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        predictions = pred_df["y_pred"].iloc[:n_samples]
        
        label = model_name.replace("Regressor", "")
        ax.plot(predictions.index, predictions.values, "-", color=color, 
               label=label, alpha=0.7, lw=1.5)
    
    ax.set_title(f"Time Series Predictions Comparison (Fold {fold}, {horizon})", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_heatmap(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    split: str = "test",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create heatmap of metrics across models and horizons.
    
    Args:
        results_df: DataFrame from load_results()
        metric: Metric to compare ("rmse", "mae", "dir_acc")
        split: "train" or "test"
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    metric_col = f"{split}_{metric}"
    
    if metric_col not in results_df.columns:
        raise ValueError(f"Metric {metric_col} not found in results")
    
    # Average across folds
    comparison = results_df.groupby(["model", "horizon"])[metric_col].mean().reset_index()
    
    # Pivot for heatmap
    pivot_df = comparison.pivot(index="model", columns="horizon", values=metric_col)
    
    # Reorder to put Ridge first
    if "RidgeRegressor" in pivot_df.index:
        new_order = ["RidgeRegressor"] + [m for m in pivot_df.index if m != "RidgeRegressor"]
        pivot_df = pivot_df.reindex(new_order)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlOrRd", 
                cbar_kws={"label": metric.upper()}, ax=ax, linewidths=0.5)
    
    ax.set_title(f"{split.upper()} {metric.upper()} Heatmap: Models vs Horizons", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fold_comparison(
    results_df: pd.DataFrame,
    metric: str = "rmse",
    split: str = "test",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Compare metrics across folds for all models.
    
    Args:
        results_df: DataFrame from load_results()
        metric: Metric to compare ("rmse", "mae", "dir_acc")
        split: "train" or "test"
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    metric_col = f"{split}_{metric}"
    
    if metric_col not in results_df.columns:
        raise ValueError(f"Metric {metric_col} not found in results")
    
    # Average across horizons
    comparison = results_df.groupby(["model", "fold"])[metric_col].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique models and ensure Ridge is first
    models = comparison["model"].unique()
    if "RidgeRegressor" in models:
        models = ["RidgeRegressor"] + [m for m in models if m != "RidgeRegressor"]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, colors):
        model_data = comparison[comparison["model"] == model]
        label = model.replace("Regressor", "")
        ax.plot(model_data["fold"], model_data[metric_col], 
               "o-", label=label, color=color, linewidth=2, markersize=8)
    
    ax.set_title(f"{split.upper()} {metric.upper()} Across Folds", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_report(
    results_dir: str = "data/experiments",
    output_dir: str = "data/experiments/plots",
    folds: Optional[List[int]] = None,
    horizons: Optional[List[str]] = None
):
    """
    Create comprehensive comparison report with all plots.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save plots
        folds: List of folds to include (None = all)
        horizons: List of horizons to include (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading results...")
    results_df = load_results(results_dir, folds=folds, horizons=horizons)
    
    if results_df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(results_df)} results")
    
    # Get available folds and horizons
    available_folds = sorted(results_df["fold"].unique())
    available_horizons = sorted(results_df["horizon"].unique())
    
    if folds is None:
        folds = available_folds
    if horizons is None:
        horizons = available_horizons
    
    # 1. Metrics comparison bar plots
    print("\nCreating metrics comparison plots...")
    for metric in ["rmse", "mae", "dir_acc"]:
        for split in ["train", "test"]:
            plot_metrics_comparison(
                results_df, metric=metric, split=split,
                save_path=os.path.join(output_dir, f"{split}_{metric}_comparison.png")
            )
    
    # 2. Heatmaps
    print("\nCreating heatmaps...")
    for metric in ["rmse", "mae", "dir_acc"]:
        for split in ["train", "test"]:
            plot_metrics_heatmap(
                results_df, metric=metric, split=split,
                save_path=os.path.join(output_dir, f"{split}_{metric}_heatmap.png")
            )
    
    # 3. Fold comparison
    print("\nCreating fold comparison plots...")
    for metric in ["rmse", "mae", "dir_acc"]:
        for split in ["train", "test"]:
            plot_fold_comparison(
                results_df, metric=metric, split=split,
                save_path=os.path.join(output_dir, f"{split}_{metric}_folds.png")
            )
    
    # 4. Predictions comparison (for first fold and horizon)
    print("\nCreating predictions comparison plots...")
    first_fold = folds[0] if folds else available_folds[0]
    first_horizon = horizons[0] if horizons else available_horizons[0]
    
    plot_predictions_comparison(
        results_dir, fold=first_fold, horizon=first_horizon,
        save_path=os.path.join(output_dir, f"predictions_scatter_fold{first_fold}_{first_horizon}.png")
    )
    
    plot_time_series_predictions(
        results_dir, fold=first_fold, horizon=first_horizon,
        save_path=os.path.join(output_dir, f"predictions_timeseries_fold{first_fold}_{first_horizon}.png")
    )
    
    print(f"\nAll plots saved to {output_dir}")


__all__ = [
    "load_results",
    "plot_metrics_comparison",
    "plot_predictions_comparison",
    "plot_time_series_predictions",
    "plot_metrics_heatmap",
    "plot_fold_comparison",
    "create_comparison_report",
]
