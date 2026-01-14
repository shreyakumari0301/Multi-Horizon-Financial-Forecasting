"""Aggregate and summarize metrics across all folds and models."""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


def load_all_results(
    results_dir: str = "data/experiments",
    model_names: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    horizons: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load all experiment results from JSON files across all folds.
    
    Args:
        results_dir: Directory containing experiment results
        model_names: List of model names to load (None = all)
        folds: List of folds to load (None = all available)
        horizons: List of horizons to load (None = all available)
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    
    if model_names is None:
        # Get all model directories
        model_names = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('.')]
    
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


def aggregate_metrics(
    results_df: pd.DataFrame,
    group_by: List[str] = ["model", "horizon"],
    metrics: List[str] = ["rmse", "mae", "dir_acc"]
) -> pd.DataFrame:
    """
    Aggregate metrics across folds.
    
    Args:
        results_df: DataFrame from load_all_results()
        group_by: Columns to group by (default: ["model", "horizon"])
        metrics: List of metrics to aggregate (default: ["rmse", "mae", "dir_acc"])
    
    Returns:
        DataFrame with aggregated statistics (mean, std, min, max)
    """
    split = "test"  # Focus on test metrics
    
    agg_data = []
    
    for group_key, group_df in results_df.groupby(group_by):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_by, group_key))
        else:
            group_dict = {group_by[0]: group_key}
        
        for metric in metrics:
            col = f"{split}_{metric}"
            if col not in group_df.columns:
                continue
            
            values = group_df[col].values
            group_dict[f"{metric}_mean"] = float(np.mean(values))
            group_dict[f"{metric}_std"] = float(np.std(values))
            group_dict[f"{metric}_min"] = float(np.min(values))
            group_dict[f"{metric}_max"] = float(np.max(values))
            group_dict[f"{metric}_median"] = float(np.median(values))
            group_dict["n_folds"] = len(values)
        
        agg_data.append(group_dict)
    
    return pd.DataFrame(agg_data)


def create_metrics_table(
    results_df: pd.DataFrame,
    metric: str = "dir_acc",
    split: str = "test",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a formatted metrics table for cross-model comparison.
    
    Args:
        results_df: DataFrame from load_all_results()
        metric: Metric to compare
        split: "train" or "test"
        save_path: Path to save table (None = return only)
    
    Returns:
        Formatted DataFrame
    """
    metric_col = f"{split}_{metric}"
    
    if metric_col not in results_df.columns:
        raise ValueError(f"Metric {metric_col} not found")
    
    # Pivot table: models vs horizons
    table = results_df.pivot_table(
        values=metric_col,
        index="model",
        columns="horizon",
        aggfunc=["mean", "std"]
    )
    
    # Flatten multi-level columns
    table.columns = [f"{horizon}_{stat}" for stat, horizon in table.columns]
    table = table.round(4)
    
    # Add summary row
    table.loc["MEAN"] = table.mean()
    
    if save_path:
        table.to_csv(save_path)
        print(f"Saved metrics table to {save_path}")
    
    return table


def aggregate_all_folds(
    results_dir: str = "data/experiments",
    output_dir: str = "data/experiments/aggregated",
    model_names: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    horizons: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate all results and create summary tables.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save aggregated results
        model_names: List of model names (None = all)
        folds: List of folds (None = all)
        horizons: List of horizons (None = all)
    
    Returns:
        Dictionary of aggregated DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading all results...")
    results_df = load_all_results(results_dir, model_names, folds, horizons)
    
    if results_df.empty:
        print("No results found!")
        return {}
    
    print(f"Loaded {len(results_df)} results")
    
    # Aggregate by model and horizon
    print("\nAggregating metrics...")
    aggregated = aggregate_metrics(results_df, group_by=["model", "horizon"])
    
    agg_path = os.path.join(output_dir, "aggregated_metrics.csv")
    aggregated.to_csv(agg_path, index=False)
    print(f"Saved aggregated metrics to {agg_path}")
    
    # Create metrics tables
    print("\nCreating metrics tables...")
    tables = {}
    for metric in ["rmse", "mae", "dir_acc"]:
        for split in ["train", "test"]:
            table = create_metrics_table(
                results_df,
                metric=metric,
                split=split,
                save_path=os.path.join(output_dir, f"{split}_{metric}_table.csv")
            )
            tables[f"{split}_{metric}"] = table
    
    # Summary statistics
    summary = {
        "total_experiments": len(results_df),
        "models": sorted(results_df["model"].unique().tolist()),
        "folds": sorted(results_df["fold"].unique().tolist()),
        "horizons": sorted(results_df["horizon"].unique().tolist()),
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"\nAggregation complete! Results saved to {output_dir}")
    
    return {
        "aggregated": aggregated,
        "tables": tables,
        "raw": results_df,
    }


__all__ = [
    "load_all_results",
    "aggregate_metrics",
    "create_metrics_table",
    "aggregate_all_folds",
]
