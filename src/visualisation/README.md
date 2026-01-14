# Visualisation

Visualization tools for comparing models and analyzing results.

## Overview

This module provides comprehensive plotting functions to compare Ridge baseline with other models (LSTM, Transformer, TCN).

## Files

### `plots.py`

All visualization functions for model comparison.

## Functions

### `load_results(results_dir, model_names, folds, horizons)`

Loads all experiment results into a DataFrame.

**Parameters:**
- `results_dir`: Directory containing experiment results (default: "data/experiments")
- `model_names`: List of model names to load (None = all)
- `folds`: List of folds to load (None = all)
- `horizons`: List of horizons to load (None = all)

**Returns:**
- DataFrame with all results

**Usage:**
```python
from src.visualisation.plots import load_results

results_df = load_results()
# Or filter specific models/folds
results_df = load_results(
    model_names=["LSTMRegressor", "RidgeRegressor"],
    folds=[0, 1],
    horizons=["target_h1"]
)
```

### `plot_metrics_comparison(results_df, metric, split, ...)`

Bar chart comparing metrics across models.

**Parameters:**
- `results_df`: DataFrame from `load_results()`
- `metric`: Metric to compare ("rmse", "mae", "dir_acc")
- `split`: "train" or "test"
- `save_path`: Path to save figure (None = show)
- `figsize`: Figure size tuple

**Usage:**
```python
from src.visualisation.plots import plot_metrics_comparison

plot_metrics_comparison(
    results_df,
    metric="rmse",
    split="test",
    save_path="comparison.png"
)
```

### `plot_predictions_comparison(results_dir, fold, horizon, ...)`

Scatter plots of predictions vs actuals for each model.

**Parameters:**
- `results_dir`: Directory containing experiment results
- `fold`: Fold number
- `horizon`: Target horizon
- `models`: List of model names (None = all)
- `save_path`: Path to save figure
- `figsize`: Figure size tuple

**Usage:**
```python
from src.visualisation.plots import plot_predictions_comparison

plot_predictions_comparison(
    results_dir="data/experiments",
    fold=0,
    horizon="target_h1",
    save_path="predictions_scatter.png"
)
```

### `plot_time_series_predictions(results_dir, fold, horizon, ...)`

Time series overlay of predictions vs actuals.

**Parameters:**
- `results_dir`: Directory containing experiment results
- `fold`: Fold number
- `horizon`: Target horizon
- `models`: List of model names (None = all, Ridge first)
- `n_samples`: Number of samples to plot (default: 100)
- `save_path`: Path to save figure
- `figsize`: Figure size tuple

**Usage:**
```python
from src.visualisation.plots import plot_time_series_predictions

plot_time_series_predictions(
    results_dir="data/experiments",
    fold=0,
    horizon="target_h1",
    n_samples=200,
    save_path="timeseries.png"
)
```

### `plot_metrics_heatmap(results_df, metric, split, ...)`

Heatmap showing metrics across models and horizons.

**Parameters:**
- `results_df`: DataFrame from `load_results()`
- `metric`: Metric to compare ("rmse", "mae", "dir_acc")
- `split`: "train" or "test"
- `save_path`: Path to save figure
- `figsize`: Figure size tuple

**Usage:**
```python
from src.visualisation.plots import plot_metrics_heatmap

plot_metrics_heatmap(
    results_df,
    metric="rmse",
    split="test",
    save_path="heatmap.png"
)
```

### `plot_fold_comparison(results_df, metric, split, ...)`

Line plot showing metrics across different folds.

**Parameters:**
- `results_df`: DataFrame from `load_results()`
- `metric`: Metric to compare ("rmse", "mae", "dir_acc")
- `split`: "train" or "test"
- `save_path`: Path to save figure
- `figsize`: Figure size tuple

**Usage:**
```python
from src.visualisation.plots import plot_fold_comparison

plot_fold_comparison(
    results_df,
    metric="rmse",
    split="test",
    save_path="folds_comparison.png"
)
```

### `create_comparison_report(results_dir, output_dir, ...)`

Generates all comparison plots and saves them.

**Parameters:**
- `results_dir`: Directory containing experiment results
- `output_dir`: Directory to save plots
- `folds`: List of folds to include (None = all)
- `horizons`: List of horizons to include (None = all)

**Usage:**
```python
from src.visualisation.plots import create_comparison_report

create_comparison_report(
    results_dir="data/experiments",
    output_dir="data/experiments/plots"
)
```

## Generated Plots

The `create_comparison_report()` function generates:

1. **Metrics Comparison**: Bar charts for RMSE, MAE, Directional Accuracy (train/test)
2. **Heatmaps**: Model vs Horizon heatmaps for each metric
3. **Fold Comparison**: Line plots showing performance across folds
4. **Predictions Scatter**: Scatter plots of predictions vs actuals
5. **Time Series**: Overlay of predictions over time

All plots are saved to the specified `output_dir`.

## Plot Features

- **Ridge Highlighting**: Ridge baseline is always shown first for easy comparison
- **Color Coding**: Consistent colors across plots for each model
- **Professional Styling**: Clean, publication-ready plots
- **High Resolution**: 300 DPI for publication quality

## Example Workflow

```python
from src.visualisation.plots import (
    load_results,
    plot_metrics_comparison,
    create_comparison_report
)

# Quick comparison
results_df = load_results()
plot_metrics_comparison(results_df, metric="rmse", split="test")

# Full report
create_comparison_report(
    results_dir="data/experiments",
    output_dir="data/experiments/plots"
)
```
