# Train

Training utilities and execution logic for model training and evaluation.

## Overview

This module handles the training pipeline, experiment execution, and results storage.

## Files

### `runner.py`

Core training and evaluation functions.

## Functions

### `load_fold_data(fold_dir, target_col)`

Loads training and test data for a specific fold.

**Parameters:**
- `fold_dir`: Directory containing `train.csv` and `test.csv`
- `target_col`: Target column name (e.g., "target_h1")

**Returns:**
- `X_train, y_train, X_test, y_test, test_index`: Data arrays and test index

**Usage:**
```python
from src.train.runner import load_fold_data

X_train, y_train, X_test, y_test, test_index = load_fold_data(
    fold_dir="data/splits/fold_0",
    target_col="target_h1"
)
```

### `compute_metrics(y_true, y_pred)`

Computes regression metrics.

**Parameters:**
- `y_true`: True values
- `y_pred`: Predicted values

**Returns:**
- Dictionary with `rmse`, `mae`, `dir_acc` (directional accuracy)

**Usage:**
```python
from src.train.runner import compute_metrics

metrics = compute_metrics(y_true, y_pred)
# {'rmse': 0.005, 'mae': 0.003, 'dir_acc': 0.55}
```

### `run_experiment(model, fold, horizon, ...)`

Runs a single experiment: trains model and evaluates on test set.

**Parameters:**
- `model`: Model instance with `.fit()` and `.predict()` methods
- `fold`: Fold number
- `horizon`: Target horizon (e.g., "target_h1")
- `splits_dir`: Directory containing fold data (default: "data/splits")
- `results_dir`: Directory to save results (default: "data/experiments")
- `save_predictions`: Whether to save predictions CSV (default: True)

**Returns:**
- Dictionary containing experiment results

**Usage:**
```python
from src.train.runner import run_experiment
from src.models import LSTMRegressor

model = LSTMRegressor(seq_len=32, hidden=128, ...)
results = run_experiment(
    model=model,
    fold=0,
    horizon="target_h1",
    results_dir="data/experiments"
)
```

### `run_grid_search(model_name, grid, folds, horizons, ...)`

Runs grid search across multiple folds and horizons.

**Parameters:**
- `model_name`: Name of the model ("lstm", "transformer", "tcn", "ridge")
- `grid`: Hyperparameter grid dictionary
- `folds`: List of fold numbers
- `horizons`: List of target horizons
- `splits_dir`: Directory containing fold data
- `results_dir`: Directory to save results
- `grid_index`: Index to use when extracting from grid (default: 0)
- `create_model_fn`: Function to create model (from main.py)

**Returns:**
- DataFrame with all experiment results

**Usage:**
```python
from src.train.runner import run_grid_search
import config.experiments as experiments

results_df = run_grid_search(
    model_name="lstm",
    grid=experiments.LSTM_GRID,
    folds=[0],
    horizons=["target_h1"],
    create_model_fn=create_model  # from main.py
)
```

## Results Structure

Results are saved in the following structure:

```
data/experiments/
├── LSTMRegressor/
│   └── fold_0/
│       ├── target_h1_results.json
│       └── target_h1_predictions.csv
├── TransformerRegressor/
├── TCNRegressor/
└── RidgeRegressor/
```

### JSON Results Format

```json
{
  "fold": 0,
  "horizon": "target_h1",
  "model_name": "LSTMRegressor",
  "train_metrics": {
    "rmse": 0.004,
    "mae": 0.003,
    "dir_acc": 0.52
  },
  "test_metrics": {
    "rmse": 0.005,
    "mae": 0.004,
    "dir_acc": 0.51
  },
  "n_train": 2520,
  "n_test": 252,
  "n_features": 10
}
```

### Predictions CSV Format

```csv
Date,y_true,y_pred
2016-03-22,0.001234,0.001456
2016-03-23,-0.000567,-0.000432
...
```

## Workflow

1. **Load Data**: `load_fold_data()` loads train/test splits
2. **Train Model**: Model's `.fit()` method trains on training data
3. **Evaluate**: `compute_metrics()` calculates performance metrics
4. **Save Results**: Results saved as JSON and predictions as CSV

## Integration

The runner is called from `scripts/training/main.py`:

```python
from src.train.runner import run_grid_search

# Main script creates models and calls runner
results_df = run_grid_search(...)
```
