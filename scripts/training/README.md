# Training Scripts

Main execution scripts for running the training pipeline.

## Files

### `main.py`

Main entry point for the training pipeline.

## Overview

The main script orchestrates the entire training pipeline:

1. **Loads Configuration**: Reads hyperparameter grids from `config/experiments.py`
2. **Creates Models**: Instantiates models using the registry system
3. **Runs Experiments**: Executes training via `runner.py`
4. **Saves Results**: Stores results in `data/experiments/`

## Usage

### Basic Execution

```bash
python scripts/training/main.py
```

### Configuration

Edit `config/experiments.py` to:
- Modify hyperparameter grids
- Select models to train
- Choose folds and horizons

### Customization

Edit `main.py` to:
- Select specific models: `MODELS = ["lstm", "ridge"]`
- Override hyperparameters
- Add custom logic

## Functions

### `get_grid_params(grid, index=0)`

Extracts hyperparameters from a grid by taking the value at specified index.

**Parameters:**
- `grid`: Dictionary with lists of hyperparameter values
- `index`: Which index to take from each list (default: 0)

**Returns:**
- Dictionary of hyperparameters with single values

### `create_model(model_name, grid, grid_index=0, **override_kwargs)`

Creates a model instance from hyperparameter grid.

**Parameters:**
- `model_name`: Name of the model ("lstm", "transformer", "tcn", "ridge")
- `grid`: Hyperparameter grid dictionary
- `grid_index`: Index to use when extracting from grid (default: 0)
- `**override_kwargs`: Additional parameters to override

**Returns:**
- Model instance

### `main()`

Main training pipeline function.

**Configuration:**
- `MODELS`: List of models to train
- `FOLDS`: List of folds (from experiments.py)
- `HORIZONS`: List of horizons (from experiments.py)
- `SPLITS_DIR`: Directory containing fold data
- `RESULTS_DIR`: Directory to save results

## Workflow

```
main.py
  ↓
Read experiments.py
  ↓
For each model:
  ↓
  Get hyperparameter grid
  ↓
  Create model instance
  ↓
  Call run_grid_search()
  ↓
  Save results
```

## Output

The script prints:
- Configuration summary
- Training progress for each model
- Metrics after each experiment
- Summary statistics

Results are saved to `data/experiments/`:
- JSON files with metrics
- CSV files with predictions
- Summary CSVs per model

## Example Output

```
============================================================
Training Pipeline
============================================================
Models: ['lstm', 'transformer', 'tcn', 'ridge']
Folds: [0]
Horizons: ['target_h1']
Splits: data/splits
Results: data/experiments
============================================================

============================================================
Training LSTM
============================================================

=== LSTM | Fold 0 | target_h1 ===
Test RMSE: 0.005234 | MAE: 0.003456 | DirAcc: 0.521

Saved summary → data/experiments/lstm_summary.csv

LSTM completed. Results shape: (1, 10)
```

## Integration

The script integrates:
- **Config**: `config/experiments.py` for hyperparameters
- **Registry**: `src/models/registry.py` for model creation
- **Runner**: `src/train/runner.py` for execution
- **Models**: `src/models/*.py` for model implementations
