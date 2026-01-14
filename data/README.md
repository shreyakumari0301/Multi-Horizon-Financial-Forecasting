# Data

Data storage directory for raw data, processed features, splits, and experiment results.

## Directory Structure

```
data/
├── raw/              # Raw time series data
├── processed/        # Processed features
├── splits/           # Train/test splits
└── experiments/      # Experiment results
```

## Raw Data (`raw/`)

Contains original time series data files (CSV format).

**Format:**
- Date-indexed time series
- OHLCV (Open, High, Low, Close, Volume) data
- Multiple symbols/assets

**Example Files:**
- `GSPC_*.csv`: S&P 500 data
- `SPY_*.csv`: SPY ETF data
- `BTC-USD_*.csv`: Bitcoin data
- etc.

## Processed Data (`processed/`)

Contains processed features ready for modeling.

**Format:**
- Date-indexed DataFrames
- Feature columns (returns, volatility, technical indicators)
- Target columns (target_h1, target_h5, target_h20)

**Example Files:**
- `GSPC_features.csv`
- `SPY_features.csv`

## Splits (`splits/`)

Contains train/test splits for time series cross-validation.

### Structure

```
splits/
├── splits.json           # Split configuration
└── fold_0/
    ├── train.csv        # Training data
    ├── test.csv         # Test data
    └── scaler.json      # Scaler metadata
```

### Data Format

**train.csv / test.csv:**
- Feature columns: `z_*` (scaled features)
- Target columns: `target_h1`, `target_h5`, `target_h20`
- Date index

**scaler.json:**
- Feature names
- Mean and scale values
- Train/test date ranges

### splits.json

Configuration file defining:
- Number of folds
- Train/test periods
- Date ranges for each fold

## Experiments (`experiments/`)

Contains results from model training and evaluation.

### Structure

```
experiments/
├── LSTMRegressor/
│   └── fold_0/
│       ├── target_h1_results.json
│       └── target_h1_predictions.csv
├── TransformerRegressor/
├── TCNRegressor/
├── RidgeRegressor/
├── lstm_summary.csv
└── plots/              # Generated visualizations
```

### Results Format

**`*_results.json`:**
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

**`*_predictions.csv`:**
```csv
Date,y_true,y_pred
2016-03-22,0.001234,0.001456
2016-03-23,-0.000567,-0.000432
...
```

**`*_summary.csv`:**
- Aggregated results across all folds/horizons
- One row per experiment

### Plots (`experiments/plots/`)

Generated visualization plots:
- Metrics comparison charts
- Prediction scatter plots
- Time series overlays
- Heatmaps
- Fold comparison plots

## Data Flow

1. **Raw Data** → Processed in notebooks/scripts
2. **Processed Data** → Split into train/test folds
3. **Splits** → Used for model training
4. **Training** → Generates results in experiments/
5. **Results** → Visualized in plots/

## Notes

- All CSV files use date index
- Feature columns are prefixed with `z_` (scaled)
- Target columns are `target_h1`, `target_h5`, `target_h20`
- Results are organized by model, fold, and horizon
