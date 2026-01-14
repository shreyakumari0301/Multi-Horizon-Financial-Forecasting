# Config

Configuration files for experiments and hyperparameters.

## Files

### `experiments.py`

Contains hyperparameter grids for all models and experiment configuration.

#### Hyperparameter Grids

- **LSTM_GRID**: LSTM model hyperparameters
  - `seq_len`: Sequence length [32, 64]
  - `hidden`: Hidden units [128, 256]
  - `layers`: Number of layers [1, 2]
  - `dropout`: Dropout rate [0.0, 0.1]
  - `epochs`: Training epochs [15]
  - `batch_size`: Batch size [128]
  - `lr`: Learning rate [1e-3]
  - `val_frac`: Validation fraction [0.1]
  - `seed`: Random seed [0]

- **TRANSFORMER_GRID**: Transformer model hyperparameters
  - `seq_len`: Sequence length [32, 64]
  - `d_model`: Model dimension [128]
  - `nhead`: Number of attention heads [4]
  - `num_layers`: Number of encoder layers [2]
  - `dim_feedforward`: Feedforward dimension [256]
  - `dropout`: Dropout rate [0.1]
  - `epochs`: Training epochs [15]
  - `batch_size`: Batch size [128]
  - `lr`: Learning rate [1e-3]
  - `val_frac`: Validation fraction [0.1]
  - `seed`: Random seed [0]

- **TCN_GRID**: TCN model hyperparameters
  - `seq_len`: Sequence length [32, 64]
  - `channels`: Channel configuration [(64,64), (64,128)]
  - `kernel_size`: Convolution kernel size [3]
  - `dropout`: Dropout rate [0.0, 0.1]
  - `epochs`: Training epochs [15]
  - `batch_size`: Batch size [128]
  - `lr`: Learning rate [1e-3]
  - `val_frac`: Validation fraction [0.1]
  - `seed`: Random seed [0]

- **RIDGE_GRID**: Ridge regression hyperparameters
  - `seq_len`: Sequence length [32, 64]
  - `alpha`: Regularization strength [0.1, 1.0, 10.0]
  - `fit_intercept`: Fit intercept [True]
  - `val_frac`: Validation fraction [0.1]
  - `seed`: Random seed [0]

#### Experiment Configuration

- **FOLDS**: List of folds to train on (e.g., `[0]`)
- **HORIZONS**: List of target horizons (e.g., `["target_h1"]`)

## Usage

```python
import config.experiments as experiments

# Access hyperparameter grid
lstm_params = experiments.LSTM_GRID

# Get first combination
from src.models.wrapper import get_grid_params
params = get_grid_params(experiments.LSTM_GRID, index=0)

# Access experiment config
folds = experiments.FOLDS
horizons = experiments.HORIZONS
```

## Customization

To add new hyperparameters or modify existing ones:

1. Edit the corresponding `*_GRID` dictionary
2. Add new values to the list for grid search
3. Ensure model implementation accepts the new parameters

## Notes

- Grid values are lists to enable grid search
- The first value in each list is used by default
- All models share common parameters: `seq_len`, `val_frac`, `seed`
