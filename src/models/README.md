# Models

Model implementations for time series forecasting.

## Overview

This module contains all model implementations with a unified scikit-learn style API (`.fit()` and `.predict()`).

## Models

### Ridge Regression (`ridge.py`)
- **Class**: `RidgeRegressor`
- **Type**: Linear baseline model
- **Features**: Flattens sequences to use all temporal information
- **Use Case**: Fast baseline for comparison

### LSTM (`lstm.py`)
- **Class**: `LSTMRegressor`
- **Type**: Recurrent neural network
- **Features**: Captures long-term temporal dependencies
- **Use Case**: Standard sequence modeling

### Transformer (`transformers.py`)
- **Class**: `TransformerRegressor`
- **Type**: Attention-based model
- **Features**: Self-attention mechanism, positional encoding
- **Use Case**: Complex temporal patterns

### TCN (`tcn.py`)
- **Class**: `TCNRegressor`
- **Type**: Temporal Convolutional Network
- **Features**: Dilated convolutions, residual connections
- **Use Case**: Efficient temporal modeling

## Architecture

All models follow the same design pattern:

1. **`_SeqMaker`**: Creates sliding windows with left-padding
2. **Model Head**: PyTorch module (for DL models) or scikit-learn model (for Ridge)
3. **Regressor Class**: Wrapper with `.fit()` and `.predict()` methods

### Key Features

- **Left-padding**: First sequences have full length
- **Sliding windows**: Every timestep can be predicted
- **Chronological validation**: No shuffling, preserves temporal order
- **Best model saving**: Saves model with best validation performance

## Registry System

Models are registered via the `@register_model()` decorator:

```python
from .registry import register_model

@register_model("lstm")
class LSTMRegressor:
    ...
```

Access models through the registry:

```python
from src.models.registry import get_model

model = get_model("lstm", seq_len=32, hidden=128, ...)
```

**Note**: The wrapper functionality (`get_grid_params`, `create_model`) is located in `scripts/training/main.py`, not in this module.

## Usage

### Direct Instantiation

```python
from src.models import LSTMRegressor

model = LSTMRegressor(
    seq_len=32,
    hidden=128,
    layers=1,
    dropout=0.0,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    val_frac=0.1,
    seed=0
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Via Registry

```python
from src.models.registry import get_model

model = get_model("lstm", seq_len=32, hidden=128, ...)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Parameters

### Common Parameters
- `seq_len`: Sequence length for sliding windows
- `val_frac`: Fraction of data for validation
- `seed`: Random seed for reproducibility

### LSTM-Specific
- `hidden`: Number of hidden units
- `layers`: Number of LSTM layers
- `dropout`: Dropout rate

### Transformer-Specific
- `d_model`: Model dimension
- `nhead`: Number of attention heads
- `num_layers`: Number of encoder layers
- `dim_feedforward`: Feedforward network dimension

### TCN-Specific
- `channels`: Tuple of channel sizes per layer
- `kernel_size`: Convolution kernel size

### Ridge-Specific
- `alpha`: Regularization strength
- `fit_intercept`: Whether to fit intercept

## Input/Output Format

- **Input X**: `(N, F)` numpy array where N = samples, F = features
- **Input y**: `(N,)` or `(N, O)` numpy array where O = output dimensions
- **Output**: `(N,)` or `(N, O)` numpy array of predictions

## Files

- `registry.py`: Model registration system
- `lstm.py`: LSTM implementation
- `transformers.py`: Transformer implementation
- `tcn.py`: TCN implementation
- `ridge.py`: Ridge regression implementation
- `__init__.py`: Package exports
