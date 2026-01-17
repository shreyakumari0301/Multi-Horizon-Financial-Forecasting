# config/experiments.py
# Optimized configuration for better directional accuracy

ESN_GRID = {
    "hidden_size":   [400, 800],
    "spectral_radius":[0.85, 0.95],
    "leak_rate":     [0.3, 1.0],
    "ridge_alpha":   [0.3, 3.0],
    "washout":       [100],
    "density":       [0.1],
    "state_clip":    [None],
    "seed":          [0],
}

LSTM_GRID = {
    "seq_len":        [64],          # Longer sequences for better context
    "hidden":         [256],         # Larger capacity
    "layers":         [2],           # Deeper network
    "dropout":        [0.0],         # No dropout (reduces over-smoothing)
    "epochs":         [40],          # More training for better accuracy
    "batch_size":     [128],         # Good batch size
    "lr":             [1e-3],        # Learning rate
    "weight_decay":  [1e-5],         # Light regularization
    "use_delta_target": [True],      # CRITICAL: Delta targets improve DirAcc
    "val_frac":       [0.1],
    "seed":           [0],
}

TRANSFORMER_GRID = {
    "seq_len":        [128],         # Longer lookback for patterns
    "d_model":        [256],         # Larger model dimension
    "nhead":          [8],           # More attention heads
    "num_layers":     [3],           # Deeper network
    "dim_feedforward":[512],         # Larger feedforward
    "dropout":        [0.1],         # Moderate dropout
    "epochs":         [40],          # More training
    "batch_size":     [128],         # Good batch size
    "lr":             [5e-4],        # Lower LR for stability
    "weight_decay":   [1e-5],        # Light regularization
    "direction_loss_weight": [0.5],  # CRITICAL: Direction loss improves DirAcc
    "val_frac":       [0.1],
    "seed":           [0],
}

TCN_GRID = {
    "seq_len":        [64],          # Good sequence length
    "channels":       [(128, 256)],   # Larger channels for capacity
    "kernel_size":    [3],
    "dropout":        [0.0],         # No dropout (reduces over-smoothing)
    "epochs":         [40],          # More training
    "batch_size":     [128],         # Good batch size
    "lr":             [1e-3],        # Learning rate
    "weight_decay":   [1e-5],        # Light regularization
    "use_delta_target": [True],      # CRITICAL: Delta targets improve DirAcc
    "val_frac":       [0.1],
    "seed":           [0],
}

RIDGE_GRID = {
    "seq_len":        [64],          # Longer sequences
    "alpha":          [1.0],        # Single good value (faster)
    "fit_intercept":  [True],
    "val_frac":       [0.1],
    "seed":           [0],
}

# Horizon-specific configurations (override defaults for speed/accuracy balance)
HORIZON_SPECIFIC_CONFIG = {
    "target_h1": {
        # For h1, we can train longer for better accuracy
        "lstm": {"epochs": 50},
        "transformer": {"epochs": 50},
        "tcn": {"epochs": 50},
    },
    "target_h5": {
        # For h5, balance speed and accuracy
        "lstm": {"epochs": 35, "batch_size": 256},
        "transformer": {"epochs": 35, "batch_size": 256},
        "tcn": {"epochs": 35, "batch_size": 256},
    },
    "target_h20": {
        # For h20, prioritize speed
        "lstm": {"epochs": 30, "batch_size": 256},
        "transformer": {"epochs": 30, "batch_size": 256},
        "tcn": {"epochs": 30, "batch_size": 256},
    },
}

# Which folds/horizons to train for default M4 run:
FOLDS     = [0]             # expand to [0,1,2,...] later
HORIZONS  = ["target_h1"]   # add "target_h5","target_h20" later