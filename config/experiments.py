# ESN_GRID = {
#     "hidden_size":   [400, 800],
#     "spectral_radius":[0.85, 0.95],
#     "leak_rate":     [0.3, 1.0],
#     "ridge_alpha":   [0.3, 3.0],
#     "washout":       [100],
#     "density":       [0.1],
#     "state_clip":    [None],
#     "seed":          [0],
# }

LSTM_GRID = {
    "seq_len":    [64],      # Longer sequences for better context
    "hidden":     [256],     # Larger capacity
    "layers":     [2],       # Deeper network
    "dropout":    [0.0],     # Less dropout (0.1->0.0) to reduce over-smoothing
    "epochs":     [50],      # More training
    "batch_size": [64],      # Smaller batches for better gradients
    "lr":         [1e-3],    # Higher learning rate (5e-4->1e-3)
    "weight_decay": [1e-5],  # L2 regularization
    "use_delta_target": [True],  # Use delta y target
    "val_frac":   [0.1],
    "seed":       [0],
}

TRANSFORMER_GRID = {
    "seq_len":        [128],     # Increased lookback
    "d_model":        [256],     # Larger model dimension
    "nhead":          [16],      # More attention heads (8->16)
    "num_layers":     [5],       # Deeper network (3->5)
    "dim_feedforward":[512],     # Larger feedforward
    "dropout":        [0.1],
    "epochs":         [50],      # More training
    "batch_size":     [64],      # Smaller batches
    "lr":             [5e-4],    # Lower learning rate
    "weight_decay":   [1e-5],    # L2 regularization
    "direction_loss_weight": [0.5],  # Increased weight for direction loss (0.3->0.5)
    "val_frac":       [0.1],
    "seed":           [0],
}

TCN_GRID = {
    "seq_len":    [64],          # Longer sequences
    "channels":   [(128, 256)],  # Larger channels
    "kernel_size":[3],
    "dropout":    [0.0],         # Less dropout (0.1->0.0) to reduce over-smoothing
    "epochs":     [50],          # More training
    "batch_size": [64],          # Smaller batches
    "lr":         [1e-3],        # Higher learning rate (5e-4->1e-3)
    "weight_decay": [1e-5],      # L2 regularization
    "use_delta_target": [True],  # Use delta y target
    "val_frac":   [0.1],
    "seed":       [0],
}

RIDGE_GRID = {
    "seq_len":      [32, 64],
    "alpha":        [0.1, 1.0, 10.0],
    "fit_intercept":[True],
    "val_frac":     [0.1],
    "seed":         [0],
}

# Horizon-specific configurations for faster training on longer horizons
# For target_h5 and target_h20, we use fewer epochs and larger batches
# This speeds up training significantly while maintaining performance
HORIZON_SPECIFIC_CONFIG = {
    "target_h1": {
        "lstm": {"epochs": 50, "batch_size": 64},
        "transformer": {"epochs": 50, "batch_size": 64},
        "tcn": {"epochs": 50, "batch_size": 64},
    },
    "target_h5": {
        "lstm": {"epochs": 30, "batch_size": 128},  # ~40% faster (30/50 epochs, 2x batch)
        "transformer": {"epochs": 30, "batch_size": 128},  # ~40% faster
        "tcn": {"epochs": 30, "batch_size": 128},  # ~40% faster
    },
    "target_h20": {
        "lstm": {"epochs": 25, "batch_size": 128},  # ~50% faster (25/50 epochs, 2x batch)
        "transformer": {"epochs": 25, "batch_size": 128},  # ~50% faster
        "tcn": {"epochs": 25, "batch_size": 128},  # ~50% faster
    },
}

# Which folds/horizons to train for default M4 run:
FOLDS     = [0]             # expand to [0,1,2,...] later
HORIZONS  = ["target_h1"]   # add "target_h5","target_h20" later