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

LSTM_GRU_XGBOOST_GRID = {
    "seq_len":        [64],          # Longer sequences for better context
    "lstm_hidden":    [128],         # LSTM hidden size
    "gru_hidden":     [128],         # GRU hidden size
    "lstm_layers":    [2],           # LSTM depth
    "gru_layers":     [2],           # GRU depth
    "dropout":        [0.0],         # No dropout (reduces over-smoothing)
    "xgb_n_estimators": [200],       # XGBoost trees
    "xgb_max_depth":  [6],           # XGBoost depth
    "xgb_learning_rate": [0.05],     # XGBoost learning rate
    "xgb_subsample":  [0.8],          # XGBoost subsample
    "xgb_colsample_bytree": [0.8],   # XGBoost feature sampling
    "use_delta_target": [True],      # CRITICAL: Delta targets improve DirAcc
    "val_frac":       [0.1],
    "seed":           [0],
}

# Novel Multi-Scale Temporal Fusion Network with Cross-Attention (MSTF-CA)
# Research contribution: Combines LSTM, GRU, TCN, Transformer with cross-attention fusion
MSTF_CA_GRID = {
    "seq_len":              [128],         # Longer sequences for multi-scale patterns
    "hidden":               [128],         # LSTM/GRU hidden size
    "d_model":              [256],         # Transformer dimension
    "nhead":                [8],           # Attention heads
    "num_transformer_layers": [3],        # Transformer depth
    "tcn_channels":         [(64, 128)],   # TCN channel progression
    "dropout":              [0.1],         # Moderate dropout
    "epochs":               [50],          # More training for complex model
    "batch_size":           [128],         # Good batch size
    "lr":                   [5e-4],        # Lower LR for stability
    "weight_decay":         [1e-5],        # Light regularization
    "use_delta_target":     [True],        # CRITICAL: Delta targets improve DirAcc
    "direction_loss_weight": [0.5],        # Direction loss for trading
    "val_frac":             [0.1],
    "seed":                 [0],
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