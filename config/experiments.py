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
    "seq_len":    [32, 64],
    "hidden":     [128, 256],
    "layers":     [1, 2],
    "dropout":    [0.0, 0.1],
    "epochs":     [15],      # increase later
    "batch_size": [128],
    "lr":         [1e-3],
    "val_frac":   [0.1],
    "seed":       [0],
}

TRANSFORMER_GRID = {
    "seq_len":        [32, 64],
    "d_model":        [128],
    "nhead":          [4],
    "num_layers":     [2],
    "dim_feedforward":[256],
    "dropout":        [0.1],
    "epochs":         [15],
    "batch_size":     [128],
    "lr":             [1e-3],
    "val_frac":       [0.1],
    "seed":           [0],
}

TCN_GRID = {
    "seq_len":    [32, 64],
    "channels":   [(64,64), (64,128)],
    "kernel_size":[3],
    "dropout":    [0.0, 0.1],
    "epochs":     [15],
    "batch_size": [128],
    "lr":         [1e-3],
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

# Which folds/horizons to train for default M4 run:
FOLDS     = [0]             # expand to [0,1,2,...] later
HORIZONS  = ["target_h1"]   # add "target_h5","target_h20" later