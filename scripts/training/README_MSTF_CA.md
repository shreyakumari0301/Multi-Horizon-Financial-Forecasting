# Training MSTF-CA Model

## Quick Start

Train MSTF-CA on all three horizons (h1, h5, h20):

```bash
# Train on fold 0 (quick test)
python scripts/training/train_mstf_ca.py --folds 0

# Train on all 9 folds
python scripts/training/train_mstf_ca.py --folds all

# Train on specific folds
python scripts/training/train_mstf_ca.py --folds 0,1,2
```

## What It Does

1. **Loads data** for each fold and horizon from `data/splits/`
2. **Trains MSTF-CA** with horizon-specific configurations:
   - **h1**: 60 epochs, seq_len=128 (longer training for accuracy)
   - **h5**: 45 epochs, seq_len=128 (balanced)
   - **h20**: 40 epochs, seq_len=128 (faster)
3. **Evaluates** on test set and computes:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - DirAcc (Directional Accuracy)
4. **Saves**:
   - Trained models to `data/models/mstf_ca/fold_{N}/{horizon}.pkl`
   - Results to `data/experiments/mstf_ca/fold_{N}/{horizon}_results.json`
   - Predictions to `data/experiments/mstf_ca/fold_{N}/{horizon}_predictions.csv`
   - Summary to `data/experiments/mstf_ca_summary.csv`

## Output Example

```
======================================================================
MSTF-CA Training - All Horizons
======================================================================
Model: MSTF_CA
Folds: [0]
Horizons: ['target_h1', 'target_h5', 'target_h20']
...

=== Training MSTF_CA | Fold 0 | target_h1 ===
Loaded features: 10 technical + 28 news = 38 total
Training on 2000 samples, 38 features...
    Using horizon-specific config: {'epochs': 60, 'seq_len': 128}
...
Train Metrics:
  RMSE: 0.012345
  MAE:  0.008901
  DirAcc: 0.523

Test Metrics:
  RMSE: 0.013456
  MAE:  0.009234
  DirAcc: 0.512
```

## Results Structure

After training, you'll have:

```
data/
├── models/
│   └── mstf_ca/
│       └── fold_0/
│           ├── target_h1.pkl
│           ├── target_h5.pkl
│           └── target_h20.pkl
└── experiments/
    └── mstf_ca/
        ├── fold_0/
        │   ├── target_h1_results.json
        │   ├── target_h1_predictions.csv
        │   ├── target_h5_results.json
        │   ├── target_h5_predictions.csv
        │   ├── target_h20_results.json
        │   └── target_h20_predictions.csv
        └── mstf_ca_summary.csv
```

## Configuration

Model hyperparameters are in `config/experiments.py`:

- **MSTF_CA_GRID**: Base configuration
- **HORIZON_SPECIFIC_CONFIG**: Overrides for each horizon

You can modify these to tune the model for your specific use case.

## Notes

- The model uses **delta targets** by default (predicts changes, then reconstructs)
- **Direction loss** is enabled (weight=0.5) to improve directional accuracy
- Models are saved as pickle files (PyTorch state)
- Training uses early stopping based on validation loss
- All metrics are computed on reconstructed predictions (if delta targets used)
