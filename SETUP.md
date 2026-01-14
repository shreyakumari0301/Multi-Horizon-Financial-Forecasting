# Setup Guide

This guide covers installation, project structure, and getting started with the DSAI time series forecasting framework.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DSAI
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import sklearn; print('Installation successful!')"
```

## Project Structure

```
DSAI/
├── config/                    # Configuration files
│   ├── experiments.py         # Hyperparameter grids for all models
│   └── README.md              # Configuration documentation
│
├── data/                      # Data storage
│   ├── raw/                   # Raw time series data
│   ├── processed/             # Processed features
│   ├── splits/                # Train/test splits (fold_0, fold_1, ...)
│   │   └── fold_{N}/         # Each fold contains train/test data
│   │       ├── X_train.csv
│   │       ├── y_train.csv
│   │       ├── X_test.csv
│   │       ├── y_test.csv
│   │       └── test_index.csv
│   ├── experiments/           # Experiment results
│   │   ├── {ModelName}/       # Results per model
│   │   │   └── fold_{N}/      # Results per fold
│   │   │       ├── {horizon}_results.json
│   │   │       └── {horizon}_predictions.csv
│   │   ├── aggregated/        # Aggregated metrics across folds
│   │   └── reports/           # Comprehensive evaluation reports
│   │       ├── tables/        # Metrics tables
│   │       ├── pnl/           # PnL charts
│   │       └── residuals/     # Residual analysis plots
│   └── models/                # Saved model files
│       ├── {model_name}/      # Individual models
│       └── hybrid/            # Hybrid ensemble models
│
├── scripts/                   # Execution scripts
│   ├── training/              # Training scripts
│   │   ├── main.py            # Main training pipeline
│   │   ├── train_all_hybrid_models.py  # Train hybrid models
│   │   └── README.md          # Training documentation
│   └── evaluation/            # Evaluation scripts
│       └── evaluate_all.py    # Comprehensive evaluation pipeline
│
├── src/                       # Source code
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── lstm.py           # LSTM model
│   │   ├── transformers.py   # Transformer model
│   │   ├── tcn.py            # TCN model
│   │   ├── ridge.py          # Ridge baseline
│   │   ├── registry.py       # Model registry
│   │   └── README.md         # Models documentation
│   ├── train/                 # Training utilities
│   │   ├── runner.py         # Experiment runner
│   │   └── README.md         # Training documentation
│   ├── eval/                  # Evaluation utilities
│   │   ├── aggregate.py      # Metrics aggregation
│   │   └── README.md         # Evaluation documentation
│   └── visualisation/         # Visualization tools
│       ├── plots.py          # Plotting functions
│       └── README.md         # Visualization documentation
│
├── requirements.txt           # Python dependencies
├── README.md                  # Technical documentation
└── SETUP.md                   # This file
```

## Quick Start

### Step 1: Configure Experiments

Edit `config/experiments.py` to set your hyperparameters and training configuration:

```python
# Example: Train on fold 0, target_h1
FOLDS = [0]
HORIZONS = ["target_h1"]

# Adjust hyperparameters as needed
LSTM_GRID = {
    "seq_len": [64],
    "hidden": [256],
    "layers": [2],
    # ... more parameters
}
```

### Step 2: Run Training

Train all models (Ridge, LSTM, Transformer, TCN):

```bash
python scripts/training/main.py
```

This will:
- Train each model on the specified folds and horizons
- Save results to `data/experiments/`
- Generate prediction CSVs and metrics JSONs

### Step 3: Run Evaluation

Generate comprehensive evaluation reports:

```bash
python scripts/evaluation/evaluate_all.py
```

This will:
- Aggregate metrics across all folds
- Create metrics tables
- Generate PnL charts
- Create residual analysis plots

Results will be saved to:
- `data/experiments/aggregated/` - Aggregated metrics
- `data/experiments/reports/` - Comprehensive reports

### Step 4: View Results

Check the generated reports:

```bash
# View aggregated metrics
cat data/experiments/aggregated/aggregated_metrics.csv

# View metrics tables
ls data/experiments/reports/tables/

# View PnL charts
ls data/experiments/reports/pnl/

# View residual plots
ls data/experiments/reports/residuals/
```

## Configuration

### Hyperparameter Grids

All hyperparameter grids are defined in `config/experiments.py`:

- `LSTM_GRID`: LSTM hyperparameters
- `TRANSFORMER_GRID`: Transformer hyperparameters
- `TCN_GRID`: TCN hyperparameters
- `RIDGE_GRID`: Ridge hyperparameters

### Training Configuration

```python
# In config/experiments.py
FOLDS = [0, 1, 2]  # Which folds to train on
HORIZONS = ["target_h1", "target_h5", "target_h20"]  # Which horizons
```

### Model-Specific Settings

Each model supports various hyperparameters. See:
- `src/models/README.md` for model-specific documentation
- `config/experiments.py` for available parameters

## Training Multiple Folds

For comprehensive evaluation, train on multiple folds:

```python
# In config/experiments.py
FOLDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # All 9 folds
HORIZONS = ["target_h1", "target_h5", "target_h20"]  # All horizons
```

Then run:
```bash
python scripts/training/main.py
```

## Training Hybrid Models

To train hybrid ensemble models on all folds and horizons:

```bash
python scripts/training/train_all_hybrid_models.py
```

This trains:
- All base models (Ridge, LSTM, Transformer, TCN)
- Hybrid ensemble with weighted voting
- On all 9 folds and 3 horizons (27 total combinations)

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root and virtual environment is activated
2. **CUDA Errors**: Models will fall back to CPU if CUDA is unavailable
3. **Memory Issues**: Reduce `batch_size` in hyperparameter grids
4. **Missing Data**: Ensure `data/splits/` contains your fold data

### Getting Help

- Check individual module READMEs in `src/` subdirectories
- Review `config/experiments.py` for configuration options
- See technical documentation in `README.md`

## Next Steps

After setup:
1. Review the technical documentation in `README.md`
2. Explore model implementations in `src/models/`
3. Customize hyperparameters in `config/experiments.py`
4. Run experiments and analyze results
