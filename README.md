# DSAI - Deep Learning for Time Series Forecasting

A comprehensive framework for time series forecasting using deep learning models (LSTM, Transformer, TCN) with Ridge regression as a baseline.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Models](#models)
- [Usage](#usage)
- [Results](#results)

## 🎯 Overview

This project implements a complete pipeline for time series forecasting with:
- **Baseline Model**: Ridge Regression
- **Deep Learning Models**: LSTM, Transformer, TCN
- **Features**: Sliding window sequences, left-padding, chronological validation
- **Evaluation**: Comprehensive metrics and visualization

## 📁 Project Structure

```
DSAI/
├── config/              # Configuration files
│   ├── experiments.py   # Hyperparameter grids
│   └── README.md
├── data/                # Data storage
│   ├── raw/            # Raw time series data
│   ├── processed/      # Processed features
│   ├── splits/         # Train/test splits
│   └── experiments/    # Experiment results
├── scripts/            # Execution scripts
│   └── training/       # Training scripts
│       ├── main.py
│       └── README.md
├── src/                # Source code
│   ├── models/         # Model implementations
│   │   ├── lstm.py
│   │   ├── transformers.py
│   │   ├── tcn.py
│   │   ├── ridge.py
│   │   ├── registry.py
│   │   └── README.md
│   ├── train/          # Training utilities
│   │   ├── runner.py
│   │   └── README.md
│   └── visualisation/  # Visualization tools
│       ├── plots.py
│       └── README.md
└── requirements.txt
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DSAI
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Experiments

Edit `config/experiments.py`:
- Set hyperparameter grids
- Choose folds and horizons (e.g., `FOLDS = [0]`, `HORIZONS = ["target_h1"]`)

### 3. Run Training

```bash
python scripts/training/main.py
```

This trains all models and saves results to `data/experiments/`.

### 4. Generate Visualizations

```python
from src.visualisation.plots import create_comparison_report

create_comparison_report(
    results_dir="data/experiments",
    output_dir="data/experiments/plots"
)
```

**For detailed step-by-step instructions, see [QUICKSTART.md](QUICKSTART.md)**

## 🏗️ Architecture

The pipeline follows this flow:

```
Config (experiments.py) 
  → Script (main.py) 
  → Registry (registry.py) 
  → Model (lstm.py, etc.) 
  → Execution (runner.py) 
  → Results (data/experiments/)
```

### Key Design Principles

- **Sliding Window Sequences**: Predict every timestep using fixed-length sequences
- **Left-Padding**: First sequences have full length via padding
- **Chronological Split**: No shuffling, preserves temporal order
- **Best Model Saving**: Saves model with best validation performance
- **Scikit-learn API**: Unified `.fit()` and `.predict()` interface

## 🤖 Models

### Ridge Regression (Baseline)
- Linear model with L2 regularization
- Flattens sequences to use all temporal information
- Fast training and inference

### LSTM
- Long Short-Term Memory networks
- Captures long-term dependencies
- Configurable layers and hidden units

### Transformer
- Self-attention mechanism
- Positional encoding
- Multi-head attention

### TCN (Temporal Convolutional Network)
- Dilated convolutions
- Residual connections
- Efficient temporal modeling

## 📊 Usage

### Training a Single Model

```python
from src.models import LSTMRegressor
from config.experiments import LSTM_GRID

# Get hyperparameters
params = {k: v[0] for k, v in LSTM_GRID.items()}
model = LSTMRegressor(**params)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Running Full Pipeline

```python
# In scripts/training/main.py
python scripts/training/main.py
```

### Comparing Models

```python
from src.visualisation.plots import (
    load_results,
    plot_metrics_comparison,
    plot_predictions_comparison
)

# Load results
results_df = load_results()

# Compare metrics
plot_metrics_comparison(results_df, metric="rmse", split="test")

# Compare predictions
plot_predictions_comparison(
    results_dir="data/experiments",
    fold=0,
    horizon="target_h1"
)
```

## 📈 Results

Results are saved in `data/experiments/`:
- **JSON files**: Metrics for each experiment
- **CSV files**: Predictions vs actuals
- **Summary CSVs**: Aggregated results per model

Visualization plots are saved in `data/experiments/plots/`:
- Metrics comparison charts
- Prediction scatter plots
- Time series overlays
- Heatmaps

## 🔧 Configuration

Edit `config/experiments.py` to:
- Adjust hyperparameter grids
- Select folds and horizons
- Configure model parameters

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
