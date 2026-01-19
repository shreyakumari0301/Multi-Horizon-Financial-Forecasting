# Setup Guide

This guide covers model architecture, installation, project structure, and getting started with the DSAI time series forecasting framework.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)

## Model Architecture

This framework implements five distinct models, each chosen for specific strengths in time series forecasting:

### 1. Ridge Regression (Baseline)

**Why Ridge Regression?**
- **Linear Baseline**: Provides a simple, interpretable baseline to assess whether deep learning adds value
- **Fast Training**: Trains in seconds, enabling quick experimentation and hyperparameter tuning
- **Surprisingly Competitive**: Often achieves competitive directional accuracy (0.516 on H1), demonstrating that the signal has a significant linear component
- **Overfitting Control**: L2 regularization prevents overfitting on limited time series data
- **Ensemble Component**: Serves as a stable, linear voter in the hybrid ensemble

**Implementation Details:**
- Flattens sequences to use all temporal information as features
- L2 regularization (alpha: 0.1-10.0) for overfitting prevention
- Fast inference suitable for real-time predictions

### 2. Echo State Network (ESN)

**Why ESN?**
- **Reservoir Computing**: Randomly initialized reservoir captures rich temporal dynamics without backpropagation
- **Fast Training**: Only the output layer (Ridge regression) is trained, making ESN extremely fast to train
- **Memory Capacity**: Large reservoir (400-800 units) provides high memory capacity for temporal patterns
- **Echo State Property**: Controlled spectral radius (0.85-0.95) ensures bounded, stable states
- **Complementary Approach**: Different architecture from gradient-based models, providing diversity in ensemble

**Implementation Details:**
- Sparse reservoir connectivity (density 0.1) for efficiency
- Leaky integrator (leak_rate 0.3-1.0) for temporal memory
- Washout period (100 timesteps) to discard transient states
- Ridge regression on reservoir states for output mapping

### 3. LSTM (Long Short-Term Memory)

**Why LSTM?**
- **Temporal Dependencies**: Explicitly designed to capture long-term dependencies in sequences
- **Gradient Flow**: Gating mechanisms (forget, input, output gates) prevent vanishing gradients
- **Sequence Modeling**: Natural fit for time series where past observations influence future predictions
- **Proven Track Record**: Widely used and well-understood architecture for time series forecasting
- **Delta Target Strategy**: Benefits significantly from predicting changes (Δy) rather than absolute values, reducing over-smoothing

**Implementation Details:**
- Multi-layer architecture (2 layers) with hidden size 256
- Delta target prediction to focus on momentum rather than absolute levels
- Reduced dropout (0.0) to prevent over-regularization
- Learning rate scheduling and gradient clipping for stable training

### 4. Transformer

**Why Transformer?**
- **Attention Mechanism**: Self-attention captures complex temporal relationships across the entire sequence
- **Best Prediction Accuracy**: Achieves lowest RMSE across all horizons, demonstrating superior pattern recognition
- **Multi-Head Attention**: 8 heads allow the model to attend to different temporal scales simultaneously
- **Direction Loss**: Custom loss component explicitly encourages correct sign prediction, critical for financial applications
- **Positional Encoding**: Sinusoidal positional encoding provides explicit temporal structure

**Implementation Details:**
- Deeper architecture (3 layers) with d_model=256 for complex pattern recognition
- Direction loss weight (0.5) balances prediction accuracy and directional correctness
- Lower learning rate (5e-4) for stable training of deeper networks
- Best individual model performance on H5 horizon (0.564 DirAcc)

### 5. TCN (Temporal Convolutional Network)

**Why TCN?**
- **Dilated Convolutions**: Exponential dilation rates (2^0, 2^1, 2^2, ...) exponentially increase receptive field without adding parameters
- **Causal Structure**: Ensures no future information leakage, critical for time series
- **Efficient Computation**: Convolutions are more efficient than RNNs for parallel processing
- **Residual Connections**: Facilitate gradient flow in deep networks
- **Delta Target Strategy**: Like LSTM, benefits from predicting changes to reduce smoothing

**Implementation Details:**
- Channel progression (128 → 256) for hierarchical feature extraction
- Kernel size 3 for local pattern recognition
- Causal padding ensures temporal order
- Higher learning rate (1e-3) for faster convergence

### Hybrid Ensemble

**Why Hybrid Ensemble?**
- **Model Diversity**: Combines five different architectures (linear, reservoir, RNN, attention, CNN) for robust predictions
- **Complementary Strengths**: Each model captures different aspects of temporal patterns
- **Validation-Based Weights**: Uses validation set to optimize ensemble weights, not fixed equal weights
- **Robustness**: Ensemble predictions are more stable across different market conditions
- **Performance**: Balances accuracy and directional correctness better than individual models

**Implementation Details:**
- Weight optimization using scipy.optimize.minimize on validation directional accuracy
- Fallback to performance-based weights if optimization fails
- Trained on all 9 folds and 3 horizons (27 total combinations)
- Saves ensemble models for production deployment

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
# Install core dependencies
pip install -r requirements.txt

# For web dashboard (optional)
cd web
pip install -r requirements.txt
cd ..
```

### 4. Verify Installation

```bash
# Verify core packages
python -c "import torch; import sklearn; import transformers; print('Core packages installed successfully!')"

# Verify web packages (if installed)
python -c "import flask; import yfinance; print('Web packages installed successfully!')"
```

### 5. Download Pre-trained Models (Optional)

If using FinBERT for news features:

```bash
# FinBERT will download automatically on first use
# Or pre-download:
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('ProsusAI/finbert'); AutoTokenizer.from_pretrained('ProsusAI/finbert')"
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

## Environment Configuration

### Create `.env` File

Create a `.env` file in the project root for configuration:

```bash
# yfinance cache duration (seconds)
YFINANCE_CACHE_DURATION=60

# Model directory path
MODEL_DIR=data/models

# Optional: News API configuration (if using external news API)
# NEWS_API_KEY=your_api_key_here
```

### Web Dashboard Configuration

For the web dashboard, ensure environment variables are set:

```bash
# Copy example file
cp .env.example .env

# Edit as needed
```

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Make sure you're in the project root and virtual environment is activated
   - Check that all dependencies are installed: `pip install -r requirements.txt`

2. **CUDA Errors**: 
   - Models will fall back to CPU if CUDA is unavailable
   - Install PyTorch with CUDA support if you have a GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **Memory Issues**: 
   - Reduce `batch_size` in hyperparameter grids
   - Use smaller models (reduce `hidden_size`, `d_model`, etc.)
   - Process data in smaller chunks

4. **Missing Data**: 
   - Ensure `data/splits/` contains your fold data
   - Run feature processing scripts if needed: `python scripts/features/process_all_features.py`

5. **yfinance Errors**:
   - Check internet connection
   - Verify symbol is valid (e.g., "AAPL" not "APPL")
   - Increase cache duration if rate limited

6. **FinBERT Download Issues**:
   - Ensure internet connection for first-time download
   - Check disk space (model is ~400MB)
   - Use VPN if Hugging Face is blocked in your region

### Getting Help

- Check individual module READMEs in `src/` subdirectories
- Review `config/experiments.py` for configuration options
- See technical documentation in `README.md`
- Check `web/README.md` for web dashboard setup

## Next Steps

After setup:

1. **Review Documentation**:
   - Technical overview: `README.md`
   - Model implementations: `src/models/README.md`
   - Training guide: `scripts/training/README.md`

2. **Explore Code**:
   - Model implementations: `src/models/`
   - Training pipeline: `scripts/training/main.py`
   - Evaluation tools: `src/eval/`

3. **Configure Experiments**:
   - Edit hyperparameters: `config/experiments.py`
   - Set folds and horizons: `FOLDS` and `HORIZONS` in `experiments.py`

4. **Run Experiments**:
   - Train models: `python scripts/training/main.py`
   - Train hybrid ensemble: `python scripts/training/train_all_hybrid_models.py`
   - Evaluate results: `python scripts/evaluation/evaluate_all.py`

5. **Launch Web Dashboard** (Optional):
   - Start server: `python web/run.py`
   - Open browser: http://localhost:5000
   - View real-time predictions and news
