# News Features Integration Guide

Complete guide for integrating FinBERT news embeddings to achieve **68.7% Directional Accuracy**.

## Overview

The system uses **38 features total**:
- **10 Technical Features**: Standard technical indicators
- **28 News Features**: PCA-reduced FinBERT embeddings from S&P 500 news headlines

## Complete Workflow

### Step 1: Prepare News Data

Create `data/raw/news_headlines.csv` with columns:
- `date`: Date of headline (YYYY-MM-DD)
- `headline`: News headline text

Example:
```csv
date,headline
2020-01-01,Stock market opens higher on positive economic data
2020-01-02,Fed announces interest rate decision
...
```

### Step 2: Process All Features (Unified Pipeline)

Run the unified processing script that handles everything:

```bash
python scripts/features/process_all_features.py \
    --news_path data/raw/news_headlines.csv \
    --splits_dir data/splits
```

**What it does:**
1. Generates FinBERT embeddings from headlines (768D)
2. Reduces to 28 features using PCA
3. Integrates with technical features in train/test splits
4. Creates 38-feature datasets automatically

**Output:**
- `data/processed/news_features_28d.csv`: 28 PCA-reduced features
- `data/processed/news_pca_reducer.pkl`: Fitted PCA reducer
- Updated `data/splits/fold_*/train.csv` and `test.csv`: With 38 features

### Step 3: Train Models

Train models as usual - they automatically detect and use news features:

```bash
python scripts/training/train_all_hybrid_models.py
```

The `load_fold_data()` function automatically:
- Detects `z_news_pc*` columns in train/test CSVs
- Loads both technical (10) and news (28) features
- Uses all 38 features for training

### Step 4: Use Production Predictor

For real-time predictions with the full 38-feature set:

```python
from scripts.production.production_predictor import ProductionPredictor

# Initialize predictor
predictor = ProductionPredictor(
    model_dir="data/models",
    fold=0,
    horizon="target_h1",
    news_features_path="data/processed/news_features_28d.csv"
)

# Single prediction
tech_features = np.array([...])  # 10 technical features
headlines = ["Stock market rises on positive earnings"]
prediction = predictor.predict(tech_features, headlines)

# Get trading signal
signal = predictor.get_signal(prediction)  # "LONG", "SHORT", or "NEUTRAL"
```

## Automatic Integration

### In Training Pipeline

The training pipeline (`scripts/training/main.py` and `train_all_hybrid_models.py`) automatically:
- Detects news features in train/test splits
- Uses all available features (10 technical + 28 news = 38 total)
- Falls back to technical only if news not available

### In Production

The production predictor (`scripts/production/production_predictor.py`):
- Automatically handles the full 38-feature set
- Processes news headlines in real-time using FinBERT
- Falls back gracefully if news unavailable
- Targets **68.7% Directional Accuracy**

## Feature Naming Convention

After integration and scaling:
- **Technical features**: `z_ret_1`, `z_vol_20`, `z_rsi_14`, etc. (10 total)
- **News features**: `z_news_pc1`, `z_news_pc2`, ..., `z_news_pc28` (28 total)
- **Total**: 38 features

## Performance Impact

According to research:
- News embeddings provide **sentiment/event signals** that technical indicators miss
- **Significant improvement** in model performance
- Hybrid model achieves **high directional accuracy** (68.7% target) with these features
- The 28 PCA components capture most variance while reducing dimensionality

## Verification

Check that features are integrated:

```python
import pandas as pd

# Check train data
train = pd.read_csv("data/splits/fold_0/train.csv", index_col=0)
tech_cols = [c for c in train.columns if c.startswith("z_") and not c.startswith("z_news_pc")]
news_cols = [c for c in train.columns if c.startswith("z_news_pc")]

print(f"Technical features: {len(tech_cols)}")
print(f"News features: {len(news_cols)}")
print(f"Total: {len(tech_cols) + len(news_cols)}")
```

Should show: 10 technical + 28 news = 38 total

## Without News Data

If news data is not available:
- System works with technical features only (10 features)
- Models still train and predict
- Lower directional accuracy expected
- Can add news features later and re-integrate

## Files Structure

```
data/
├── raw/
│   └── news_headlines.csv          # Input: date, headline columns
├── processed/
│   ├── news_embeddings_raw.csv      # Full FinBERT embeddings (768D)
│   ├── news_features_28d.csv        # PCA-reduced features (28D)
│   └── news_pca_reducer.pkl         # Fitted PCA reducer
└── splits/
    └── fold_*/
        ├── train.csv                # 38 features (z_* + z_news_pc*)
        ├── test.csv                 # 38 features
        └── scaler.json              # Metadata with n_technical, n_news
```

## Troubleshooting

**Q: Models still using 10 features?**
- Run `python scripts/features/integrate_news_features.py` to update splits
- Check that train/test CSVs have `z_news_pc*` columns

**Q: News features not found during training?**
- Verify `data/processed/news_features_28d.csv` exists
- Run integration script: `python scripts/features/integrate_news_features.py`

**Q: Production predictor errors?**
- Ensure trained models exist in `data/models/hybrid/`
- Check that `scaler.json` exists in fold directory with `n_news: 28`

## Best Practices

1. **Process news first**: Run `process_all_features.py` before training
2. **Verify integration**: Check that splits have 38 features
3. **Use production predictor**: For real-time predictions with full feature set
4. **Monitor performance**: Track directional accuracy to target 68.7%
