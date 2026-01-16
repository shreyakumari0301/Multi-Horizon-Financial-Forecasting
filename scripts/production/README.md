# Production Predictor

Production-ready predictor for real-time forecasting using the full 38-feature set (10 technical + 28 news).

## Overview

The production predictor uses trained hybrid ensemble models to generate predictions with best-in-class performance targeting **68.7% Directional Accuracy**.

## Features

- **Automatic Feature Integration**: Handles both technical and news features automatically
- **Real-Time Processing**: Can process news headlines in real-time using FinBERT
- **Graceful Degradation**: Falls back to technical features only if news unavailable
- **Hybrid Ensemble**: Uses optimized weighted ensemble of all base models
- **Production Ready**: Handles scaling, sequences, and model loading

## Usage

### Basic Usage

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

### Batch Prediction

```python
# Predict for multiple dates
tech_df = pd.DataFrame(...)  # Technical features with date index
news_df = pd.DataFrame({"date": [...], "headline": [...]})  # News headlines

predictions = predictor.predict_batch(tech_df, news_df)
```

### Command Line

```bash
python scripts/production/production_predictor.py \
    --model_dir data/models \
    --fold 0 \
    --horizon target_h1 \
    --news_features data/processed/news_features_28d.csv
```

## Feature Set

The predictor uses **38 features total**:

### Technical Features (10)
- `ret_1`: One-day log return
- `ret_2`: 2-day cumulative return
- `ret_5`: 5-day cumulative return
- `vol_20`: 20-day realized volatility
- `ma_10`: 10-day moving average
- `ma_20`: 20-day moving average
- `ma_gap`: Price/MA gap
- `rsi_14`: RSI(14) indicator
- `vol_z`: Volume z-score
- `dow`: Day of week

### News Features (28)
- `news_pc1` through `news_pc28`: PCA-reduced FinBERT embeddings

## Performance Target

- **Directional Accuracy**: 68.7% (best-in-class)
- **Features**: 38 total (10 technical + 28 news)
- **Model**: Hybrid Ensemble (Ridge + LSTM + Transformer + TCN)

## Integration with Data Pipeline

If you've run the standard data pipeline with news data available, the features are automatically integrated:

1. **Process News**: `scripts/features/process_news_features.py`
2. **Integrate Features**: `scripts/features/integrate_news_features.py`
3. **Train Models**: `scripts/training/train_all_hybrid_models.py`
4. **Use Predictor**: `scripts/production/production_predictor.py`

The predictor automatically detects and uses news features if available, falling back gracefully if not.
