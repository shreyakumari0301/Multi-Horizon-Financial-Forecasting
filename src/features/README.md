# Features

Feature engineering modules for FinBERT embeddings and dimensionality reduction.

## Overview

This module implements FinBERT-based embeddings for financial news headlines, which capture sentiment and market events that technical indicators often miss. The embeddings are reduced to 28 key features using PCA and integrated with 10 technical features for a total of 38 features.

## Components

### FinBERT Embeddings (`finbert_embeddings.py`)

Uses FinBERT (ProsusAI/finbert), a BERT model pre-trained on financial communication, to generate sentence embeddings from news headlines.

**Key Features:**
- Pre-trained on financial text for better domain understanding
- Captures sentiment and market events
- Batch processing for efficiency
- Mean pooling of hidden states for sentence embeddings

**Usage:**
```python
from src.features.finbert_embeddings import FinBERTEmbedder, generate_news_embeddings

# Initialize embedder
embedder = FinBERTEmbedder(model_name="ProsusAI/finbert")

# Generate embeddings for headlines
headlines = ["Stock market surges on positive earnings", "Fed raises interest rates"]
embeddings = embedder.embed_sentences(headlines)

# Or process a DataFrame
embeddings_df = generate_news_embeddings(
    news_df,
    headline_col="headline",
    date_col="date"
)
```

### PCA Reduction (`pca_reduction.py`)

Reduces high-dimensional FinBERT embeddings (typically 768 dimensions) to 28 key features using Principal Component Analysis.

**Key Features:**
- Reduces from 768D to 28D while preserving variance
- Fits on training data only (no data leakage)
- StandardScaler normalization before PCA
- Saves/loads fitted reducers for consistency

**Usage:**
```python
from src.features.pca_reduction import PCAReducer, reduce_embeddings

# Reduce embeddings
reduced_df = reduce_embeddings(
    embeddings_df,
    n_components=28,
    train_dates=train_date_range
)
```

## Processing Pipeline

### Step 1: Process News Headlines

```bash
python scripts/features/process_news_features.py \
    --news_path data/raw/news_headlines.csv \
    --output_dir data/processed \
    --n_components 28
```

This generates:
- `data/processed/news_embeddings_raw.csv`: Full FinBERT embeddings
- `data/processed/news_features_28d.csv`: 28 PCA-reduced features
- `data/processed/news_pca_reducer.pkl`: Fitted PCA reducer

### Step 2: Integrate with Technical Features

```bash
python scripts/features/integrate_news_features.py \
    --news_features data/processed/news_features_28d.csv \
    --splits_dir data/splits
```

This updates all train/test splits to include:
- 10 technical features (z_ret_1, z_vol_20, z_rsi_14, etc.)
- 28 news features (z_news_pc1, z_news_pc2, ..., z_news_pc28)
- **Total: 38 features**

## News Data Format

Input CSV should have columns:
- `date`: Date of the headline
- `headline`: News headline text

Example:
```csv
date,headline
2020-01-01,Stock market opens higher on positive economic data
2020-01-02,Fed announces interest rate decision
...
```

## Impact on Performance

According to research:
- News embeddings provide sentiment/event signals not captured by technical indicators
- Significant improvement in directional accuracy
- Helps hybrid models achieve higher performance
- 28 PCA components capture most variance while reducing dimensionality

## Model Requirements

- **FinBERT Model**: Automatically downloads from HuggingFace (ProsusAI/finbert)
- **Fallback**: Uses bert-base-uncased if FinBERT unavailable
- **GPU**: Recommended for faster processing (automatically uses if available)

## Files

- `finbert_embeddings.py`: FinBERT embedding generation
- `pca_reduction.py`: PCA dimensionality reduction
- `__init__.py`: Package exports
- `process_news_features.py`: Script to process news data
- `integrate_news_features.py`: Script to integrate with technical features
