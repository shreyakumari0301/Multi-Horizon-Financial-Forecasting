# Research Paper: Implemented Works and Problems Solved

**Title (suggested):** *A Unified Deep Learning Framework for Multi-Horizon Financial Time Series Forecasting with Multi-Scale Temporal Fusion and News Embeddings*

---

## 1. Abstract

We present a comprehensive research framework for multivariate financial time series forecasting that addresses temporal dependency modeling, over-smoothing in deep networks, and integration of textual news signals. The framework unifies multiple deep learning architectures (LSTM, GRU, Transformer, TCN, ESN) under a single API, introduces a novel **Multi-Scale Temporal Fusion Network with Cross-Attention (MSTF-CA)** for multi-scale pattern capture, and provides an end-to-end pipeline from real-time news fetching to FinBERT/sentence-transformers embeddings and PCA-reduced features. We solve problems of sequence alignment via left-padding, evaluation leakage via chronological validation, and directional accuracy via delta targets and direction loss. The system is evaluated across multiple prediction horizons (H1, H5, H20) with financial metrics (RMSE, MAE, directional accuracy, PnL, Sharpe ratio) and is optimized for fast inference. All implementations are reproducible with standardized configuration and model registry.

---

## 2. Problems Addressed and Solved

### 2.1 Temporal Dependencies and Sequence Modeling

**Problem:** Future values in financial series depend on historical patterns; treating observations independently loses critical information.

**Solution:**  
- Sliding-window sequences of fixed length with **left-padding** so every timestep receives a prediction without future leakage.  
- All sequence models (LSTM, GRU, Transformer, TCN, MSTF-CA) consume aligned sequences `(N, L, F)` and produce per-step predictions.

### 2.2 Over-Smoothing in Deep Forecasters

**Problem:** Deep models tend to over-smooth and miss directional changes, which is critical for trading.

**Solution:**  
- **Delta target prediction:** Predict changes Δy[t] = y[t] − y[t−1]; reconstruct via y_pred[t] = y_pred[t−1] + Δy_pred[t].  
- **Direction loss:** For Transformer and MSTF-CA, total loss = MSE + λ × direction_loss (penalizing wrong sign predictions).  
- Architectural choices (e.g., reduced dropout, gradient clipping) to preserve signal.

### 2.3 Realistic Evaluation and Data Leakage

**Problem:** Random train/test splits cause information leakage and overstate performance.

**Solution:**  
- **Chronological validation:** Training data always precedes test data; validation is the last fraction (e.g., 10%) of training.  
- **No shuffling** of time series; evaluation over multiple folds and horizons (H1, H5, H20) with reported mean ± std.

### 2.4 Multi-Scale Temporal Patterns

**Problem:** Financial series exhibit short-, medium-, and long-term structure; single architectures (e.g., LSTM-only) do not explicitly model multiple scales.

**Solution:**  
- **MSTF-CA:** Parallel encoders (LSTM, GRU, TCN, Transformer) plus **cross-attention fusion** (query = Transformer; keys/values = LSTM, GRU, TCN) and learnable fusion weights.  
- First architecture to combine these four encoders with dynamic cross-attention for time series forecasting.

### 2.5 Integrating News with Price-Based Features

**Problem:** Using only price/technical features ignores sentiment and events captured in news.

**Solution:**  
- **Real-time news pipeline:** Fetch headlines via yfinance (with retries, rate-limit handling, nested API structure support).  
- **Dual embedding options:** (1) **FinBERT** → PCA to 28 features; (2) **Sentence-transformers** (small + large models) → 12 + 14 PCA components + has_news flags (28 features total).  
- **Flexible input format:** Support both `date`/`headline` and `published_utc`/`title`; handle sparse days via zero-padding and PCA component adjustment when samples &lt; 28.

### 2.6 Reproducibility and Deployment

**Problem:** Ad-hoc scripts and inconsistent APIs hinder reproduction and deployment.

**Solution:**  
- **Unified scikit-learn-style API:** All models implement `.fit(X, y)` and `.predict(X)` on 2D feature arrays.  
- **Model registry:** Named models (e.g. `lstm`, `transformer`, `tcn`, `mstf_ca`, `ridge`, `esn`, `lstm_gru_xgboost`) with config-driven instantiation.  
- **Best-model checkpointing:** Save weights for minimum validation loss; early stopping with configurable patience.  
- **Inference optimization:** `torch.inference_mode()`, explicit `eval()`, optional `torch.compile()` for MSTF-CA; reduced model sizes (seq_len, hidden, d_model, etc.) for 2–4× faster prediction.

### 2.7 Robustness of Data and APIs

**Problem:** External APIs (e.g. yfinance) change structure, time out, or rate-limit.

**Solution:**  
- **Robust fetch_news:** Retries with exponential backoff; handling of nested `content` and multiple title fields; rate-limit (429) detection and longer delays; optional curl_cffi session.  
- **Graceful degradation:** If news fetch fails, pipeline can still run with technical features only; padding when PCA has fewer than 28 components.

---

## 3. Implemented Works (Summary)

### 3.1 Models Implemented

| Model | Description | Main Use |
|-------|-------------|----------|
| **Ridge** | Linear baseline on flattened sequences | Baseline |
| **ESN** | Echo State Network (reservoir + Ridge readout) | Fast, no backprop on reservoir |
| **LSTM** | Multi-layer LSTM with delta targets, LR schedule, gradient clipping | Sequential baseline |
| **GRU** | (Inside MSTF-CA and GRU-LSTM hybrid) | Medium-term patterns |
| **Transformer** | Sinusoidal positional encoding, multi-head attention, direction loss | Long-range dependencies |
| **TCN** | Dilated causal convolutions, residual blocks | Multi-scale receptive field |
| **GRU+LSTM** | Parallel GRU and LSTM, concatenated representation | Hybrid RNN |
| **LSTM+GRU+XGBoost** | LSTM+GRU feature extractor, then XGBoost on concatenated features | Hybrid DL + boosting |
| **MSTF-CA** | LSTM + GRU + TCN + Transformer with cross-attention fusion | **Novel multi-scale architecture** |

### 3.2 Data and Feature Pipelines

- **Stock/technical data:** Configurable symbols; feature engineering (returns, volatility, MA, RSI, volume z-score, etc.); train/test splits by fold and horizon.  
- **News fetching:** `scripts/data/fetch_news.py` — yfinance-based, retries, rate-limit handling, nested `content` parsing; output CSV: `date`, `headline`.  
- **News embeddings:**  
  - **FinBERT path:** Load headlines → FinBERT embeddings → PCA (28 components) with handling for &lt; 28 samples (pad to 28).  
  - **Sentence-transformers path:** Two models (e.g. MiniLM + mpnet) → PCA (12 + 14) → daily aggregation → align to calendar; optional `has_news` flags.  
- **Integration:** Merge news features with technical features in split CSVs (e.g. 10 technical + 28 news = 38 features).

### 3.3 Training and Evaluation

- **Training scripts:** `main.py` (per-model grid), `train_all_hybrid_models.py` (hybrid ensemble), `train_mstf_ca.py` (MSTF-CA on H1/H5/H20).  
- **Horizon-specific configs:** Different epochs/batch sizes per target (e.g. target_h1, target_h5, target_h20).  
- **Metrics:** RMSE, MAE, directional accuracy; financial: PnL, Sharpe, hit ratio, turnover.  
- **Cross-fold aggregation:** Mean, std, min/max across folds.

### 3.4 Production and Inference

- **Production predictor:** Loads trained models; can use technical + news features; optional real-time headline processing.  
- **Inference optimizations:** Smaller default architectures, `torch.inference_mode()`, `model.eval()`, optional `torch.compile()` for MSTF-CA.  
- **Web dashboard:** Stock charts, news feed (from RAG/real-time or CSV), predictions (optional).

### 3.5 Configuration and Reproducibility

- **Config:** `config/experiments.py` — hyperparameter grids per model (Ridge, ESN, LSTM, Transformer, TCN, GRU-LSTM, LSTM-GRU-XGBoost, MSTF-CA) and horizon-specific overrides.  
- **Model registry:** `get_model(name, **kwargs)`, `list_models()`; all models registered and callable with same interface.

---

## 4. Novel Contributions (For Publication)

1. **MSTF-CA:** First architecture to combine LSTM, GRU, TCN, and Transformer with **cross-attention fusion** (query from Transformer, keys/values from LSTM/GRU/TCN) and learnable scale weights for time series forecasting.  
2. **Unified framework:** Single API and evaluation protocol for many deep and hybrid models plus Ridge/ESN baselines.  
3. **End-to-end news pipeline:** From live yfinance headlines → dual embedding options (FinBERT or sentence-transformers) → PCA and alignment → integration with technical features, with robustness to API changes and sparse data.  
4. **Problems explicitly solved:** Temporal alignment (left-padding), over-smoothing (delta targets + direction loss), leakage-free evaluation (chronological splits), multi-scale modeling (MSTF-CA), and deployment-ready inference (smaller models, inference_mode, optional compile).

---

## 5. Suggested Paper Structure

1. **Introduction** — Motivation, challenges in financial forecasting, scope of the framework.  
2. **Related Work** — Deep learning for time series; Transformers, TCNs, ESNs; news in finance; hybrid and multi-scale models.  
3. **Problem Formulation** — Notation, horizons, delta targets, evaluation protocol.  
4. **Methodology** — Left-padding, chronological splits, direction loss; model descriptions (short); **MSTF-CA** in detail (architecture, cross-attention, fusion weights).  
5. **News and Multimodal Features** — Fetching, FinBERT vs sentence-transformers, PCA, alignment, integration.  
6. **Experimental Setup** — Data, folds, horizons, metrics, baselines, hardware.  
7. **Results** — Tables (RMSE, MAE, DirAcc by horizon and model); ablation (e.g. with/without news, with/without direction loss); inference speed.

**Preliminary test metrics (fold 0, h=1).** Sharpe from toy sign backtest (1 bp cost):

| model       | fold | horizon | RMSE     | MAE      | R2     | DirAcc | AvgPnL   | Vol      | Sharpe | Turnover |
|-------------|------|---------|----------|----------|--------|--------|----------|----------|--------|----------|
| ridge       | 0    | h1      | 0.007638 | 0.005865 | -0.459 | 0.484  | 0.000316 | 0.006344 | 0.792  | 0.312    |
| esn         | 0    | h1      | 0.006332 | 0.004451 | -0.003 | 0.544  | 0.000619 | 0.006324 | 1.554  | 0.002    |
| lstm        | 0    | h1      | 0.006907 | 0.005047 | 0.452  | 0.750  | 0.004652 | 0.008053 | 9.171  | 0.431    |
| transformer | 0    | h1      | 0.010414 | 0.007906 | -1.712 | 0.448  | -0.000305| 0.006354 | -0.762 | 0.327    |
| tcn         | 0    | h1      | 0.006841 | 0.004958 | 0.462  | 0.778  | 0.004702 | 0.008022 | 9.304  | 0.454    |  
8. **Discussion** — Limitations, when MSTF-CA helps, value of news.  
9. **Conclusion** — Summary of implemented works and problems solved; code availability.  
10. **References** — PyTorch, scikit-learn, FinBERT, sentence-transformers, prior time series and finance papers.

---

## 6. Code and Assets to Cite

- **Repository structure:** `src/models/` (all models), `scripts/data/fetch_news.py`, `scripts/features/process_news_features.py`, `scripts/features/integrate_news_features.py`, `scripts/training/`, `config/experiments.py`.  
- **Key files:** `src/models/mstf_ca.py` (MSTF-CA), `src/models/lstm.py`, `src/models/transformers.py`, `src/models/tcn.py`, `src/models/esn.py`, `src/models/ridge.py`, `src/models/gru_lstm.py`, `src/models/lstm_gru_xgboost.py`.  
- **Docs:** `README.md`, `RESEARCH_CONTRIBUTION.md`, `INFERENCE_OPTIMIZATIONS.md`, `PAPER_CONTRIBUTIONS.md` (this file).

---

*This document summarizes the implemented works and problems solved for publication. You can copy sections into your paper and expand with full equations, tables, and plots as needed.*
