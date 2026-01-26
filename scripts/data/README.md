# Data Fetching Scripts

## Fetch Real News Headlines

The frontend was showing **demo/fake news**. Use this script to fetch **real news** from yfinance.

### Quick Start

```bash
# Fetch news for all default symbols (last 365 days)
python scripts/data/fetch_news.py

# If you get timeouts, try with fewer symbols first:
python scripts/data/fetch_news.py --symbols SPY AAPL MSFT

# Increase delay between requests (helps with rate limiting)
python scripts/data/fetch_news.py --delay 3.0

# More retries for unreliable connections
python scripts/data/fetch_news.py --max_retries 5

# Custom output path
python scripts/data/fetch_news.py --output data/raw/my_news.csv
```

### Troubleshooting Timeouts

If you get connection timeouts:

1. **Try fewer symbols first:**
   ```bash
   python scripts/data/fetch_news.py --symbols SPY AAPL
   ```

2. **Increase delay between requests:**
   ```bash
   python scripts/data/fetch_news.py --delay 3.0
   ```

3. **More retries:**
   ```bash
   python scripts/data/fetch_news.py --max_retries 5
   ```

4. **Check your internet connection** - yfinance requires stable connection

5. **Try again later** - Yahoo Finance may be temporarily unavailable

### What It Does

1. **Fetches real news** from yfinance for your symbols
2. **Saves to** `data/raw/news_headlines.csv` (format: date, headline)
3. **Removes duplicates** and sorts by date
4. **Ready for training** - can be processed with `process_news_features.py`

### After Fetching

Once you have real news:

```bash
# 1. Process news features (FinBERT embeddings + PCA)
python scripts/features/process_news_features.py --news_path data/raw/news_headlines.csv

# 2. Integrate into train/test splits
python scripts/features/integrate_news_features.py

# 3. Train models (will now use 38 features: 10 technical + 28 news)
python scripts/training/train_mstf_ca.py
```

### Frontend Integration

The web app (`web/rag_news.py`) will automatically:
- Load real news from `data/raw/news_headlines.csv` if it exists
- Fall back to fetching real-time news from yfinance if file is missing
- Only use demo news as last resort

### Default Symbols

If no symbols specified, fetches news for:
- `^GSPC`, `SPY` (US indices)
- `BTC-USD`, `ETH-USD` (crypto)
- `^NSEI`, `^NSEBANK` (India indices)
- `RELIANCE.NS`, `TCS.NS` (India stocks)
- `EURUSD=X`, `USDINR=X` (FX)
- `GC=F`, `CL=F` (commodities)
- `^VIX` (volatility)

### Notes

- **Free**: No API key needed (uses yfinance)
- **Rate limits**: yfinance may throttle if fetching too many symbols
- **Historical**: Fetches news from last N days (default: 365)
- **Format**: Outputs CSV with `date,headline` columns
