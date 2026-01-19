# Real-Time Data Integration with yfinance

Complete guide for using real-time stock data from Yahoo Finance via yfinance API.

## Overview

The application now uses **yfinance** to fetch real-time stock data directly from Yahoo Finance, eliminating the need for local CSV files and providing up-to-date market information.

## Features

- **Real-time Data**: Live stock prices and historical data
- **No API Key Required**: yfinance uses public Yahoo Finance data
- **Automatic Caching**: 60-second cache to reduce API calls
- **Multiple Symbols**: Supports any valid Yahoo Finance symbol
- **Environment Variables**: Configurable via `.env` file
- **Graceful Fallbacks**: Falls back to local files if yfinance unavailable

## Installation

### 1. Install yfinance

```bash
pip install yfinance
```

Or install all web dependencies:

```bash
cd web
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Create a `.env` file in the project root:

```bash
# Copy example
cp .env.example .env

# Edit .env (optional - defaults work fine)
YFINANCE_CACHE_DURATION=60
```

## Usage

### In Web Application

The web app automatically uses yfinance when available:

1. **Primary Source**: Fetches real-time data from yfinance
2. **Fallback**: Uses local CSV files if yfinance fails
3. **Final Fallback**: Generates demo data

### Programmatic Usage

```python
from src.data.yfinance_fetcher import get_fetcher

# Get fetcher instance
fetcher = get_fetcher()

# Get historical data (last 365 days)
df = fetcher.get_historical_data("AAPL", days=365)

# Get current price
price = fetcher.get_current_price("AAPL")

# Get multiple symbols
data = fetcher.get_multiple_symbols(["AAPL", "GOOGL", "MSFT"])
```

## Supported Symbols

### US Stocks
- AAPL (Apple), GOOGL (Alphabet), MSFT (Microsoft)
- AMZN (Amazon), TSLA (Tesla), NVDA (NVIDIA)
- META (Meta), JPM (JPMorgan), etc.

### Indices
- ^GSPC (S&P 500)
- ^DJI (Dow Jones Industrial Average)
- ^IXIC (NASDAQ Composite)

### ETFs
- SPY (S&P 500 ETF)
- QQQ (NASDAQ-100 ETF)
- VTI (Total Stock Market ETF)

### International
- TCS.NS (Tata Consultancy Services - NSE India)
- RELIANCE.NS (Reliance Industries - NSE India)
- 7203.T (Toyota - Tokyo Stock Exchange)

## Data Format

yfinance returns pandas DataFrame with:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume
- `dividends`: Dividends (if available)
- `stock_splits`: Stock splits (if available)

## Caching

Data is cached for 60 seconds by default:
- **Reduces API calls**: Avoids excessive requests
- **Improves speed**: Faster response times
- **Prevents rate limiting**: Respects API limits

Configure cache duration via `YFINANCE_CACHE_DURATION` in `.env`:

```bash
YFINANCE_CACHE_DURATION=120  # 2 minutes
```

## API Endpoints

### GET `/api/stock/<symbol>`

Fetches stock data with automatic source selection:

**Response:**
```json
{
  "success": true,
  "data": {
    "dates": ["2024-01-01", "2024-01-02", ...],
    "prices": [150.25, 152.10, ...],
    "volumes": [1000000, 1200000, ...],
    "returns": [0.0, 0.012, ...]
  },
  "mode": "yfinance",
  "symbol": "AAPL"
}
```

**Mode values:**
- `"yfinance"`: Real-time data from Yahoo Finance
- `"local"`: Data from local CSV files
- `"demo"`: Generated demo data

## Error Handling

The system includes robust error handling:

1. **Network Errors**: Falls back to cached data if available
2. **Invalid Symbols**: Returns empty DataFrame with error message
3. **Rate Limiting**: Uses cache to avoid excessive requests
4. **Missing yfinance**: Falls back to local files or demo data

## Environment Variables

### Required
None - yfinance works without API keys

### Optional
- `YFINANCE_CACHE_DURATION`: Cache duration in seconds (default: 60)
- `MODEL_DIR`: Path to trained models (default: data/models)

## Performance

- **First Request**: ~1-2 seconds (fetches from Yahoo Finance)
- **Cached Requests**: < 100ms (serves from cache)
- **Cache Hit Rate**: High for frequently accessed symbols

## Limitations

1. **Rate Limiting**: Yahoo Finance may rate limit excessive requests
2. **Data Availability**: Some symbols may not have complete historical data
3. **Market Hours**: Real-time data only available during market hours
4. **International Markets**: Some international symbols may have limited data

## Troubleshooting

### Issue: "No data returned for symbol"

**Solution**: 
- Check symbol is valid (use Yahoo Finance website to verify)
- Try different symbol format (e.g., "AAPL" vs "AAPL.US")
- Check if market is open (some data only available during trading hours)

### Issue: Slow response times

**Solution**:
- Increase cache duration: `YFINANCE_CACHE_DURATION=300`
- Check network connection
- Verify yfinance is installed: `pip install yfinance`

### Issue: Rate limiting errors

**Solution**:
- Increase cache duration
- Reduce number of simultaneous requests
- Use local CSV files for bulk data

## Integration with Predictions

The prediction system can use real-time data:

1. **Fetch Current Data**: Uses yfinance to get latest prices
2. **Process Features**: Computes technical indicators from real-time data
3. **Generate Prediction**: Uses trained models with current features
4. **Return Result**: Provides prediction with current price context

## Best Practices

1. **Use Caching**: Keep default 60-second cache for performance
2. **Handle Errors**: Always check `success` field in API responses
3. **Validate Symbols**: Verify symbols before making requests
4. **Monitor Usage**: Track API calls to avoid rate limiting
5. **Fallback Strategy**: Always have fallback to local files

## Example Workflow

```python
# 1. Fetch real-time data
from src.data.yfinance_fetcher import get_fetcher
fetcher = get_fetcher()
df = fetcher.get_historical_data("AAPL", days=30)

# 2. Process features (if needed)
# ... compute technical indicators ...

# 3. Generate prediction
from scripts.production.production_predictor import ProductionPredictor
predictor = ProductionPredictor(...)
prediction = predictor.predict(features)

# 4. Display result
print(f"Current Price: ${df['close'].iloc[-1]:.2f}")
print(f"Prediction: {prediction}")
```

## Future Enhancements

- **WebSocket Support**: Real-time price updates via WebSocket
- **Multiple Exchanges**: Support for other data providers
- **Data Persistence**: Save fetched data to local database
- **Advanced Caching**: Redis-based distributed caching
- **Rate Limit Management**: Automatic throttling and retry logic
