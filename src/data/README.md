# Data Fetching Module

Real-time stock data fetching using yfinance API.

## Overview

This module provides real-time stock data fetching capabilities using the yfinance library, which accesses Yahoo Finance data without requiring API keys.

## Components

### YFinanceFetcher (`yfinance_fetcher.py`)

Fetches real-time and historical stock data from Yahoo Finance.

**Key Features:**
- Real-time price data
- Historical data (1 day to 5 years)
- Automatic caching (60 seconds default)
- Multiple symbol support
- Error handling and fallbacks

**Usage:**
```python
from src.data.yfinance_fetcher import get_fetcher

fetcher = get_fetcher()

# Get historical data
df = fetcher.get_historical_data("AAPL", days=365)

# Get current price
price = fetcher.get_current_price("AAPL")

# Get multiple symbols
data = fetcher.get_multiple_symbols(["AAPL", "GOOGL", "MSFT"])
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Cache duration in seconds (default: 60)
YFINANCE_CACHE_DURATION=60
```

## API Integration

The web application automatically uses yfinance when available:

1. **Primary**: Fetches real-time data from yfinance
2. **Fallback**: Uses local CSV files if yfinance fails
3. **Final Fallback**: Generates demo data

## Supported Symbols

Any valid Yahoo Finance symbol:
- Stocks: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, etc.
- Indices: ^GSPC (S&P 500), ^DJI (Dow Jones)
- ETFs: SPY, QQQ, etc.
- International: TCS.NS (NSE), RELIANCE.NS, etc.

## Data Format

Returns pandas DataFrame with columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume
- `dividends`: Dividends (if available)
- `stock_splits`: Stock splits (if available)

## Caching

Data is cached for 60 seconds by default to:
- Reduce API calls
- Improve response time
- Avoid rate limiting

Cache duration can be configured via `YFINANCE_CACHE_DURATION` environment variable.

## Error Handling

The fetcher includes robust error handling:
- Network errors: Returns cached data if available
- Invalid symbols: Returns empty DataFrame
- Rate limiting: Uses cache to avoid excessive requests

## Installation

```bash
pip install yfinance
```

Or install all web dependencies:
```bash
cd web
pip install -r requirements.txt
```
