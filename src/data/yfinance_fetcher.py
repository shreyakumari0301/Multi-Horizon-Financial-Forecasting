"""
Real-time stock data fetcher using yfinance API.
"""
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time


class YFinanceFetcher:
    """
    Fetches real-time stock data from Yahoo Finance using yfinance.
    """
    
    def __init__(self, cache_duration: int = 60):
        """
        Initialize yfinance fetcher.
        
        Args:
            cache_duration: Cache duration in seconds (default: 60)
        """
        self.cache_duration = cache_duration
        self._cache = {}
        self._cache_timestamps = {}
    
    def _get_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from cache if still valid."""
        if symbol in self._cache:
            timestamp = self._cache_timestamps.get(symbol, 0)
            if time.time() - timestamp < self.cache_duration:
                return self._cache[symbol].copy()
        return None
    
    def _update_cache(self, symbol: str, data: pd.DataFrame):
        """Update cache with new data."""
        self._cache[symbol] = data.copy()
        self._cache_timestamps[symbol] = time.time()
    
    def fetch_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cached = self._get_from_cache(symbol)
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            
            # Standardize column names - preserve original column mapping
            original_cols = df.columns.tolist()
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Explicitly use 'Close' price (not 'Adj Close')
            # yfinance returns 'Close' and 'Adj Close' - we want 'Close' for actual prices
            if 'close' in df.columns:
                # Ensure we're using Close, not Adj Close
                # If both exist, explicitly use 'close' (which came from 'Close')
                if 'adj_close' in df.columns:
                    # Keep 'close' as is (it's from 'Close'), don't use 'adj_close'
                    pass
            else:
                # Try to find Close column with different casing
                close_col = None
                for orig_col, new_col in zip(original_cols, df.columns):
                    if 'close' in new_col.lower() and 'adj' not in new_col.lower():
                        close_col = orig_col
                        break
                
                if close_col:
                    df['close'] = df[close_col]
                elif 'adj_close' in df.columns:
                    print(f"⚠ Warning: Using Adj Close for {symbol} (Close not available)")
                    df['close'] = df['adj_close']
                else:
                    raise ValueError(f"Close price column not found for {symbol}. Available columns: {df.columns.tolist()}")
            
            # Debug: Print latest price for verification
            if not df.empty:
                latest_date = df.index[-1]
                latest_close = df['close'].iloc[-1]
                print(f"✓ {symbol} - Latest: {latest_date.strftime('%Y-%m-%d')} - Close: ${latest_close:.2f}")
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Update cache
            self._update_cache(symbol, df)
            
            return df.copy()
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            raise
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Current price or None if unavailable
        """
        try:
            df = self.fetch_stock_data(symbol, period="1d", interval="1d")
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
        
        Returns:
            DataFrame with historical OHLCV data
        """
        # Map days to yfinance period
        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"
        
        return self.fetch_stock_data(symbol, period=period, interval="1d")
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_stock_data(symbol, period=period)
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results


# Global fetcher instance
_fetcher_instance = None

def get_fetcher() -> YFinanceFetcher:
    """Get or create global fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        cache_duration = int(os.getenv('YFINANCE_CACHE_DURATION', '60'))
        _fetcher_instance = YFinanceFetcher(cache_duration=cache_duration)
    return _fetcher_instance
