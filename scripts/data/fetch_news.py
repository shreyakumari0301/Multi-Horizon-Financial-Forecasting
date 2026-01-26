"""
Automatically fetch real news headlines from yfinance.

This script fetches real news headlines for stock symbols and saves them
in the format needed for training (date, headline).
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf


def fetch_news_for_symbol(symbol: str, days_back: int = 365, max_retries: int = 3, delay: float = 2.0, timeout: int = 60) -> pd.DataFrame:
    """
    Fetch news headlines for a symbol using yfinance with retry logic.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'SPY', '^GSPC')
        days_back: How many days back to fetch news
        max_retries: Maximum number of retry attempts
        delay: Delay between retries (seconds)
    
    Returns:
        DataFrame with columns: date, headline
    """
    for attempt in range(max_retries):
        try:
            # yfinance now requires curl_cffi, not requests
            # Try to use curl_cffi if available, otherwise use default
            try:
                from curl_cffi import requests as curl_requests
                session = curl_requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                ticker = yf.Ticker(symbol, session=session)
            except ImportError:
                # Fallback: use yfinance without custom session (it will use curl_cffi internally)
                ticker = yf.Ticker(symbol)
            
            # Add timeout and retry logic
            try:
                # Try to fetch news with explicit timeout
                news = ticker.news
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limiting (429) or connection issues
                if ("429" in str(e) or "rate limit" in error_str or "too many requests" in error_str):
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt) * 2  # Longer wait for rate limits
                        print(f"  ⚠ Rate limited for {symbol}, waiting {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        print(f"     Tip: Yahoo Finance is throttling requests. Waiting longer...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ⚠ Rate limited for {symbol} after {max_retries} attempts")
                        print(f"     Error: Rate limiting (429) - Yahoo Finance is blocking requests")
                        print(f"     Solution: Wait 10-15 minutes and try again, or use fewer symbols")
                elif "timeout" in error_str or "connection" in error_str or "failed to connect" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        print(f"  ⚠ Connection issue for {symbol}, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  ⚠ Connection failed for {symbol} after {max_retries} attempts")
                        print(f"     Error: {str(e)[:100]}")
                elif "curl_cffi" in error_str:
                    print(f"  ⚠ yfinance requires curl_cffi. Installing...")
                    try:
                        import subprocess
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "curl_cffi"])
                        print(f"  ✓ Installed curl_cffi. Please run the script again.")
                    except:
                        print(f"  ⚠ Failed to install curl_cffi. Please install manually: pip install curl_cffi")
                    return pd.DataFrame(columns=['date', 'headline'])
                raise
            
            if not news:
                print(f"  ⚠ No news found for {symbol}")
                return pd.DataFrame(columns=['date', 'headline'])
            
            # Convert to DataFrame
            news_list = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in news:
                # yfinance news format changed - now has nested 'content' structure
                # Structure: {'id': '...', 'content': {'title': '...', 'providerPublishTime': ...}}
                content = item.get('content', {})
                
                # Get title from nested content
                title = (content.get('title') or 
                        item.get('title') or 
                        content.get('headline') or 
                        item.get('headline') or 
                        content.get('linkTitle') or 
                        item.get('linkTitle') or 
                        content.get('summary') or 
                        item.get('summary') or '')
                
                # Skip if no title found
                if not title or (isinstance(title, str) and len(title.strip()) == 0):
                    continue
                
                # Get publish time from nested content or top level
                pub_time = (content.get('providerPublishTime') or 
                           item.get('providerPublishTime') or 
                           content.get('pubDate') or 
                           item.get('pubDate') or 0)
                
                if pub_time:
                    try:
                        # Handle both Unix timestamp and other formats
                        if isinstance(pub_time, (int, float)):
                            news_date = datetime.fromtimestamp(pub_time)
                        else:
                            news_date = datetime.fromtimestamp(int(pub_time))
                    except (ValueError, OSError):
                        news_date = datetime.now()
                else:
                    # If no timestamp, use current date
                    news_date = datetime.now()
                
                # Only include recent news
                if news_date >= cutoff_date:
                    # Get publisher and link from nested content or top level
                    publisher = content.get('publisher', {})
                    if isinstance(publisher, dict):
                        publisher = publisher.get('name', '') or publisher.get('displayName', '')
                    else:
                        publisher = publisher or item.get('publisher', '')
                    
                    link = content.get('clickThroughUrl', {})
                    if isinstance(link, dict):
                        link = link.get('url', '') or link.get('canonicalUrl', '')
                    else:
                        link = link or item.get('link', '') or item.get('url', '')
                    
                    news_list.append({
                        'date': news_date.strftime('%Y-%m-%d'),
                        'headline': title.strip(),
                        'symbol': symbol,
                        'publisher': publisher if isinstance(publisher, str) else '',
                        'link': link if isinstance(link, str) else ''
                    })
            
            if not news_list:
                print(f"  ⚠ No valid news items found for {symbol} (all had empty titles or were outside date range)")
                return pd.DataFrame(columns=['date', 'headline'])
            
            df = pd.DataFrame(news_list)
            
            # Debug: show sample headlines
            if len(df) > 0:
                sample_headline = df['headline'].iloc[0][:60] if len(df['headline'].iloc[0]) > 0 else "[empty]"
                print(f"  ✓ Fetched {len(df)} news items for {symbol}")
                print(f"    Sample: {sample_headline}...")
            else:
                print(f"  ⚠ No valid news items for {symbol}")
            
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    print(f"  ⚠ Connection issue for {symbol}, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"  ⚠ Error for {symbol}: {error_msg[:50]}..., retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠ Failed to fetch news for {symbol} after {max_retries} attempts: {str(e)[:100]}")
                return pd.DataFrame(columns=['date', 'headline'])
    
    return pd.DataFrame(columns=['date', 'headline'])


def fetch_all_news(
    symbols: list = None,
    output_path: str = "data/raw/news_headlines.csv",
    days_back: int = 365,
    delay: float = 1.5,
    max_retries: int = 3,
    timeout: int = 60
) -> pd.DataFrame:
    """
    Fetch news for multiple symbols and combine into single DataFrame.
    
    Args:
        symbols: List of symbols to fetch news for
        output_path: Path to save news CSV
        days_back: How many days back to fetch
    
    Returns:
        Combined DataFrame with all news
    """
    if symbols is None:
        # Default symbols - start with most reliable ones
        # Focus on major US indices/stocks that have consistent news coverage
        symbols = [
            "SPY",                   # S&P 500 ETF (most reliable, high volume)
            "AAPL",                  # Apple (very active news)
            "MSFT",                  # Microsoft
            "^GSPC",                 # S&P 500 index
            "BTC-USD",               # Bitcoin (if available)
        ]
    
    print("=" * 70)
    print("Fetching Real News Headlines from yfinance")
    print("=" * 70)
    print(f"Symbols: {len(symbols)}")
    print(f"Days back: {days_back}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    all_news = []
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        print(f"\nFetching news for {symbol}... ({i+1}/{len(symbols)})")
        
        # Add delay between requests to avoid rate limiting (longer for rate limits)
        if i > 0:
            time.sleep(delay * 2)  # Longer delay to avoid 429 errors
        
        news_df = fetch_news_for_symbol(symbol, days_back, max_retries=max_retries, delay=delay, timeout=timeout)
        if len(news_df) > 0:
            all_news.append(news_df)
            successful += 1
        else:
            failed += 1
    
    print(f"\n  Summary: {successful} successful, {failed} failed")
    
    if not all_news:
        print("\n⚠ No news found for any symbols")
        print("  Tip: Try again later or use --symbols to fetch specific symbols")
        return pd.DataFrame(columns=['date', 'headline'])
    
    # Combine all news
    combined = pd.concat(all_news, ignore_index=True)
    
    # Remove duplicates (same headline on same date)
    combined = combined.drop_duplicates(subset=['date', 'headline'])
    
    # Sort by date
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date')
    combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')
    
    # Keep only date and headline for training
    output_df = combined[['date', 'headline']].copy()
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"✓ Saved {len(output_df)} unique news headlines")
    print(f"  Date range: {output_df['date'].min()} to {output_df['date'].max()}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)
    
    return output_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch real news headlines from yfinance")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="List of symbols to fetch news for (default: all symbols from data extraction)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/news_headlines.csv",
        help="Output path for news CSV"
    )
    parser.add_argument(
        "--days_back",
        type=int,
        default=365,
        help="How many days back to fetch news (default: 365)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between symbol requests in seconds (default: 1.5)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts per symbol (default: 3)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Update fetch_all_news to use delay parameter
    # (We'll modify the function to accept delay)
    fetch_all_news(
        symbols=args.symbols,
        output_path=args.output,
        days_back=args.days_back,
        delay=args.delay,
        max_retries=args.max_retries,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()
