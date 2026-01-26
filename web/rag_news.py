"""
RAG (Retrieval-Augmented Generation) layer for news retrieval and processing.
"""
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class NewsRAG:
    """
    RAG system for retrieving and processing relevant news for stock symbols.
    """
    
    def __init__(self, news_db_path: Optional[str] = None):
        """
        Initialize RAG news system.
        
        Args:
            news_db_path: Path to news database/JSON file (optional)
        """
        self.news_db_path = news_db_path or "data/raw/news_headlines.csv"
        self.news_cache = []
        self._load_news_data()
    
    def _load_news_data(self):
        """Load news data from file or create demo data."""
        # Try to load from CSV if exists
        if os.path.exists(self.news_db_path):
            try:
                import pandas as pd
                df = pd.read_csv(self.news_db_path)
                if 'date' in df.columns and 'headline' in df.columns:
                    self.news_cache = df.to_dict('records')
                    print(f"✓ Loaded {len(self.news_cache)} real news items from {self.news_db_path}")
                    return
            except Exception as e:
                print(f"⚠ Could not load news from {self.news_db_path}: {e}")
        
        # Try to fetch real-time news from yfinance (optional)
        try:
            self._try_fetch_realtime_news()
            if len(self.news_cache) > 0:
                return
        except Exception as e:
            print(f"⚠ Could not fetch real-time news: {e}")
        
        # Create recent demo news (last 7 days) as fallback
        print("⚠ Using demo news (no real news data found)")
        self._create_recent_demo_news()
    
    def _try_fetch_realtime_news(self):
        """Try to fetch real-time news from yfinance for common symbols."""
        try:
            import yfinance as yf
            
            # Try fetching news for common symbols
            symbols = ['SPY', '^GSPC', 'AAPL', 'MSFT', 'GOOGL']
            all_news = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news[:10]  # Get top 10 recent news
                    
                    for item in news:
                        title = item.get('title', '')
                        pub_time = item.get('providerPublishTime', 0)
                        
                        if pub_time:
                            news_date = datetime.fromtimestamp(pub_time)
                        else:
                            news_date = datetime.now()
                        
                        all_news.append({
                            'date': news_date.strftime('%Y-%m-%d'),
                            'headline': title,
                            'symbol': symbol,
                            'sentiment': 'neutral'  # Could add sentiment analysis here
                        })
                except:
                    continue
            
            if all_news:
                # Remove duplicates
                seen = set()
                unique_news = []
                for news in all_news:
                    key = (news['date'], news['headline'])
                    if key not in seen:
                        seen.add(key)
                        unique_news.append(news)
                
                self.news_cache = unique_news
                print(f"✓ Fetched {len(self.news_cache)} real-time news items from yfinance")
        except ImportError:
            pass  # yfinance not available
    
    def _create_recent_demo_news(self):
        """Create demo news with recent dates (last 7 days)."""
        today = datetime.now()
        demo_news = [
            {
                'date': (today - timedelta(days=1)).strftime('%Y-%m-%d'),
                'headline': 'Market opens with strong momentum as tech stocks rally',
                'symbol': 'AAPL',
                'sentiment': 'positive'
            },
            {
                'date': (today - timedelta(days=2)).strftime('%Y-%m-%d'),
                'headline': 'Federal Reserve signals potential rate cuts in upcoming meetings',
                'symbol': 'AAPL',
                'sentiment': 'positive'
            },
            {
                'date': (today - timedelta(days=3)).strftime('%Y-%m-%d'),
                'headline': 'Tech sector shows resilience amid market volatility',
                'symbol': 'AAPL',
                'sentiment': 'neutral'
            },
            {
                'date': (today - timedelta(days=4)).strftime('%Y-%m-%d'),
                'headline': 'Analysts upgrade price targets following strong quarterly results',
                'symbol': 'AAPL',
                'sentiment': 'positive'
            },
            {
                'date': (today - timedelta(days=5)).strftime('%Y-%m-%d'),
                'headline': 'Regulatory concerns weigh on market sentiment',
                'symbol': 'AAPL',
                'sentiment': 'negative'
            },
            {
                'date': (today - timedelta(days=6)).strftime('%Y-%m-%d'),
                'headline': 'Institutional investors increase positions in blue-chip stocks',
                'symbol': 'AAPL',
                'sentiment': 'positive'
            },
            {
                'date': (today - timedelta(days=7)).strftime('%Y-%m-%d'),
                'headline': 'Market volatility expected as earnings season approaches',
                'symbol': 'AAPL',
                'sentiment': 'neutral'
            }
        ]
        self.news_cache = demo_news
        print(f"✓ Created {len(demo_news)} recent demo news items")
    
    def retrieve_relevant_news(
        self,
        symbol: str,
        limit: int = 5,
        days_back: int = 7
    ) -> List[Dict]:
        """
        Retrieve relevant news for a symbol using RAG approach.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            limit: Maximum number of news items to return
            days_back: How many days back to search
        
        Returns:
            List of relevant news items with dates, headlines, and sentiment
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter news by symbol and date
        relevant_news = []
        for news in self.news_cache:
            news_date = datetime.strptime(news.get('date', ''), '%Y-%m-%d')
            
            # Check if news is recent and relevant
            if news_date >= cutoff_date:
                # Simple keyword matching for symbol relevance
                headline = news.get('headline', '').upper()
                symbol_upper = symbol.upper()
                
                # Check if symbol is mentioned or news is general market news
                if symbol_upper in headline or 'MARKET' in headline or 'FED' in headline or 'TECH' in headline:
                    relevant_news.append({
                        'date': news.get('date'),
                        'headline': news.get('headline', ''),
                        'sentiment': news.get('sentiment', 'neutral'),
                        'relevance_score': self._calculate_relevance(news, symbol)
                    })
        
        # Sort by relevance and date
        relevant_news.sort(key=lambda x: (x['relevance_score'], x['date']), reverse=True)
        
        # Return top N most relevant
        return relevant_news[:limit]
    
    def _calculate_relevance(self, news: Dict, symbol: str) -> float:
        """
        Calculate relevance score for news item.
        
        Args:
            news: News item dictionary
            symbol: Stock symbol
        
        Returns:
            Relevance score (0.0 to 1.0)
        """
        headline = news.get('headline', '').upper()
        symbol_upper = symbol.upper()
        
        score = 0.0
        
        # Direct mention of symbol
        if symbol_upper in headline:
            score += 0.8
        
        # Market-related keywords
        market_keywords = ['MARKET', 'STOCK', 'TRADING', 'INVESTMENT']
        for keyword in market_keywords:
            if keyword in headline:
                score += 0.1
        
        # Sector keywords
        sector_keywords = ['TECH', 'TECHNOLOGY', 'EARNINGS', 'QUARTERLY']
        for keyword in sector_keywords:
            if keyword in headline:
                score += 0.15
        
        # Recent news gets slight boost
        try:
            news_date = datetime.strptime(news.get('date', ''), '%Y-%m-%d')
            days_ago = (datetime.now() - news_date).days
            if days_ago <= 2:
                score += 0.2
            elif days_ago <= 5:
                score += 0.1
        except:
            pass
        
        return min(score, 1.0)
    
    def generate_enhanced_news(
        self,
        symbol: str,
        retrieved_news: List[Dict]
    ) -> List[Dict]:
        """
        Enhance retrieved news with context and summaries.
        
        Args:
            symbol: Stock symbol
            retrieved_news: List of retrieved news items
        
        Returns:
            Enhanced news items with additional context
        """
        enhanced = []
        for news in retrieved_news:
            enhanced_item = {
                'date': news['date'],
                'headline': news['headline'],
                'sentiment': news['sentiment'],
                'symbol': symbol,
                'relevance': news.get('relevance_score', 0.0)
            }
            
            # Add contextual information
            if news['sentiment'] == 'positive':
                enhanced_item['impact'] = 'Bullish signal for ' + symbol
            elif news['sentiment'] == 'negative':
                enhanced_item['impact'] = 'Bearish signal for ' + symbol
            else:
                enhanced_item['impact'] = 'Neutral market development'
            
            enhanced.append(enhanced_item)
        
        return enhanced
    
    def get_recent_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Get recent relevant news for a symbol using RAG.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of news items
        
        Returns:
            List of recent relevant news items
        """
        # Retrieve relevant news
        retrieved = self.retrieve_relevant_news(symbol, limit=limit)
        
        # Enhance with context
        enhanced = self.generate_enhanced_news(symbol, retrieved)
        
        return enhanced


# Global RAG instance
_rag_instance = None

def get_rag_instance() -> NewsRAG:
    """Get or create global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = NewsRAG()
    return _rag_instance