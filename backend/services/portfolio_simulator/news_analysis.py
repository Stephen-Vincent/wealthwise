# services/news_analysis.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from textblob import TextBlob
import os

logger = logging.getLogger(__name__)

class NewsAnalysisService:
    def __init__(self, finnhub_api_key: Optional[str] = None):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", finnhub_api_key)
        if not self.finnhub_api_key:
            logger.warning("FINNHUB_API_KEY not set - news analysis will be disabled")
        self.base_url = "https://finnhub.io/api/v1"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_market_news(self, symbols: List[str], days_back: int = 7) -> Dict:
        """
        Fetch recent news for given symbols from Finnhub
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            days_back: Number of days to look back for news
        
        Returns:
            Dictionary with symbol as key and list of news articles as value
        """
        if not self.finnhub_api_key:
            logger.warning("No API key available - returning empty news data")
            return {symbol: [] for symbol in symbols}
            
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        news_data = {}
        
        # Create tasks for concurrent requests
        tasks = []
        for symbol in symbols:
            task = self._fetch_company_news(symbol, from_date, to_date)
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching news for {symbol}: {result}")
                news_data[symbol] = []
            else:
                news_data[symbol] = result
        
        return news_data
    
    async def _fetch_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Fetch company news for a single symbol"""
        if not self.finnhub_api_key:
            logger.warning(f"No API key available for news fetching {symbol}")
            return []
            
        url = f"{self.base_url}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.finnhub_api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                elif response.status == 429:
                    logger.warning(f"Rate limit hit for {symbol}, waiting...")
                    await asyncio.sleep(1)  # Wait 1 second and retry
                    return await self._fetch_company_news(symbol, from_date, to_date)
                else:
                    logger.error(f"HTTP {response.status} for {symbol}")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching news for {symbol}: {e}")
            return []
    
    async def get_general_market_news(self, category: str = "general", limit: int = 50) -> List[Dict]:
        """
        Get general market news from Finnhub
        
        Args:
            category: News category ('general', 'forex', 'crypto', 'merger')
            limit: Maximum number of articles to return
        
        Returns:
            List of news articles
        """
        if not self.finnhub_api_key:
            logger.warning("No API key available for general news")
            return []
            
        url = f"{self.base_url}/news"
        params = {
            'category': category,
            'token': self.finnhub_api_key
        }
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data[:limit] if isinstance(data, list) else []
                else:
                    logger.error(f"HTTP {response.status} for general news")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching general news: {e}")
            return []
    
    async def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment of news articles using TextBlob
        
        Args:
            articles: List of news articles from Finnhub
        
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not articles:
            return {
                'average_sentiment': 0.0,
                'sentiment_trend': [],
                'total_articles': 0,
                'sentiment_category': 'Neutral',
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
            }
        
        sentiments = []
        sentiment_scores = []
        
        for article in articles:
            try:
                # Extract text from Finnhub article format
                headline = article.get('headline', '')
                summary = article.get('summary', '')
                
                # Combine headline and summary for analysis
                text = f"{headline} {summary}".strip()
                
                if text:
                    # Analyze sentiment using TextBlob
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity  # -1 to 1
                    subjectivity = blob.sentiment.subjectivity  # 0 to 1
                    
                    sentiment_data = {
                        'polarity': polarity,
                        'subjectivity': subjectivity,
                        'headline': headline,
                        'datetime': article.get('datetime', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'category': self._categorize_sentiment(polarity)
                    }
                    
                    sentiments.append(sentiment_data)
                    sentiment_scores.append(polarity)
                    
            except Exception as e:
                logger.error(f"Error analyzing sentiment for article: {e}")
                continue
        
        # Calculate overall metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Calculate sentiment distribution
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        for score in sentiment_scores:
            if score > 0.1:
                distribution['positive'] += 1
            elif score < -0.1:
                distribution['negative'] += 1
            else:
                distribution['neutral'] += 1
        
        return {
            'average_sentiment': round(avg_sentiment, 3),
            'sentiment_trend': sentiments,
            'total_articles': len(sentiments),
            'sentiment_category': self._categorize_sentiment(avg_sentiment),
            'sentiment_distribution': distribution,
            'sentiment_strength': self._calculate_sentiment_strength(sentiment_scores)
        }
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score into readable categories"""
        if score > 0.3:
            return "Very Positive"
        elif score > 0.1:
            return "Positive"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def _calculate_sentiment_strength(self, scores: List[float]) -> str:
        """Calculate the strength/volatility of sentiment"""
        if not scores:
            return "No Data"
        
        # Calculate standard deviation as a measure of sentiment volatility
        import statistics
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            if std_dev > 0.4:
                return "High Volatility"
            elif std_dev > 0.2:
                return "Moderate Volatility"
            else:
                return "Low Volatility"
        else:
            return "Insufficient Data"
    
    async def get_market_events(self, articles: List[Dict]) -> List[Dict]:
        """
        Identify potential market-moving events from news articles
        
        Args:
            articles: List of news articles
        
        Returns:
            List of identified events
        """
        event_keywords = {
            'earnings': ['earnings', 'quarterly results', 'Q1', 'Q2', 'Q3', 'Q4', 'revenue', 'profit'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'buyout', 'deal', 'acquire'],
            'regulatory': ['FDA', 'SEC', 'regulation', 'compliance', 'approval', 'investigation'],
            'leadership': ['CEO', 'CFO', 'president', 'resignation', 'appointed', 'fired', 'steps down'],
            'product': ['launch', 'new product', 'release', 'unveil', 'patent', 'innovation'],
            'financial': ['bankruptcy', 'debt', 'loan', 'dividend', 'split', 'buyback'],
            'legal': ['lawsuit', 'court', 'settlement', 'legal action', 'fine', 'penalty']
        }
        
        events = []
        
        for article in articles:
            headline = article.get('headline', '').lower()
            summary = article.get('summary', '').lower()
            text = f"{headline} {summary}"
            
            detected_events = []
            for event_type, keywords in event_keywords.items():
                # FIXED: Changed Any to any (built-in Python function)
                if any(keyword.lower() in text for keyword in keywords):
                    detected_events.append(event_type)
            
            if detected_events:
                events.append({
                    'headline': article.get('headline', ''),
                    'datetime': article.get('datetime', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'event_types': detected_events,
                    'summary': article.get('summary', '')[:200] + '...' if len(article.get('summary', '')) > 200 else article.get('summary', '')
                })
        
        return events
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()


