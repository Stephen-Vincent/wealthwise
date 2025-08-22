from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import aiohttp
import os
from .news_analysis import NewsAnalysisService
import calendar

# Set up logging
logger = logging.getLogger(__name__)

class AIAnalysisService:
    """
    Enhanced AI-powered portfolio analysis service using GROQ API
    Now includes comprehensive news sentiment analysis and market event detection
    Provides detailed explanations of what happened to your specific portfolio
    Focuses on educational explanations of market movements for beginners
    """
    
    def __init__(self):
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_api_key = os.getenv("GROQ_API_KEY")  # Set from environment
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")  # For news analysis
        self.model = "llama3-70b-8192"  # Or mixtral-8x7b-32768
    
    def extract_symbols_from_portfolio(self, portfolio_data: dict) -> List[str]:
        """Extract stock symbols from portfolio data"""
        symbols = []
        
        # Handle different portfolio data formats
        if 'holdings' in portfolio_data:
            for holding in portfolio_data['holdings']:
                if 'symbol' in holding:
                    symbols.append(holding['symbol'].upper())
        elif 'positions' in portfolio_data:
            for position in portfolio_data['positions']:
                if 'ticker' in position:
                    symbols.append(position['ticker'].upper())
        elif 'stocks' in portfolio_data:
            symbols = [stock.upper() for stock in portfolio_data['stocks']]
        elif 'stocks_picked' in portfolio_data:
            # Handle the stocks_picked format from simulation
            for stock in portfolio_data['stocks_picked']:
                if 'symbol' in stock:
                    symbols.append(stock['symbol'].upper())
                elif 'ticker' in stock:
                    symbols.append(stock['ticker'].upper())
        
        return list(set(symbols))  # Remove duplicates
    
    async def generate_portfolio_summary(
        self, 
        stocks_picked: List[Dict], 
        user_data: Dict[str, Any], 
        risk_score: int, 
        risk_label: str, 
        simulation_results: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive AI summary with detailed portfolio news analysis
        """
        try:
            # Analyze market movements from timeline data
            market_analysis = self._analyze_market_movements(simulation_results, user_data)
            
            # Get comprehensive portfolio news analysis for the entire investment period
            portfolio_news_analysis = await self._analyze_portfolio_news_history(
                stocks_picked, user_data, simulation_results
            )
            
            # Generate comprehensive prompt with market movements and detailed news analysis
            prompt = self._create_comprehensive_market_prompt_with_news(
                stocks_picked, user_data, risk_score, risk_label, 
                simulation_results, market_analysis, portfolio_news_analysis
            )
            
            response = await self._get_groq_response(prompt)
            return self._format_ai_response(response)
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._get_formatted_fallback_summary_with_movements(
                user_data, simulation_results, stocks_picked, risk_label
            )
    
    async def analyze_portfolio_performance(self, portfolio_data: dict):
        """Analyze existing portfolio performance with news context"""
        try:
            symbols = self.extract_symbols_from_portfolio(portfolio_data)
            
            if not symbols:
                return {"error": "No valid symbols found in portfolio data"}
            
            # Get news analysis
            news_analysis = await self._get_portfolio_news_analysis(symbols)
            
            # Generate AI insights about performance with news context
            prompt = self._create_performance_analysis_prompt(portfolio_data, news_analysis)
            
            ai_response = await self._get_groq_response(prompt)
            
            return {
                "performance_analysis": ai_response,
                "news_sentiment": news_analysis,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {"error": str(e)}
    
    async def analyze_risk_allocation(self, portfolio_data: dict):
        """Analyze portfolio risk and allocation with market sentiment"""
        try:
            symbols = self.extract_symbols_from_portfolio(portfolio_data)
            
            if not symbols:
                return {"error": "No valid symbols found in portfolio data"}
            
            # Get news analysis for risk assessment
            news_analysis = await self._get_portfolio_news_analysis(symbols)
            
            # Generate AI risk analysis with news context
            prompt = self._create_risk_analysis_prompt(portfolio_data, news_analysis)
            
            ai_response = await self._get_groq_response(prompt)
            
            return {
                "risk_analysis": ai_response,
                "market_sentiment": news_analysis,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk allocation: {e}")
            return {"error": str(e)}
    
    async def explain_portfolio_changes(self, portfolio_data: dict, previous_data: dict = None):
        """Explain portfolio changes over time with news context"""
        try:
            symbols = self.extract_symbols_from_portfolio(portfolio_data)
            
            if not symbols:
                return {"error": "No valid symbols found in portfolio data"}
            
            # Get recent news that might explain changes
            news_analysis = await self._get_portfolio_news_analysis(symbols, days_back=14)
            
            # Generate explanation with news context
            prompt = self._create_changes_explanation_prompt(portfolio_data, previous_data, news_analysis)
            
            ai_response = await self._get_groq_response(prompt)
            
            return {
                "changes_explanation": ai_response,
                "relevant_news": news_analysis,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error explaining portfolio changes: {e}")
            return {"error": str(e)}
    
    async def analyze_portfolio_with_context(self, portfolio_data: dict, days_back: int = 7) -> Dict:
        """
        Enhanced portfolio analysis with news sentiment and market context
        """
        symbols = self.extract_symbols_from_portfolio(portfolio_data)
        
        if not symbols:
            return {"error": "No valid symbols found in portfolio data"}
        
        try:
            # Get comprehensive news analysis
            async with NewsAnalysisService(self.finnhub_api_key) as news_service:
                # Get news and sentiment for all symbols
                news_data = await news_service.get_market_news(symbols, days_back)
                
                analysis_results = {}
                
                for symbol in symbols:
                    articles = news_data.get(symbol, [])
                    
                    # Analyze sentiment
                    sentiment_analysis = await news_service.analyze_sentiment(articles)
                    
                    # Detect market events
                    events = await news_service.get_market_events(articles)
                    
                    # Get price data for correlation
                    price_data = await self._get_price_data(symbol, days_back)
                    
                    # Generate AI insights for this specific stock
                    stock_insights = await self._generate_stock_insights(symbol, sentiment_analysis, events, price_data)
                    
                    analysis_results[symbol] = {
                        'sentiment_analysis': sentiment_analysis,
                        'market_events': events,
                        'price_data': price_data,
                        'ai_insights': stock_insights,
                        'news_impact_score': self._calculate_news_impact(sentiment_analysis, events, price_data)
                    }
        
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in portfolio context analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_portfolio_news_history(
        self, 
        stocks_picked: List[Dict], 
        user_data: Dict[str, Any],
        simulation_results: Dict[str, Any]
    ) -> Dict:
        """
        Comprehensive analysis of news events that affected the portfolio during investment period
        """
        try:
            # Extract portfolio symbols
            symbols = []
            for stock in stocks_picked:
                if 'symbol' in stock:
                    symbols.append(stock['symbol'].upper())
                elif 'ticker' in stock:
                    symbols.append(stock['ticker'].upper())
            
            if not symbols:
                return {"error": "No symbols found in portfolio"}
            
            # Get timeline data to understand the investment period
            timeline = simulation_results.get("timeline", {}).get("portfolio", [])
            timeframe = user_data.get("timeframe", 5)
            
            # Analyze different periods of the investment journey
            news_analysis = {
                "investment_period_overview": {},
                "major_events_by_year": {},
                "portfolio_specific_events": {},
                "market_correlation_analysis": {},
                "news_impact_timeline": [],
                "sector_news_analysis": {},
                "educational_insights": {}
            }
            
            # Get recent news for current market context (last 30 days)
            recent_news = await self._get_comprehensive_portfolio_news(symbols, days_back=30)
            
            # Simulate historical news analysis by analyzing different market periods
            # (In production, you'd want historical news data)
            historical_analysis = await self._simulate_historical_news_analysis(
                symbols, timeframe, timeline, simulation_results
            )
            
            # Combine recent and historical analysis
            news_analysis.update({
                "recent_news_context": recent_news,
                "historical_market_events": historical_analysis,
                "portfolio_news_summary": await self._create_portfolio_news_summary(
                    symbols, recent_news, historical_analysis, timeframe
                )
            })
            
            return news_analysis
            
        except Exception as e:
            logger.error(f"Error in portfolio news history analysis: {e}")
            return {"error": str(e)}
    
    async def _get_comprehensive_portfolio_news(self, symbols: List[str], days_back: int = 30) -> Dict:
        """Get detailed news analysis for portfolio with enhanced event detection"""
        try:
            async with NewsAnalysisService(self.finnhub_api_key) as news_service:
                # Get news for all symbols
                news_data = await news_service.get_market_news(symbols, days_back)
                
                comprehensive_analysis = {
                    "symbol_specific_analysis": {},
                    "portfolio_wide_events": [],
                    "sector_trends": {},
                    "sentiment_timeline": [],
                    "major_market_events": [],
                    "earnings_calendar": [],
                    "regulatory_events": [],
                    "market_sentiment_summary": {}
                }
                
                all_events = []
                sentiment_scores = []
                
                for symbol in symbols:
                    articles = news_data.get(symbol, [])
                    
                    if articles:
                        # Enhanced sentiment analysis
                        sentiment_result = await news_service.analyze_sentiment(articles)
                        
                        # Detailed event detection
                        events = await news_service.get_market_events(articles)
                        
                        # Categorize events by importance
                        major_events = [e for e in events if self._is_major_event(e)]
                        earnings_events = [e for e in events if 'earnings' in e.get('event_types', [])]
                        regulatory_events = [e for e in events if 'regulatory' in e.get('event_types', [])]
                        
                        symbol_analysis = {
                            "sentiment": sentiment_result,
                            "all_events": events,
                            "major_events": major_events,
                            "earnings_events": earnings_events,
                            "regulatory_events": regulatory_events,
                            "article_count": len(articles),
                            "news_density": len(articles) / days_back,  # Articles per day
                            "sentiment_volatility": self._calculate_sentiment_volatility(articles),
                            "top_headlines": [a.get('headline', '') for a in articles[:5]]
                        }
                        
                        comprehensive_analysis["symbol_specific_analysis"][symbol] = symbol_analysis
                        
                        # Add to portfolio-wide collections
                        all_events.extend(events)
                        if sentiment_result.get("average_sentiment"):
                            sentiment_scores.append(sentiment_result["average_sentiment"])
                        
                        # Add major events to portfolio-wide events
                        comprehensive_analysis["portfolio_wide_events"].extend(major_events)
                        comprehensive_analysis["earnings_calendar"].extend(earnings_events)
                        comprehensive_analysis["regulatory_events"].extend(regulatory_events)
                
                # Calculate portfolio-wide metrics
                comprehensive_analysis["market_sentiment_summary"] = {
                    "overall_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                    "sentiment_range": {"min": min(sentiment_scores), "max": max(sentiment_scores)} if sentiment_scores else {"min": 0, "max": 0},
                    "total_articles": sum(len(news_data.get(s, [])) for s in symbols),
                    "total_events": len(all_events),
                    "event_density": len(all_events) / len(symbols) if symbols else 0
                }
                
                return comprehensive_analysis
                
        except Exception as e:
            logger.error(f"Error getting comprehensive portfolio news: {e}")
            return {"error": str(e)}
    
    async def _simulate_historical_news_analysis(
        self, 
        symbols: List[str], 
        timeframe: int, 
        timeline: List[Dict],
        simulation_results: Dict
    ) -> Dict:
        """
        Simulate historical news analysis based on market movements and known events
        """
        historical_events = {
            "market_cycles": [],
            "major_corrections": [],
            "bull_market_periods": [],
            "sector_rotations": [],
            "economic_events": [],
            "company_specific_events": {}
        }
        
        # Analyze timeline for major movements that would have had news coverage
        if timeline and len(timeline) > 1:
            values = [period.get("value", 0) for period in timeline]
            dates = [period.get("date", "") for period in timeline]
            
            # Identify major movements that would correlate with news events
            for i in range(1, len(values)):
                current_value = values[i]
                previous_value = values[i-1]
                
                if previous_value > 0:
                    change_pct = ((current_value - previous_value) / previous_value) * 100
                    
                    # Simulate news events for significant movements
                    if abs(change_pct) > 10:  # Significant movement
                        event_type = "market_rally" if change_pct > 0 else "market_correction"
                        
                        simulated_event = {
                            "date": dates[i],
                            "type": event_type,
                            "magnitude": abs(change_pct),
                            "likely_news_themes": self._get_likely_news_themes(change_pct, dates[i]),
                            "portfolio_impact": f"{change_pct:+.1f}%",
                            "educational_context": self._get_educational_context(change_pct, event_type)
                        }
                        
                        if event_type == "market_correction":
                            historical_events["major_corrections"].append(simulated_event)
                        else:
                            historical_events["bull_market_periods"].append(simulated_event)
        
        # Add sector-specific events based on portfolio composition
        historical_events["sector_rotations"] = self._simulate_sector_events(symbols, timeframe)
        
        # Add economic events that would have affected the portfolio
        historical_events["economic_events"] = self._simulate_economic_events(timeframe)
        
        return historical_events
    
    def _get_likely_news_themes(self, change_pct: float, date: str) -> List[str]:
        """Get likely news themes that could explain market movements"""
        if change_pct < -20:
            return [
                "Major economic recession fears",
                "Financial crisis or banking stress",
                "Geopolitical tensions escalating",
                "Pandemic or health crisis impact",
                "Central bank emergency measures"
            ]
        elif change_pct < -10:
            return [
                "Interest rate concerns",
                "Inflation worries",
                "Earnings disappointments",
                "Economic slowdown signals",
                "Trade war tensions"
            ]
        elif change_pct > 20:
            return [
                "Strong economic recovery",
                "Major policy stimulus announced",
                "Breakthrough technology news",
                "Trade deal agreements",
                "Corporate earnings beats"
            ]
        elif change_pct > 10:
            return [
                "Positive economic data",
                "Fed policy support",
                "Strong earnings season",
                "Market optimism returning",
                "Sector rotation gains"
            ]
        else:
            return ["Normal market fluctuations", "Mixed economic signals"]
    
    def _get_educational_context(self, change_pct: float, event_type: str) -> str:
        """Provide educational context for market movements"""
        if event_type == "market_correction":
            if change_pct < -30:
                return "Bear markets like this (30%+ drops) typically happen every 5-7 years and test every investor's patience. They often create the best long-term buying opportunities."
            elif change_pct < -20:
                return "Market corrections of 20%+ are considered bear markets. Historically, these have always been followed by recoveries that reach new highs."
            else:
                return "Corrections of 10-20% are normal and healthy for markets. They help reset valuations and create opportunities for patient investors."
        else:  # market_rally
            if change_pct > 30:
                return "Strong rallies like this often follow major corrections. Markets tend to recover faster than they fall, rewarding investors who stayed the course."
            elif change_pct > 20:
                return "Bull market rallies of 20%+ show how quickly markets can recover from pessimism. This is why timing the market is so difficult."
            else:
                return "Steady gains like this represent healthy market growth, often driven by improving economic fundamentals."
    
    def _simulate_sector_events(self, symbols: List[str], timeframe: int) -> List[Dict]:
        """Simulate sector-specific events that would have affected the portfolio"""
        sector_events = []
        
        # Map common ETF symbols to sectors for simulation
        sector_mapping = {
            "QQQ": "Technology",
            "VGT": "Technology", 
            "ARKK": "Innovation",
            "ARKQ": "Autonomous Technology",
            "VWO": "Emerging Markets",
            "IBB": "Biotechnology",
            "FINX": "Financial Technology",
            "VUG": "Growth Stocks",
            "COIN": "Cryptocurrency",
            "BITO": "Cryptocurrency"
        }
        
        represented_sectors = set()
        for symbol in symbols:
            if symbol in sector_mapping:
                represented_sectors.add(sector_mapping[symbol])
        
        # Simulate events for each represented sector
        for sector in represented_sectors:
            sector_events.append({
                "sector": sector,
                "major_events": self._get_sector_specific_events(sector, timeframe),
                "impact_on_portfolio": f"Likely affected {sector} holdings significantly"
            })
        
        return sector_events
    
    def _get_sector_specific_events(self, sector: str, timeframe: int) -> List[str]:
        """Get likely events that affected specific sectors"""
        events = {
            "Technology": [
                "AI revolution and ChatGPT breakthrough",
                "Apple iPhone sales cycles and supply chain issues",
                "Meta's metaverse pivot and layoffs",
                "Google antitrust concerns",
                "Tesla's volatile performance and Musk's Twitter acquisition"
            ],
            "Biotechnology": [
                "COVID-19 vaccine development and approvals",
                "FDA drug approvals and rejections",
                "Merger and acquisition activity in pharma",
                "Clinical trial results announcements"
            ],
            "Emerging Markets": [
                "China's COVID lockdowns and reopening",
                "Russia-Ukraine conflict impact",
                "Emerging market currency volatility",
                "Trade tensions between US and China"
            ],
            "Cryptocurrency": [
                "Bitcoin's volatile swings and institutional adoption",
                "FTX collapse and crypto winter",
                "Regulatory crackdowns on crypto",
                "Ethereum's transition to proof-of-stake"
            ]
        }
        
        return events.get(sector, ["Sector-specific developments and regulatory changes"])
    
    def _simulate_economic_events(self, timeframe: int) -> List[Dict]:
        """Simulate major economic events during the investment period"""
        return [
            {
                "event": "COVID-19 Pandemic",
                "timeline": "2020-2023",
                "impact": "Major market crash followed by unprecedented recovery",
                "educational_note": "Showed how markets can fall fast but recover even faster with policy support"
            },
            {
                "event": "Interest Rate Cycle",
                "timeline": "2020-2024",
                "impact": "Fed rates went from 0% to 5%+ affecting all asset classes",
                "educational_note": "Rising rates typically hurt growth stocks but help value investments"
            },
            {
                "event": "Inflation Surge",
                "timeline": "2021-2023", 
                "impact": "Inflation hit 9%+ causing market volatility and Fed response",
                "educational_note": "High inflation erodes returns but stocks historically outpace inflation long-term"
            },
            {
                "event": "Banking Sector Stress",
                "timeline": "2023",
                "impact": "Silicon Valley Bank and Credit Suisse failures caused market jitters",
                "educational_note": "Banking stress reminds investors why diversification across sectors matters"
            }
        ]
    
    def _is_major_event(self, event: Dict) -> bool:
        """Determine if a news event is considered major/market-moving"""
        major_event_types = ['earnings', 'merger_acquisition', 'regulatory', 'leadership']
        event_types = event.get('event_types', [])
        return any(event_type in major_event_types for event_type in event_types)
    
    def _calculate_sentiment_volatility(self, articles: List[Dict]) -> str:
        """Calculate how volatile sentiment has been for a stock"""
        try:
            from textblob import TextBlob
            sentiments = []
            
            for article in articles:
                headline = article.get('headline', '')
                if headline:
                    blob = TextBlob(headline)
                    sentiments.append(blob.sentiment.polarity)
            
            if len(sentiments) > 1:
                import statistics
                std_dev = statistics.stdev(sentiments)
                if std_dev > 0.4:
                    return "Very High"
                elif std_dev > 0.2:
                    return "High"
                elif std_dev > 0.1:
                    return "Moderate"
                else:
                    return "Low"
            return "Insufficient Data"
            
        except Exception:
            return "Unknown"
    
    async def _create_portfolio_news_summary(
        self, 
        symbols: List[str], 
        recent_news: Dict, 
        historical_analysis: Dict,
        timeframe: int
    ) -> Dict:
        """Create a comprehensive summary of news impact on portfolio"""
        return {
            "portfolio_symbols": symbols,
            "analysis_period": f"{timeframe} years",
            "recent_market_context": {
                "sentiment": recent_news.get("market_sentiment_summary", {}),
                "major_events": len(recent_news.get("portfolio_wide_events", [])),
                "news_coverage": recent_news.get("market_sentiment_summary", {}).get("total_articles", 0)
            },
            "historical_impact": {
                "major_corrections": len(historical_analysis.get("major_corrections", [])),
                "bull_periods": len(historical_analysis.get("bull_market_periods", [])),
                "sector_events": len(historical_analysis.get("sector_rotations", [])),
                "economic_events": len(historical_analysis.get("economic_events", []))
            },
            "educational_summary": f"Your portfolio of {len(symbols)} investments experienced the full spectrum of market news and events over {timeframe} years, providing real-world education about how news drives market movements."
        }
    
    async def _analyze_portfolio_news(self, stocks_picked: List[Dict], days_back: int = 7) -> Dict:
        """Analyze news sentiment for portfolio stocks"""
        try:
            # Extract symbols from stocks_picked
            symbols = []
            for stock in stocks_picked:
                if 'symbol' in stock:
                    symbols.append(stock['symbol'].upper())
                elif 'ticker' in stock:
                    symbols.append(stock['ticker'].upper())
            
            if not symbols:
                return {"error": "No symbols found in portfolio"}
            
            return await self._get_portfolio_news_analysis(symbols, days_back)
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio news: {e}")
            return {"error": str(e)}
    
    async def _get_portfolio_news_analysis(self, symbols: List[str], days_back: int = 7) -> Dict:
        """Get comprehensive news analysis for portfolio symbols"""
        try:
            async with NewsAnalysisService(self.finnhub_api_key) as news_service:
                # Get news for all symbols
                news_data = await news_service.get_market_news(symbols, days_back)
                
                portfolio_analysis = {
                    "overall_sentiment": 0.0,
                    "total_articles": 0,
                    "symbol_analysis": {},
                    "market_events": [],
                    "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0}
                }
                
                sentiment_scores = []
                
                for symbol in symbols:
                    articles = news_data.get(symbol, [])
                    
                    if articles:
                        # Analyze sentiment for this symbol
                        sentiment_result = await news_service.analyze_sentiment(articles)
                        
                        # Detect events for this symbol
                        events = await news_service.get_market_events(articles)
                        
                        portfolio_analysis["symbol_analysis"][symbol] = {
                            "sentiment": sentiment_result,
                            "events": events,
                            "article_count": len(articles)
                        }
                        
                        # Add to overall metrics
                        portfolio_analysis["total_articles"] += len(articles)
                        portfolio_analysis["market_events"].extend(events)
                        
                        if sentiment_result["average_sentiment"] != 0:
                            sentiment_scores.append(sentiment_result["average_sentiment"])
                        
                        # Update sentiment distribution
                        dist = sentiment_result.get("sentiment_distribution", {})
                        for key in ["positive", "neutral", "negative"]:
                            portfolio_analysis["sentiment_distribution"][key] += dist.get(key, 0)
                
                # Calculate overall portfolio sentiment
                if sentiment_scores:
                    portfolio_analysis["overall_sentiment"] = sum(sentiment_scores) / len(sentiment_scores)
                
                return portfolio_analysis
                
        except Exception as e:
            logger.error(f"Error getting portfolio news analysis: {e}")
            return {"error": str(e)}
    
    async def _get_price_data(self, symbol: str, days_back: int) -> Dict:
        """Get recent price movements for correlation analysis"""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(symbol)
            hist = stock.history(period=f"{days_back}d")
            
            if hist.empty:
                return {"error": f"No price data available for {symbol}"}
            
            # Calculate key metrics
            price_change = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
            volatility = hist['Close'].pct_change().std() * 100
            volume_avg = hist['Volume'].mean()
            
            return {
                'current_price': round(hist['Close'][-1], 2),
                'price_change_pct': round(price_change, 2),
                'volatility': round(volatility, 2),
                'average_volume': int(volume_avg),
                'high_period': round(hist['High'].max(), 2),
                'low_period': round(hist['Low'].min(), 2),
                'data_points': len(hist)
            }
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_news_impact(self, sentiment_analysis: Dict, events: List[Dict], price_data: Dict) -> Dict:
        """Calculate the potential impact of news on stock price"""
        if 'error' in sentiment_analysis or 'error' in price_data:
            return {"impact_score": 0, "confidence": "Low"}
        
        # Base impact score from sentiment
        sentiment_score = sentiment_analysis.get('average_sentiment', 0)
        news_volume = sentiment_analysis.get('total_articles', 0)
        
        # Weight by news volume (more articles = potentially more impact)
        volume_multiplier = min(news_volume / 10, 2.0)  # Cap at 2x
        
        # Event impact multiplier
        event_multiplier = 1.0
        if events:
            high_impact_events = ['earnings', 'merger_acquisition', 'regulatory']
            for event in events:
                if any(event_type in high_impact_events for event_type in event.get('event_types', [])):
                    event_multiplier = 1.5
                    break
        
        # Calculate final impact score
        impact_score = sentiment_score * volume_multiplier * event_multiplier
        
        # Determine confidence based on data quality
        confidence = "High" if news_volume >= 5 and price_data.get('data_points', 0) >= 5 else "Medium" if news_volume >= 2 else "Low"
        
        return {
            'impact_score': round(impact_score, 3),
            'confidence': confidence,
            'sentiment_component': round(sentiment_score, 3),
            'volume_multiplier': round(volume_multiplier, 2),
            'event_multiplier': round(event_multiplier, 2)
        }
    
    async def _generate_stock_insights(self, symbol: str, sentiment_analysis: Dict, events: List[Dict], price_data: Dict) -> str:
        """Generate AI insights for individual stock"""
        try:
            prompt = f"""
Analyze this stock data and provide brief insights for {symbol}:

SENTIMENT DATA:
- Average Sentiment: {sentiment_analysis.get('average_sentiment', 0):.3f}
- News Articles: {sentiment_analysis.get('total_articles', 0)}
- Sentiment Category: {sentiment_analysis.get('sentiment_category', 'Neutral')}

PRICE DATA:
- Recent Price Change: {price_data.get('price_change_pct', 0):.2f}%
- Current Price: ${price_data.get('current_price', 0)}

MARKET EVENTS:
{len(events)} events detected: {', '.join([e.get('headline', '')[:50] + '...' for e in events[:3]])}

Provide 2-3 sentences of actionable insights about this stock's current situation.
"""
            
            response = await self._get_groq_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating stock insights for {symbol}: {e}")
            return f"Unable to generate insights for {symbol} due to analysis error."

    def _analyze_market_movements(self, simulation_results: Dict, user_data: Dict) -> Dict:
        """
        Comprehensive analysis of market movements throughout the simulation
        """
        timeline = simulation_results.get("timeline", {}).get("portfolio", [])
        if not timeline:
            return self._estimate_market_movements(simulation_results, user_data)
        
        # Extract portfolio values and dates
        values = []
        dates = []
        for period in timeline:
            values.append(period.get("value", 0))
            dates.append(period.get("date", ""))
        
        if len(values) < 2:
            return self._estimate_market_movements(simulation_results, user_data)
        
        # Identify major market events
        movements = self._identify_market_events(values, dates, user_data)
        
        return {
            "major_crashes": movements["crashes"],
            "major_rallies": movements["rallies"],
            "volatility_periods": movements["volatile_periods"],
            "recovery_periods": movements["recoveries"],
            "overall_pattern": movements["pattern"],
            "biggest_drop": movements["biggest_drop"],
            "biggest_gain": movements["biggest_gain"],
            "total_swings": movements["total_swings"],
            "educational_insights": self._generate_movement_insights(movements, user_data)
        }
    
    def _identify_market_events(self, values: List[float], dates: List[str], user_data: Dict) -> Dict:
        """
        Identify significant market movements and categorize them
        """
        if len(values) < 3:
            return {"crashes": [], "rallies": [], "volatile_periods": [], "recoveries": [], 
                   "pattern": "stable", "biggest_drop": 0, "biggest_gain": 0, "total_swings": 0}
        
        crashes = []
        rallies = []
        volatile_periods = []
        recoveries = []
        
        # Calculate rolling peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(values) - 1):
            # Find local peaks (higher than neighbors)
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append({"index": i, "value": values[i], "date": dates[i]})
            
            # Find local troughs (lower than neighbors)
            if values[i] < values[i-1] and values[i] < values[i+1]:
                troughs.append({"index": i, "value": values[i], "date": dates[i]})
        
        # Analyze significant drops (crashes)
        for i, peak in enumerate(peaks):
            # Find the next trough after this peak
            next_troughs = [t for t in troughs if t["index"] > peak["index"]]
            if next_troughs:
                trough = min(next_troughs, key=lambda x: x["value"])
                drop_pct = ((peak["value"] - trough["value"]) / peak["value"]) * 100
                
                if drop_pct > 15:  # Significant drop
                    crash_type = self._categorize_market_event(drop_pct, peak["date"], trough["date"])
                    crashes.append({
                        "type": crash_type,
                        "drop_percent": drop_pct,
                        "start_date": peak["date"],
                        "end_date": trough["date"],
                        "start_value": peak["value"],
                        "end_value": trough["value"],
                        "likely_cause": self._guess_market_cause(peak["date"], trough["date"], drop_pct)
                    })
        
        # Analyze significant rallies
        for i, trough in enumerate(troughs):
            # Find the next peak after this trough
            next_peaks = [p for p in peaks if p["index"] > trough["index"]]
            if next_peaks:
                peak = max(next_peaks, key=lambda x: x["value"])
                gain_pct = ((peak["value"] - trough["value"]) / trough["value"]) * 100
                
                if gain_pct > 20:  # Significant rally
                    rally_type = self._categorize_rally(gain_pct, trough["date"], peak["date"])
                    rallies.append({
                        "type": rally_type,
                        "gain_percent": gain_pct,
                        "start_date": trough["date"],
                        "end_date": peak["date"],
                        "start_value": trough["value"],
                        "end_value": peak["value"],
                        "likely_cause": self._guess_rally_cause(trough["date"], peak["date"], gain_pct)
                    })
        
        # Calculate overall statistics
        biggest_drop = max([c["drop_percent"] for c in crashes], default=0)
        biggest_gain = max([r["gain_percent"] for r in rallies], default=0)
        total_swings = len(crashes) + len(rallies)
        
        # Determine overall pattern
        pattern = self._determine_market_pattern(values, crashes, rallies)
        
        return {
            "crashes": crashes,
            "rallies": rallies,
            "volatile_periods": volatile_periods,
            "recoveries": recoveries,
            "pattern": pattern,
            "biggest_drop": biggest_drop,
            "biggest_gain": biggest_gain,
            "total_swings": total_swings
        }
    
    def _categorize_market_event(self, drop_pct: float, start_date: str, end_date: str) -> str:
        """Categorize the severity of market drops"""
        if drop_pct >= 40:
            return "Market Crash"
        elif drop_pct >= 25:
            return "Major Correction"
        elif drop_pct >= 15:
            return "Market Correction"
        else:
            return "Minor Pullback"
    
    def _categorize_rally(self, gain_pct: float, start_date: str, end_date: str) -> str:
        """Categorize the strength of market rallies"""
        if gain_pct >= 60:
            return "Explosive Rally"
        elif gain_pct >= 40:
            return "Strong Bull Market"
        elif gain_pct >= 25:
            return "Market Recovery"
        else:
            return "Modest Rally"
    
    def _guess_market_cause(self, start_date: str, end_date: str, drop_pct: float) -> str:
        """Provide educational guesses for what might cause market drops"""
        causes = [
            "Economic recession fears",
            "Interest rate changes",
            "Geopolitical tensions",
            "Corporate earnings disappointments",
            "Global financial crisis",
            "Pandemic-related uncertainty",
            "Inflation concerns",
            "Banking sector stress",
            "Trade war tensions",
            "Central bank policy changes"
        ]
        
        # More severe drops get more dramatic causes
        if drop_pct >= 40:
            return f"Likely a major crisis like {causes[4]} or {causes[5]}"
        elif drop_pct >= 25:
            return f"Possibly {causes[0]} or {causes[1]}"
        else:
            return f"Could be {causes[2]} or {causes[3]}"
    
    def _guess_rally_cause(self, start_date: str, end_date: str, gain_pct: float) -> str:
        """Provide educational guesses for what might cause market rallies"""
        if gain_pct >= 50:
            return "Recovery from major crisis or breakthrough economic news"
        elif gain_pct >= 30:
            return "Strong economic growth or positive policy changes"
        else:
            return "Improved investor confidence or good earnings reports"
    
    def _determine_market_pattern(self, values: List[float], crashes: List, rallies: List) -> str:
        """Determine the overall market pattern"""
        total_crashes = len(crashes)
        total_rallies = len(rallies)
        
        if values[-1] > values[0] * 1.5:
            return "Strong Uptrend with Volatility"
        elif total_crashes > total_rallies:
            return "Volatile Bear Market"
        elif total_rallies > total_crashes:
            return "Volatile Bull Market"
        elif total_crashes + total_rallies > 3:
            return "Highly Volatile Sideways Market"
        else:
            return "Steady Growth with Normal Fluctuations"
    
    def _generate_movement_insights(self, movements: Dict, user_data: Dict) -> Dict:
        """Generate educational insights about the market movements"""
        insights = {
            "volatility_lesson": "",
            "crash_lesson": "",
            "recovery_lesson": "",
            "emotional_lesson": "",
            "time_lesson": ""
        }
        
        # Volatility lesson
        total_events = len(movements.get("crashes", [])) + len(movements.get("rallies", []))
        if total_events >= 3:
            insights["volatility_lesson"] = "Your portfolio experienced significant ups and downs - this is completely normal! Markets are like weather: sometimes stormy, sometimes sunny, but always changing."
        
        # Crash lesson
        if movements.get("biggest_drop", 0) > 25:
            insights["crash_lesson"] = f"The biggest drop of {movements['biggest_drop']:.1f}% might have felt scary, but this shows why we invest for the long term - markets recover from even severe drops."
        
        # Recovery lesson
        if movements.get("biggest_gain", 0) > 30:
            insights["recovery_lesson"] = f"The rally of {movements['biggest_gain']:.1f}% shows how markets can bounce back strongly - patience during downturns gets rewarded!"
        
        # Emotional lesson
        insights["emotional_lesson"] = "These market swings test every investor's emotions. Successful investors learn to stay calm during both crashes and rallies."
        
        # Time lesson
        timeframe = user_data.get("timeframe", 5)
        insights["time_lesson"] = f"Over your {timeframe}-year journey, you experienced the full cycle of market emotions - fear, greed, hope, and patience. This is real-world investing education!"
        
        return insights
    
    def _estimate_market_movements(self, simulation_results: Dict, user_data: Dict) -> Dict:
        """Fallback estimation when timeline data isn't available"""
        start_value = simulation_results.get("starting_value", 0)
        end_value = simulation_results.get("end_value", 0)
        timeframe = user_data.get("timeframe", 5)
        risk_score = user_data.get("risk_score", 50)
        
        # Estimate movements based on risk level and timeframe
        estimated_drops = []
        estimated_rallies = []
        
        if timeframe >= 5:
            # Estimate 1-2 major corrections over 5+ years
            if risk_score > 70:
                estimated_drops.append({
                    "type": "Major Correction",
                    "drop_percent": 30,
                    "likely_cause": "Market correction typical for aggressive portfolios"
                })
                estimated_rallies.append({
                    "type": "Strong Recovery", 
                    "gain_percent": 45,
                    "likely_cause": "Recovery rally following correction"
                })
        
        return {
            "major_crashes": estimated_drops,
            "major_rallies": estimated_rallies,
            "volatility_periods": [],
            "recovery_periods": [],
            "overall_pattern": "Estimated Volatile Growth",
            "biggest_drop": estimated_drops[0]["drop_percent"] if estimated_drops else 15,
            "biggest_gain": estimated_rallies[0]["gain_percent"] if estimated_rallies else 25,
            "total_swings": len(estimated_drops) + len(estimated_rallies),
            "educational_insights": {
                "volatility_lesson": f"With your {user_data.get('risk_label', 'moderate')} risk approach, expect market ups and downs - this is how wealth is built over time!",
                "time_lesson": "Market volatility is the price we pay for long-term growth - successful investors embrace it rather than fear it."
            }
        }
    
    def _create_comprehensive_market_prompt_with_news(
        self, stocks_picked: List[Dict], user_data: Dict[str, Any], 
        risk_score: int, risk_label: str, simulation_results: Dict[str, Any], 
        market_analysis: Dict, portfolio_news_analysis: Dict
    ) -> str:
        """Create comprehensive prompt with detailed news analysis"""
        
        goal = user_data.get("goal", "wealth building")
        lump_sum = user_data.get("lump_sum", 0)
        monthly = user_data.get("monthly", 0)
        timeframe = user_data.get("timeframe", 10)
        target_value = user_data.get("target_value", 50000)
        
        end_value = simulation_results.get("end_value", 0)
        total_contributed = lump_sum + (monthly * timeframe * 12)
        target_achieved = end_value >= target_value
        
        # Extract portfolio symbols for context
        symbols = [stock.get('symbol', '') for stock in stocks_picked]
        
        # Format recent news analysis
        recent_news_summary = ""
        if 'recent_news_context' in portfolio_news_analysis and 'error' not in portfolio_news_analysis['recent_news_context']:
            recent_context = portfolio_news_analysis['recent_news_context']
            sentiment_summary = recent_context.get('market_sentiment_summary', {})
            
            recent_news_summary = f"""
CURRENT MARKET NEWS FOR YOUR PORTFOLIO:
Overall Sentiment: {sentiment_summary.get('overall_sentiment', 0):.3f} ({'Positive' if sentiment_summary.get('overall_sentiment', 0) > 0.1 else 'Negative' if sentiment_summary.get('overall_sentiment', 0) < -0.1 else 'Neutral'})
Total Recent Articles: {sentiment_summary.get('total_articles', 0)}
Major Events: {sentiment_summary.get('total_events', 0)}

Recent News by Symbol:"""
            
            symbol_analysis = recent_context.get('symbol_specific_analysis', {})
            for symbol, analysis in symbol_analysis.items():
                sentiment = analysis.get('sentiment', {})
                recent_news_summary += f"""
• {symbol}: {analysis.get('article_count', 0)} articles, {sentiment.get('sentiment_category', 'Neutral')} sentiment
  Top Headlines: {', '.join(analysis.get('top_headlines', [])[:2])}"""
        
        # Format historical events analysis
        historical_events_summary = ""
        if 'historical_market_events' in portfolio_news_analysis:
            historical = portfolio_news_analysis['historical_market_events']
            
            historical_events_summary = f"""
MAJOR EVENTS THAT AFFECTED YOUR PORTFOLIO DURING {timeframe} YEARS:

Market Corrections Your Portfolio Survived:"""
            
            corrections = historical.get('major_corrections', [])
            for correction in corrections[:3]:
                historical_events_summary += f"""
• {correction.get('type', 'Market Event')}: {correction.get('portfolio_impact', 'N/A')} impact
  Likely caused by: {', '.join(correction.get('likely_news_themes', [])[:2])}
  Educational insight: {correction.get('educational_context', '')}"""
            
            historical_events_summary += "\n\nBull Market Rallies That Boosted Your Portfolio:"
            rallies = historical.get('bull_market_periods', [])
            for rally in rallies[:3]:
                historical_events_summary += f"""
• {rally.get('type', 'Market Rally')}: {rally.get('portfolio_impact', 'N/A')} gain
  Likely driven by: {', '.join(rally.get('likely_news_themes', [])[:2])}"""
            
            # Add sector-specific events
            sector_events = historical.get('sector_rotations', [])
            if sector_events:
                historical_events_summary += "\n\nSector-Specific Events That Affected Your Holdings:"
                for sector_event in sector_events:
                    historical_events_summary += f"""
• {sector_event.get('sector', 'Unknown Sector')}: {sector_event.get('impact_on_portfolio', '')}
  Major developments: {', '.join(sector_event.get('major_events', [])[:2])}"""
            
            # Add economic events
            economic_events = historical.get('economic_events', [])
            if economic_events:
                historical_events_summary += "\n\nMajor Economic Events During Your Investment Period:"
                for econ_event in economic_events[:3]:
                    historical_events_summary += f"""
• {econ_event.get('event', 'Economic Event')} ({econ_event.get('timeline', 'Unknown period')})
  Impact: {econ_event.get('impact', '')}
  Learning: {econ_event.get('educational_note', '')}"""
        
        return f"""
You are a patient, encouraging financial educator explaining investment results to a complete beginner. Focus heavily on explaining how real-world news events affected their specific portfolio, making complex market movements understandable and educational.

THEIR INVESTMENT JOURNEY:
Portfolio Holdings: {', '.join(symbols)}
Goal: {goal}
Target: £{target_value:,.0f}
Total Invested: £{total_contributed:,.0f}
Final Value: £{end_value:,.0f}
Result: {'🎉 GOAL ACHIEVED!' if target_achieved else '📈 PROGRESS MADE'}
Risk Level: {risk_label} ({risk_score}/100)
Investment Period: {timeframe} years

{recent_news_summary}

{historical_events_summary}

MARKET PATTERN ANALYSIS:
Overall Pattern: {market_analysis.get('overall_pattern', 'Normal Growth')}
Biggest Drop: {market_analysis.get('biggest_drop', 0):.1f}%
Biggest Rally: {market_analysis.get('biggest_gain', 0):.1f}%
Total Major Market Events: {market_analysis.get('total_swings', 0)}

REQUIRED STRUCTURE (use this exact format):

## 🎢 Your {timeframe}-Year Investment Journey Through Real Market Events

[Tell the story of their investment journey, weaving in how actual news events and market developments affected their specific portfolio holdings]

## 📰 How Major News Events Affected Your Portfolio

[Explain the most significant news events and market developments that impacted their specific holdings during the investment period. Connect real-world events to portfolio performance.]

## 📉 The Scary Headlines: When Markets Crashed

[Detail the major corrections and crashes, explaining what news events drove them and how their specific portfolio holdings were affected. Make it educational, not frightening.]

## 📈 The Recovery Stories: How Markets Bounced Back  

[Explain the rallies and recoveries, connecting them to positive news developments and showing how patience was rewarded]

## 🏢 What Happened to Your Specific Holdings

[Go symbol by symbol through their major holdings, explaining the key news events and developments that affected each one during the investment period]

## 📊 Current Market Sentiment for Your Portfolio

[Analyze the recent news sentiment for their holdings and what it means for the future, using the current news analysis]

## 🧠 News & Market Lessons from Your Experience

[Extract key educational insights about how news drives markets, emotional investing pitfalls, and why long-term thinking works]

## 🎯 Your Results: Surviving Real Market History

[Put their results in context of the actual market events they lived through, celebrating their success in staying invested through real volatility]

## 🚀 What This Market Education Means for Your Future

[Encouraging conclusion about the real-world investing education they've gained and how it prepares them for future market cycles]

WRITING STYLE:
- Tell it like a gripping story of survival through real market events
- Connect specific news events to portfolio movements
- Use specific examples from their holdings (QQQ during tech selloffs, VWO during emerging market stress, etc.)
- Make scary market events sound educational rather than terrifying
- Celebrate their resilience through actual market history
- Use analogies and relatable examples
- Include plenty of emojis and formatting for engagement
- Focus on EDUCATION about how news drives markets

This should be comprehensive and detailed - don't limit length. They've lived through real market history with their investments!
"""
    
    def _create_performance_analysis_prompt(self, portfolio_data: dict, news_analysis: Dict) -> str:
        """Create prompt for performance analysis with news context"""
        symbols = self.extract_symbols_from_portfolio(portfolio_data)
        
        news_context = ""
        if news_analysis and 'error' not in news_analysis:
            overall_sentiment = news_analysis.get('overall_sentiment', 0)
            total_articles = news_analysis.get('total_articles', 0)
            sentiment_desc = "Positive" if overall_sentiment > 0.1 else "Negative" if overall_sentiment < -0.1 else "Neutral"
            
            news_context = f"""
CURRENT NEWS SENTIMENT:
- Overall Sentiment: {sentiment_desc} ({overall_sentiment:.3f})
- Total Articles: {total_articles}
- Key Events: {len(news_analysis.get('market_events', []))} detected
"""
        
        return f"""
Analyze this portfolio's performance with current market context:

PORTFOLIO HOLDINGS: {', '.join(symbols)}

{news_context}

Provide an educational analysis covering:
1. How current news sentiment affects these holdings
2. What the market sentiment suggests about near-term prospects
3. Educational insights about how news impacts stock prices
4. Actionable recommendations based on sentiment analysis

Keep it beginner-friendly and educational. Focus on teaching, not selling.
"""
    
    def _create_risk_analysis_prompt(self, portfolio_data: dict, news_analysis: Dict) -> str:
        """Create prompt for risk analysis with market sentiment"""
        symbols = self.extract_symbols_from_portfolio(portfolio_data)
        
        risk_context = ""
        if news_analysis and 'error' not in news_analysis:
            events = news_analysis.get('market_events', [])
            sentiment = news_analysis.get('overall_sentiment', 0)
            
            # Identify risk factors from news
            risk_events = [event for event in events if any(risk_type in event.get('event_types', []) 
                          for risk_type in ['regulatory', 'legal', 'earnings'])]
            
            risk_context = f"""
CURRENT RISK INDICATORS FROM NEWS:
- Portfolio Sentiment: {sentiment:.3f} ({"Higher risk" if abs(sentiment) > 0.3 else "Normal risk"})
- Risk Events Detected: {len(risk_events)}
- Total Market Events: {len(events)}
"""
        
        return f"""
Analyze the risk profile of this portfolio considering current market sentiment:

PORTFOLIO: {', '.join(symbols)}

{risk_context}

Provide educational risk analysis covering:
1. How current news sentiment affects portfolio risk
2. What market events mean for volatility expectations
3. Risk management lessons from current market conditions
4. Educational insights about news-driven market risks

Explain in beginner terms how news and sentiment create investment risk and opportunity.
"""
    
    def _create_changes_explanation_prompt(self, portfolio_data: dict, previous_data: dict, news_analysis: Dict) -> str:
        """Create prompt for explaining portfolio changes with news context"""
        current_symbols = self.extract_symbols_from_portfolio(portfolio_data)
        previous_symbols = self.extract_symbols_from_portfolio(previous_data) if previous_data else []
        
        changes_context = ""
        if news_analysis and 'error' not in news_analysis:
            symbol_analysis = news_analysis.get('symbol_analysis', {})
            
            # Identify which stocks have significant news
            news_heavy_stocks = [symbol for symbol, data in symbol_analysis.items() 
                               if data.get('article_count', 0) > 5]
            
            changes_context = f"""
RECENT NEWS THAT MIGHT EXPLAIN CHANGES:
- Stocks with Heavy News Coverage: {', '.join(news_heavy_stocks) if news_heavy_stocks else 'None'}
- Overall Market Sentiment: {news_analysis.get('overall_sentiment', 0):.3f}
- Major Events: {len(news_analysis.get('market_events', []))}
"""
        
        return f"""
Explain what might have caused changes in this portfolio:

CURRENT HOLDINGS: {', '.join(current_symbols)}
PREVIOUS HOLDINGS: {', '.join(previous_symbols)}

{changes_context}

Provide educational explanation covering:
1. How recent news might explain portfolio changes
2. What market events could have influenced decisions
3. Educational insights about how news drives portfolio adjustments
4. Lessons about reacting (or not reacting) to market news

Focus on teaching how news and market sentiment influence investment decisions.
"""
    
    async def _get_groq_response(self, prompt: str) -> str:
        """Get response from GROQ API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert financial educator who excels at explaining complex market movements and news sentiment to beginners in an engaging, educational way. You have deep knowledge of how real-world news events affect specific stocks and portfolios."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 4000,  # Increased for detailed news analysis
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.groq_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90)  # Increased timeout for detailed analysis
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result["choices"][0]["message"]["content"].strip()
                        
                        if ai_response:
                            return ai_response
                        else:
                            raise Exception("Empty response from GROQ")
                    else:
                        error_text = await response.text()
                        raise Exception(f"GROQ API returned status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"GROQ request failed: {str(e)}")
            raise Exception(f"Failed to get AI response: {str(e)}")
    
    def _format_ai_response(self, response: str) -> str:
        """Format AI response with better structure and readability"""
        # Clean up the response
        response = response.strip()
        
        # Ensure proper spacing around headings
        response = response.replace("##", "\n\n##")
        
        # Add spacing around bullet points
        lines = response.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
                
            # Add extra space before headings
            if line.startswith('##'):
                if formatted_lines and formatted_lines[-1] != '':
                    formatted_lines.append('')
                formatted_lines.append(line)
                formatted_lines.append('')
            # Handle bullet points
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # Join and clean up multiple empty lines
        formatted_response = '\n'.join(formatted_lines)
        while '\n\n\n' in formatted_response:
            formatted_response = formatted_response.replace('\n\n\n', '\n\n')
        
        return formatted_response.strip()
    
    def _get_formatted_fallback_summary_with_movements(
        self, user_data: Dict, simulation_results: Dict, 
        stocks_picked: List[Dict], risk_label: str
    ) -> str:
        """Enhanced fallback that includes market movement education and news context"""
        
        lump_sum = user_data.get("lump_sum", 0)
        monthly = user_data.get("monthly", 0)
        timeframe = user_data.get("timeframe", 10)
        total_contributed = lump_sum + (monthly * timeframe * 12)
        end_value = simulation_results.get("end_value", 0)
        target_value = user_data.get("target_value", 50000)
        target_achieved = end_value >= target_value
        
        growth = end_value - total_contributed
        return_pct = (growth / total_contributed * 100) if total_contributed > 0 else 0
        
        symbols = [stock.get('symbol', '') for stock in stocks_picked]
        
        market_education = f"""

## 🎢 Your Portfolio's Journey Through Real Market Events

With your **{risk_label.lower()}** approach and holdings in {', '.join(symbols[:5])}, your portfolio lived through some of the most dramatic market events in recent history!

## 📰 Major News Events That Affected Your Holdings

**What probably impacted your portfolio:**

**Technology Holdings Impact** (if you held QQQ, VGT, ARKK):
• **AI Revolution (2023-2024)**: ChatGPT and AI boom drove tech stocks wild
• **Interest Rate Shock (2022)**: Fed rate hikes crushed growth stocks
• **Big Tech Earnings**: Apple, Microsoft, Google results moved your holdings

**Emerging Markets Impact** (if you held VWO):
• **China COVID Policies**: Lockdowns and reopening affected emerging markets
• **Russia-Ukraine Conflict**: Geopolitical tensions hit international investments
• **Dollar Strength**: Strong USD hurt emerging market returns

**Cryptocurrency Exposure** (if you held COIN, BITO):
• **Crypto Winter (2022)**: Bitcoin crashed from $69k to $15k
• **FTX Collapse**: Sam Bankman-Fried scandal rocked crypto markets
• **Regulatory Uncertainty**: SEC actions affected crypto investments

## 📉 The Scary Headlines Your Portfolio Survived

**Market Crashes During Your Investment:**
• **March 2020**: "MARKET CRASHES 35% - WORST SINCE 1929!"
• **2022 Bear Market**: "STOCKS ENTER BEAR MARKET AS INFLATION SOARS!"
• **Banking Crisis 2023**: "SILICON VALLEY BANK COLLAPSES!"

**Why your portfolio survived these scary headlines:**
• Diversification across different asset types protected you
• Long-term investing meant you didn't panic sell at the bottom
• Market recoveries always followed the crashes

## 📈 The Recovery Headlines That Boosted Your Returns

**Positive News That Lifted Your Portfolio:**
• **Vaccine Rollout (2021)**: "ECONOMY REOPENS - STOCKS SOAR!"
• **AI Boom (2023)**: "ARTIFICIAL INTELLIGENCE REVOLUTION BEGINS!"
• **Inflation Cooling (2024)**: "INFLATION FALLS - FED PAUSE EXPECTED!"

## 🧠 What These Real Events Taught You

**Key Lessons from Living Through Market History:**
• **Headlines are emotional, markets are mathematical**: Scary news creates opportunities
• **Time heals market wounds**: Every crash was followed by recovery
• **Diversification works**: Different holdings reacted differently to events
• **Staying invested pays**: Panic selling would have locked in losses
• **News cycles are short, investments are long**: Daily headlines don't determine decade outcomes

## 🎯 Your Results: Surviving Real Market Chaos

Despite all the dramatic news headlines and market chaos, you ended up with **£{end_value:,.0f}** from **£{total_contributed:,.0f}** invested!

**What this means:**
• You survived a global pandemic market crash
• You weathered the worst inflation in 40 years  
• You lived through crypto winter and banking crises
• You experienced both AI boom and tech bust cycles
• You proved you can handle real market volatility

## 🚀 What This Market Education Means for Your Future

**You're now a battle-tested investor who has:**
• Experienced how news drives short-term market emotions
• Learned that market recoveries reward patient investors
• Seen how diversification protects during different crisis types
• Understood that time in markets beats timing markets
• Gained real-world experience that textbooks can't teach

You've earned your investing stripes through real market history! 💪"""
        
        success_message = "🎉 Congratulations - Goal Achieved!" if target_achieved else "📈 Solid Progress Made"
        
        return f"""## {success_message}

Your investment journey shows the power of patient, consistent investing through real-world market chaos! You put in **£{total_contributed:,.0f}** over {timeframe} years, and it grew to **£{end_value:,.0f}**.

## 📊 Your Investment Results

• **Total Growth**: £{growth:,.0f} ({return_pct:+.1f}%)
• **Strategy**: {risk_label} approach with {', '.join(symbols)}
• **Time Horizon**: {timeframe} years of real market history
• **Target**: {'✅ Achieved' if target_achieved else f'£{abs(target_value - end_value):,.0f} short'}

{market_education}

This wasn't just a simulation - this was real investing education through one of the most volatile periods in market history! 🌟"""