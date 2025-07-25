from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
import aiohttp
import os
from services.news_analysis import NewsAnalysisService

# Set up logging
logger = logging.getLogger(__name__)

class AIAnalysisService:
    """
    Enhanced AI-powered portfolio analysis service using GROQ API
    Now includes news sentiment analysis and market event detection
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
        Generate AI summary with enhanced market movement explanations
        Now includes news sentiment analysis
        """
        try:
            # Analyze market movements from timeline data
            market_analysis = self._analyze_market_movements(simulation_results, user_data)
            
            # Get news sentiment analysis for the portfolio stocks
            news_analysis = await self._analyze_portfolio_news(stocks_picked)
            
            # Generate comprehensive prompt with market movement insights and news sentiment
            prompt = self._create_educational_market_prompt_with_news(
                stocks_picked, user_data, risk_score, risk_label, 
                simulation_results, market_analysis, news_analysis
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

    # ... [Keep all your existing market movement analysis methods] ...
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
    
    def _create_educational_market_prompt_with_news(
        self, stocks_picked: List[Dict], user_data: Dict[str, Any], 
        risk_score: int, risk_label: str, simulation_results: Dict[str, Any], 
        market_analysis: Dict, news_analysis: Dict
    ) -> str:
        """Create comprehensive prompt focusing on market movement education with news context"""
        
        goal = user_data.get("goal", "wealth building")
        lump_sum = user_data.get("lump_sum", 0)
        monthly = user_data.get("monthly", 0)
        timeframe = user_data.get("timeframe", 10)
        target_value = user_data.get("target_value", 50000)
        
        end_value = simulation_results.get("end_value", 0)
        total_contributed = lump_sum + (monthly * timeframe * 12)
        target_achieved = end_value >= target_value
        
        # Format market events for the prompt
        crashes_summary = ""
        if market_analysis["major_crashes"]:
            crashes_summary = "\n".join([
                f"• {crash['type']}: {crash['drop_percent']:.1f}% drop - {crash['likely_cause']}"
                for crash in market_analysis["major_crashes"][:3]
            ])
        
        rallies_summary = ""
        if market_analysis["major_rallies"]:
            rallies_summary = "\n".join([
                f"• {rally['type']}: {rally['gain_percent']:.1f}% gain - {rally['likely_cause']}"
                for rally in market_analysis["major_rallies"][:3]
            ])
        
        # Format news analysis
        news_summary = ""
        if news_analysis and 'error' not in news_analysis:
            overall_sentiment = news_analysis.get('overall_sentiment', 0)
            total_articles = news_analysis.get('total_articles', 0)
            
            sentiment_desc = "Positive" if overall_sentiment > 0.1 else "Negative" if overall_sentiment < -0.1 else "Neutral"
            
            news_summary = f"""
CURRENT NEWS SENTIMENT ANALYSIS:
- Overall Portfolio Sentiment: {sentiment_desc} ({overall_sentiment:.3f})
- Total News Articles Analyzed: {total_articles}
- Market Events Detected: {len(news_analysis.get('market_events', []))}

This gives context about how the market currently views your portfolio stocks.
"""
        
        return f"""
You are a patient, encouraging financial educator explaining investment results to a complete beginner. Focus heavily on explaining market movements in simple, educational terms, and now also incorporate current news sentiment.

THEIR INVESTMENT JOURNEY:
- Goal: {goal}
- Target: £{target_value:,.0f}
- Total Invested: £{total_contributed:,.0f}
- Final Value: £{end_value:,.0f}
- Result: {'🎉 GOAL ACHIEVED!' if target_achieved else '📈 PROGRESS MADE'}
- Risk Level: {risk_label} ({risk_score}/100)

MAJOR MARKET EVENTS THEY EXPERIENCED:
Market Pattern: {market_analysis['overall_pattern']}
Biggest Drop: {market_analysis['biggest_drop']:.1f}%
Biggest Rally: {market_analysis['biggest_gain']:.1f}%
Total Major Events: {market_analysis['total_swings']}

MARKET CRASHES/CORRECTIONS:
{crashes_summary if crashes_summary else "• No major crashes detected"}

MARKET RALLIES/RECOVERIES:
{rallies_summary if rallies_summary else "• Steady growth without dramatic rallies"}

{news_summary}

EDUCATIONAL INSIGHTS:
{market_analysis['educational_insights']}

REQUIRED STRUCTURE (use this exact format):

## 🎢 Your Investment Roller Coaster Journey

[Explain their overall experience in simple terms, like a story. Include current news sentiment context.]

## 📉 When Markets Went Down (The Scary Parts)

[Explain each major drop in beginner terms - what happened, why it's normal, and what it teaches us]

## 📈 When Markets Bounced Back (The Exciting Parts)

[Explain the recoveries and rallies - how markets heal themselves and reward patient investors]

## 📰 What Current News Tells Us

[Incorporate the news sentiment analysis - explain what current market sentiment means for their portfolio]

## 🧠 What These Market Swings Teach Us

[Extract 3-4 key investing lessons from their specific market experience, enhanced with news insights]

## 🎯 Your Results in Perspective

[Explain their final results and what the market journey means for their success]

## 🚀 What This Means for Your Future

[Encouraging conclusion about their investing education and next steps, considering current market sentiment]

WRITING STYLE:
- Explain like you're talking to a curious friend over coffee
- Use analogies (roller coasters, weather, sports, etc.)
- Make market crashes sound normal and educational, not scary
- Celebrate their patience through volatility
- Use emojis and formatting for engagement
- Focus on EDUCATION, not sales
- Integrate news sentiment naturally into the narrative

Be comprehensive - don't limit length. This is their investment education!
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
                        "content": "You are an expert financial educator who excels at explaining complex market movements and news sentiment to beginners in an engaging, educational way."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 3000,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.groq_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
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
        
        market_education = ""
        if risk_label.lower() in ["aggressive", "moderate aggressive"]:
            market_education = f"""

## 🎢 Understanding Your Market Journey

With your **{risk_label.lower()}** approach, your portfolio likely experienced some dramatic ups and downs - and that's completely normal! Here's what probably happened:

## 📉 The Downs (Market Corrections)

**What you might have experienced:**
• **Market corrections** (10-20% drops): These happen every 1-2 years
• **Bear markets** (20%+ drops): These occur every 3-5 years  
• **Flash crashes**: Sudden, sharp drops that recover quickly

**Why they happen:**
• Economic uncertainty or recession fears
• Interest rate changes by central banks
• Global events affecting investor confidence
• Company earnings disappointments
• General market psychology and emotions

## 📈 The Ups (Market Recoveries)

**What followed the downs:**
• **Relief rallies**: Quick bounces after oversold conditions
• **Bull market runs**: Extended periods of growth
• **Recovery phases**: Gradual climbing back to new highs

**Why markets recover:**
• Companies adapt and improve over time
• Economic growth continues long-term
• Innovation drives new opportunities
• Central banks provide support during crises

## 📰 News and Your Portfolio

**How news affects your investments:**
• Daily headlines create short-term volatility
• Positive news can drive temporary rallies
• Negative news often causes overreactions
• Long-term fundamentals matter more than daily news cycles

## 🧠 Key Lessons from Market Volatility

• **Volatility is the price of admission**: Higher returns come with bigger swings
• **Time heals market wounds**: Patience during downturns gets rewarded
• **Stay the course**: Panic selling locks in losses at the worst times
• **Markets climb a wall of worry**: Good things happen despite scary headlines
• **News creates noise, not necessarily direction**: Headlines are emotional, markets are mathematical

## 🎯 Why Your Strategy Worked

Despite all the market drama and daily news cycles, you ended up with **£{end_value:,.0f}** from **£{total_contributed:,.0f}** invested - that's the power of staying invested through the ups and downs!

The key: You didn't let short-term market movements or scary headlines derail your long-term plan. That's exactly what successful investors do! 🚀"""
        
        success_message = "🎉 Congratulations - Goal Achieved!" if target_achieved else "📈 Solid Progress Made"
        
        return f"""## {success_message}

Your investment journey shows the power of patient, consistent investing! You put in **£{total_contributed:,.0f}** over {timeframe} years, and it grew to **£{end_value:,.0f}**.

## 📊 Your Investment Results

• **Total Growth**: £{growth:,.0f} ({return_pct:+.1f}%)
• **Strategy**: {risk_label} approach  
• **Time Horizon**: {timeframe} years
• **Target**: {'✅ Achieved' if target_achieved else f'£{abs(target_value - end_value):,.0f} short'}

{market_education}

## 🌟 What This Means for Your Future

You've experienced real-world investing - complete with market ups and downs, news cycles, and sentiment swings. This education is invaluable for your future financial decisions!

**Key takeaways:**
• Market volatility is normal and manageable
• Consistent investing builds wealth over time
• Staying patient during downturns pays off
• Your strategy can handle market stress
• News sentiment creates opportunities, not just risks

You're now a more experienced investor with real market battle scars - wear them proudly! 💪"""