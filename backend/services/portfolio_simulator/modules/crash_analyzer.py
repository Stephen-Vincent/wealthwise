"""
Market Crash Analyzer Module

This module detects and analyzes market crashes in portfolio simulations,
integrates with news analysis services, and provides educational insights
about market volatility and recovery patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

class MarketCrashAnalyzer:
    """
    Analyzes market crashes and provides educational context with news integration.
    
    Features:
    - Detects significant market downturns (30%+ drops)
    - Integrates news analysis for crash periods
    - Provides educational insights about market recovery
    - Calculates portfolio-specific impact
    """
    
    def __init__(self, crash_threshold: float = 0.30):
        """
        Initialize the market crash analyzer.
        
        Args:
            crash_threshold: Minimum decline percentage to classify as a crash
        """
        self.crash_threshold = crash_threshold
        self.news_service = None
        logger.info(f"📉 MarketCrashAnalyzer initialized with {crash_threshold*100}% threshold")
    
    async def add_crash_analysis(self, simulation_results: Dict[str, Any], 
                                stock_data: pd.DataFrame, 
                                stocks_picked: List[Dict]) -> Dict[str, Any]:
        """
        Add comprehensive crash analysis to simulation results.
        
        Args:
            simulation_results: Existing simulation results
            stock_data: Historical stock data used in simulation
            stocks_picked: List of stocks in the portfolio
            
        Returns:
            Enhanced simulation results with crash analysis
        """
        
        try:
            logger.info("🔍 Adding market crash analysis to simulation results")
            
            # Detect crashes in the historical data
            crashes = self.detect_market_crashes(stock_data)
            
            # Get news analysis for each significant crash
            crash_analyses = []
            for crash in crashes:
                crash_date = crash['crash_date']
                
                # Only analyze crashes from the last 15 years (better news availability)
                if crash_date.year >= 2010:
                    logger.info(f"📰 Analyzing crash on {crash_date.strftime('%Y-%m-%d')}")
                    news_analysis = await self.get_news_for_crash_period(
                        crash_date, stocks_picked
                    )
                    
                    crash_analysis = {
                        'crash_date': crash['crash_date'].isoformat(),
                        'severity': f"{crash['severity']:.1%}",
                        'peak_date': crash['peak_date'].isoformat(),
                        'recovery_time_days': crash.get('recovery_time_days'),
                        'recovery_message': self.get_recovery_message(crash.get('recovery_time_days')),
                        'news_analysis': news_analysis,
                        'educational_insight': self.generate_crash_insight(crash),
                        'user_friendly_explanation': self.generate_user_friendly_crash_explanation(crash, news_analysis)
                    }
                    
                    crash_analyses.append(crash_analysis)
                else:
                    # For older crashes, use basic analysis without news
                    crash_analysis = {
                        'crash_date': crash['crash_date'].isoformat(),
                        'severity': f"{crash['severity']:.1%}",
                        'peak_date': crash['peak_date'].isoformat(),
                        'recovery_time_days': crash.get('recovery_time_days'),
                        'recovery_message': self.get_recovery_message(crash.get('recovery_time_days')),
                        'educational_insight': self.generate_crash_insight(crash),
                        'historical_note': f"This crash occurred in {crash['crash_date'].year}, before detailed news analysis was available."
                    }
                    
                    crash_analyses.append(crash_analysis)
            
            # Add to simulation results
            simulation_results['market_crash_analysis'] = {
                'crashes_detected': len(crashes),
                'crashes_with_news_analysis': len([c for c in crash_analyses if 'news_analysis' in c]),
                'crash_details': crash_analyses,
                'overall_message': self.generate_overall_crash_message(crashes),
                'educational_summary': self.generate_crash_education_summary(crashes),
                'key_insights': self.generate_key_crash_insights(crashes, crash_analyses)
            }
            
            logger.info(f"✅ Added analysis for {len(crash_analyses)} market crashes ({len([c for c in crash_analyses if 'news_analysis' in c])} with news)")
            return simulation_results
            
        except Exception as e:
            logger.error(f"❌ Error adding crash analysis: {e}")
            # Return original results if analysis fails
            return simulation_results
    
    def detect_market_crashes(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect significant market crashes (drops of 30% or more) in the portfolio timeline.
        
        Args:
            data: DataFrame with stock price data
        
        Returns:
            List of detected crashes with dates and severity
        """
        
        try:
            logger.info(f"🔍 Detecting market crashes with threshold: {self.crash_threshold*100}%")
            
            # Calculate portfolio performance (assuming equal weights for simplicity)
            if data.empty:
                return []
            
            # Normalize to starting values
            normalized_data = data.div(data.iloc[0])
            
            # Calculate portfolio value (equal weights)
            portfolio_performance = normalized_data.mean(axis=1)
            
            crashes = []
            
            # Look for significant drops
            for i in range(1, len(portfolio_performance)):
                current_date = portfolio_performance.index[i]
                current_value = portfolio_performance.iloc[i]
                
                # Look back over various periods to find peak
                lookback_periods = [30, 60, 90, 180, 252]  # Days to look back
                
                for lookback in lookback_periods:
                    start_idx = max(0, i - lookback)
                    period_data = portfolio_performance.iloc[start_idx:i+1]
                    
                    if len(period_data) < 2:
                        continue
                    
                    peak_value = period_data.max()
                    peak_date = period_data.idxmax()
                    
                    # Calculate drop from peak
                    drop = (peak_value - current_value) / peak_value
                    
                    if drop >= self.crash_threshold:
                        # Check if we already detected this crash
                        existing_crash = None
                        for crash in crashes:
                            if abs((current_date - crash['crash_date']).days) < 30:
                                existing_crash = crash
                                break
                        
                        if existing_crash:
                            # Update if this is a bigger drop
                            if drop > existing_crash['severity']:
                                existing_crash.update({
                                    'severity': drop,
                                    'crash_date': current_date,
                                    'peak_date': peak_date,
                                    'peak_value': peak_value,
                                    'crash_value': current_value,
                                    'lookback_period': lookback
                                })
                        else:
                            # New crash detected
                            crashes.append({
                                'crash_date': current_date,
                                'peak_date': peak_date,
                                'severity': drop,
                                'peak_value': peak_value,
                                'crash_value': current_value,
                                'lookback_period': lookback,
                                'recovery_date': None,
                                'recovery_time_days': None
                            })
                            
                            logger.warning(f"💥 Market crash detected: {drop:.1%} drop on {current_date.strftime('%Y-%m-%d')}")
            
            # Calculate recovery times
            for crash in crashes:
                crash_date = crash['crash_date']
                peak_value = crash['peak_value']
                
                # Look for recovery (when portfolio returns to 95% of peak)
                recovery_threshold = peak_value * 0.95
                
                post_crash_data = portfolio_performance[portfolio_performance.index > crash_date]
                recovery_points = post_crash_data[post_crash_data >= recovery_threshold]
                
                if not recovery_points.empty:
                    recovery_date = recovery_points.index[0]
                    recovery_time = (recovery_date - crash_date).days
                    crash['recovery_date'] = recovery_date
                    crash['recovery_time_days'] = recovery_time
                    
                    logger.info(f"📈 Recovery detected: {recovery_time} days after crash")
            
            logger.info(f"✅ Found {len(crashes)} significant market crashes")
            return crashes
            
        except Exception as e:
            logger.error(f"❌ Error detecting market crashes: {e}")
            return []
    
    async def get_news_for_crash_period(self, crash_date: datetime, 
                                       stocks_picked: List[Dict]) -> Dict[str, Any]:
        """
        Get news analysis for market crash period.
        
        Args:
            crash_date: Date of the market crash
            stocks_picked: List of stocks in the portfolio
        
        Returns:
            Analysis of news around the crash date with AI explanations
        """
        
        try:
            logger.info(f"📰 Getting news analysis for crash on {crash_date.strftime('%Y-%m-%d')}")
            
            # Check if we have Finnhub API key for news analysis
            finnhub_key = os.getenv("FINNHUB_API_KEY")
            
            if not finnhub_key:
                logger.warning("⚠️ No Finnhub API key found, using fallback analysis")
                return self.get_fallback_crash_explanation(crash_date)
            
            # Try to import and use news analysis service
            try:
                from services.news_analysis import NewsAnalysisService
                
                async with NewsAnalysisService(finnhub_key) as news_service:
                    # Get general market news around crash date
                    general_news = await news_service.get_general_market_news(
                        category='general', limit=20
                    )
                    
                    # Filter news around crash date (±7 days)
                    start_date = crash_date - timedelta(days=7)
                    end_date = crash_date + timedelta(days=7)
                    
                    relevant_news = []
                    for article in general_news:
                        article_date = datetime.fromtimestamp(article.get('datetime', 0))
                        if start_date <= article_date <= end_date:
                            relevant_news.append(article)
                    
                    # Get portfolio-specific news if available
                    portfolio_symbols = [stock.get('symbol', '') for stock in stocks_picked]
                    portfolio_news = {}
                    
                    # Only get portfolio news for actual stock symbols (not ETFs)
                    stock_symbols = [s for s in portfolio_symbols 
                                   if not s.startswith('V') and s not in ['BND', 'VEA', 'VTEB', 'VWO', 'VNQ', 'VGT', 'VUG', 'ARKK']]
                    
                    if stock_symbols:
                        portfolio_news_data = await news_service.get_market_news(stock_symbols, days_back=14)
                        
                        # Filter to crash period
                        for symbol, articles in portfolio_news_data.items():
                            crash_period_articles = []
                            for article in articles:
                                article_date = datetime.fromtimestamp(article.get('datetime', 0))
                                if start_date <= article_date <= end_date:
                                    crash_period_articles.append(article)
                            
                            if crash_period_articles:
                                portfolio_news[symbol] = crash_period_articles
                    
                    # Analyze sentiment
                    all_crash_news = relevant_news + [article for articles in portfolio_news.values() for article in articles]
                    sentiment_analysis = await news_service.analyze_sentiment(all_crash_news)
                    
                    # Detect market events
                    market_events = await news_service.get_market_events(all_crash_news)
                    
                    # Generate AI explanation
                    ai_explanation = await self.generate_crash_explanation_with_news(
                        crash_date, all_crash_news, market_events, sentiment_analysis
                    )
                    
                    return {
                        "crash_date": crash_date.isoformat(),
                        "news_summary": {
                            "general_market_articles": len(relevant_news),
                            "portfolio_specific_articles": sum(len(articles) for articles in portfolio_news.values()),
                            "total_articles_analyzed": len(all_crash_news)
                        },
                        "sentiment_analysis": sentiment_analysis,
                        "market_events": market_events[:5],  # Top 5 events
                        "ai_explanation": ai_explanation,
                        "portfolio_impact": self.analyze_portfolio_crash_impact(portfolio_news, stocks_picked),
                        "key_headlines": [article.get('headline', '')[:100] + "..." for article in all_crash_news[:3]]
                    }
                    
            except ImportError:
                logger.warning("⚠️ News analysis service not available, using fallback")
                return self.get_fallback_crash_explanation(crash_date)
                
        except Exception as e:
            logger.error(f"❌ Error getting news for crash: {e}")
            return self.get_fallback_crash_explanation(crash_date)
    
    async def generate_crash_explanation_with_news(self, crash_date: datetime, 
                                                  news_articles: List[Dict], 
                                                  market_events: List[Dict],
                                                  sentiment_analysis: Dict) -> str:
        """
        Generate AI explanation of market crash using news data.
        """
        
        try:
            # Prepare news context for AI
            news_context = ""
            if news_articles:
                news_context = "KEY NEWS DURING CRASH PERIOD:\n"
                for i, article in enumerate(news_articles[:8]):  # Top 8 articles
                    headline = article.get('headline', 'No headline')
                    summary = article.get('summary', 'No summary')
                    news_context += f"{i+1}. {headline}\n   Summary: {summary[:150]}...\n\n"
            
            events_context = ""
            if market_events:
                events_context = "MAJOR MARKET EVENTS DETECTED:\n"
                for event in market_events[:5]:
                    events_context += f"• {event.get('headline', 'Unknown event')}\n"
                    events_context += f"  Event types: {', '.join(event.get('event_types', []))}\n\n"
            
            sentiment_context = f"""
MARKET SENTIMENT ANALYSIS:
• Overall sentiment: {sentiment_analysis.get('sentiment_category', 'Unknown')}
• Sentiment score: {sentiment_analysis.get('average_sentiment', 0):.2f} (-1 to +1 scale)
• Articles analyzed: {sentiment_analysis.get('total_articles', 0)}
• Sentiment strength: {sentiment_analysis.get('sentiment_strength', 'Unknown')}
"""

            # Try to use AI service for explanation
            try:
                from services.ai_analysis import AIAnalysisService
                ai_service = AIAnalysisService()
                
                analysis_prompt = f"""
Based on the following news and market data from around {crash_date.strftime('%B %d, %Y')}, 
explain what caused this significant market crash in simple, educational terms.

{news_context}

{events_context}

{sentiment_context}

Please provide:
1. A clear, beginner-friendly explanation of what caused the market crash
2. The main factors that contributed to investor panic
3. How this type of event typically affects different types of investments
4. What long-term investors should understand about such market events
5. Key lessons for managing through market volatility

Keep the explanation educational and reassuring, focusing on helping users understand 
that market crashes are normal parts of investing and that patient investors typically recover.
"""

                explanation = await ai_service._get_groq_response(analysis_prompt)
                return explanation
                
            except Exception as ai_error:
                logger.warning(f"⚠️ AI explanation failed: {ai_error}")
                return self._generate_basic_crash_explanation(crash_date, sentiment_analysis)
                
        except Exception as e:
            logger.error(f"❌ Error generating crash explanation: {e}")
            return f"Market crash occurred on {crash_date.strftime('%B %d, %Y')}. This was likely due to various economic factors and market conditions during this period. Historically, markets have recovered from such downturns over time."
    
    def _generate_basic_crash_explanation(self, crash_date: datetime, 
                                        sentiment_analysis: Dict) -> str:
        """Generate basic crash explanation without AI service."""
        
        sentiment = sentiment_analysis.get('sentiment_category', 'Very Negative')
        
        return f"""
**Market Crash - {crash_date.strftime('%B %Y')}**

During this period, markets experienced significant volatility with {sentiment.lower()} investor sentiment. 
Market crashes typically result from a combination of economic uncertainty, investor fear, and external events 
that shake confidence in the financial system.

**Understanding Market Crashes:**
Market corrections of 20% or more happen roughly every 3-4 years, while severe crashes (40%+) occur about 
once per decade. These events, while unsettling, are normal parts of market cycles.

**Key Takeaways:**
1. Market volatility is the price investors pay for higher long-term returns
2. Time in the market is more important than timing the market
3. Regular investing during downturns can actually improve long-term outcomes
4. Staying disciplined during crashes is crucial for investment success

Remember: Every major market crash in history has eventually been followed by recovery and new market highs.
"""
    
    def get_fallback_crash_explanation(self, crash_date: datetime) -> Dict[str, Any]:
        """
        Fallback crash explanation when news service is unavailable.
        """
        
        # Known historical crashes for context
        historical_crashes = {
            "2020-03": {
                "name": "COVID-19 Pandemic Crash",
                "cause": "Global pandemic fears and economic lockdowns",
                "recovery_info": "Markets recovered in about 5 months with government stimulus"
            },
            "2008-09": {
                "name": "Financial Crisis",
                "cause": "Housing bubble burst and banking system failures",
                "recovery_info": "Markets took about 2-3 years to fully recover"
            },
            "2000-03": {
                "name": "Dot-com Bubble Burst",
                "cause": "Technology stock overvaluation and speculative bubble",
                "recovery_info": "Tech-heavy markets took several years to recover"
            },
            "1987-10": {
                "name": "Black Monday",
                "cause": "Program trading and market psychology panic",
                "recovery_info": "Markets recovered within 2 years"
            }
        }
        
        crash_key = crash_date.strftime('%Y-%m')
        
        # Try to match historical crash
        historical_info = None
        for period, info in historical_crashes.items():
            if crash_key.startswith(period[:4]) and abs(int(crash_key.split('-')[1]) - int(period.split('-')[1])) <= 2:
                historical_info = info
                break
        
        if historical_info:
            ai_explanation = f"""
**{historical_info['name']} - {crash_date.strftime('%B %Y')}**

This market crash was primarily caused by {historical_info['cause']}. 

**What Happened:**
During this period, investors became concerned about fundamental economic conditions, leading to widespread selling. This type of market event, while scary at the time, is a normal part of investing cycles.

**Recovery:**
{historical_info['recovery_info']}. This demonstrates that while market crashes can be severe, patient long-term investors who stayed invested were ultimately rewarded.

**Key Lessons:**
1. Market crashes are temporary, but recoveries are historically permanent
2. Diversified portfolios tend to recover more steadily than individual stocks
3. Continuing to invest during downturns can improve long-term returns
4. Emotional decisions during crashes often hurt long-term performance

Remember: Every major crash in history has been followed by recovery and new market highs.
"""
        else:
            ai_explanation = f"""
**Market Crash - {crash_date.strftime('%B %Y')}**

A significant market decline occurred during this period. While specific news details aren't available, market crashes typically result from a combination of economic uncertainty, investor sentiment, and external events.

**Understanding Market Crashes:**
Market corrections of 20% or more happen roughly every 3-4 years, while severe crashes (40%+) occur about once per decade. These events, while unsettling, are normal parts of market cycles.

**Historical Perspective:**
- Every major market crash in history has eventually been followed by recovery
- Patient investors who continued their investment plans typically saw strong long-term returns
- Diversified portfolios tend to be more resilient during market stress

**Key Takeaways:**
1. Market volatility is the price investors pay for higher long-term returns
2. Time in the market is more important than timing the market
3. Regular investing during downturns can actually improve long-term outcomes
4. Staying disciplined during crashes is crucial for investment success
"""
        
        return {
            "crash_date": crash_date.isoformat(),
            "news_summary": {
                "general_market_articles": 0,
                "portfolio_specific_articles": 0,
                "total_articles_analyzed": 0,
                "fallback_used": True
            },
            "sentiment_analysis": {
                "sentiment_category": "Very Negative",
                "average_sentiment": -0.6,
                "total_articles": 0,
                "note": "Historical crash period - sentiment typically very negative"
            },
            "market_events": [],
            "ai_explanation": ai_explanation,
            "portfolio_impact": "General market stress affected most asset classes during this period.",
            "key_headlines": []
        }
    
    def analyze_portfolio_crash_impact(self, portfolio_news: Dict, 
                                      stocks_picked: List[Dict]) -> str:
        """
        Analyze how the crash specifically affected the user's portfolio stocks.
        """
        
        if not portfolio_news:
            return "Your portfolio consists mainly of diversified ETFs, which typically weather market storms better than individual stocks."
        
        impact_analysis = []
        
        for symbol, articles in portfolio_news.items():
            if articles:
                # Find the stock name
                stock_name = symbol
                for stock in stocks_picked:
                    if stock.get('symbol') == symbol:
                        stock_name = stock.get('name', symbol)
                        break
                
                negative_articles = sum(1 for article in articles 
                                      if 'decline' in article.get('headline', '').lower() 
                                      or 'fall' in article.get('headline', '').lower())
                
                if negative_articles > 0:
                    impact_analysis.append(f"{stock_name} ({symbol}) saw {negative_articles} negative headlines during the crash period")
                else:
                    impact_analysis.append(f"{stock_name} ({symbol}) had {len(articles)} news articles but no obviously negative headlines")
        
        if impact_analysis:
            return "Portfolio-specific impact: " + "; ".join(impact_analysis)
        else:
            return "No specific negative impact detected on your individual portfolio holdings during this crash period."
    
    def get_recovery_message(self, recovery_days: Optional[int]) -> str:
        """Generate user-friendly recovery message."""
        
        if recovery_days is None:
            return "Market recovery data not available for this period."
        
        if recovery_days <= 30:
            return f"Markets recovered quickly in just {recovery_days} days. 📈"
        elif recovery_days <= 90:
            return f"Markets took {recovery_days} days to recover - typical for minor corrections. 📊"
        elif recovery_days <= 365:
            return f"Recovery took {recovery_days} days ({recovery_days//30} months) - patience paid off for long-term investors. ⏳"
        else:
            years = recovery_days // 365
            return f"This was a major crash that took {years} year(s) to recover from. Long-term investing still prevailed. 💪"
    
    def generate_crash_insight(self, crash: Dict[str, Any]) -> str:
        """Generate educational insight about a specific crash."""
        
        severity = crash['severity']
        recovery_days = crash.get('recovery_time_days')
        
        if severity >= 0.5:  # 50%+ crash
            insight = "This was a major market crash that tested investor patience. "
        elif severity >= 0.3:  # 30%+ crash
            insight = "This significant market correction reminded investors of the importance of diversification. "
        else:
            insight = "This market decline was a normal part of investing cycles. "
        
        if recovery_days:
            if recovery_days <= 365:
                insight += f"The relatively quick recovery in {recovery_days} days shows markets' resilience over time."
            else:
                insight += f"Though recovery took {recovery_days//365} years, patient investors were ultimately rewarded."
        else:
            insight += "Historical data shows that markets eventually recover from downturns."
        
        return insight
    
    def generate_user_friendly_crash_explanation(self, crash: Dict[str, Any], 
                                                news_analysis: Dict[str, Any]) -> str:
        """
        Generate a user-friendly explanation of what happened during the crash.
        """
        
        crash_date = datetime.fromisoformat(crash['crash_date'].replace('Z', '+00:00'))
        severity = crash['severity']
        
        explanation = f"**Market Crash Alert - {crash_date.strftime('%B %Y')}** 📉\n\n"
        explanation += f"During this period, your portfolio experienced a {severity} decline. "
        
        # Add news-based explanation if available
        if 'ai_explanation' in news_analysis and news_analysis['ai_explanation']:
            explanation += "Here's what our analysis found:\n\n"
            explanation += news_analysis['ai_explanation'][:500] + "..."
        else:
            explanation += "This was part of a broader market correction during this period."
        
        # Add recovery information
        recovery_days = crash.get('recovery_time_days')
        if recovery_days:
            if recovery_days < 365:
                explanation += f"\n\n**Good News**: Markets recovered in approximately {recovery_days} days, showing the resilience of long-term investing."
            else:
                years = recovery_days // 365
                explanation += f"\n\n**Important Context**: While recovery took {years} year(s), patient investors who stayed invested were ultimately rewarded."
        
        explanation += "\n\n**Remember**: Market crashes are scary but temporary. Recoveries are permanent. 💪"
        
        return explanation
    
    def generate_overall_crash_message(self, crashes: List[Dict[str, Any]]) -> str:
        """Generate overall message about crashes in the simulation period."""
        
        if not crashes:
            return "No major market crashes (30%+ declines) occurred during your simulation period."
        
        total_crashes = len(crashes)
        severe_crashes = len([c for c in crashes if c['severity'] >= 0.4])
        
        if total_crashes == 1:
            return f"Your simulation period included 1 significant market decline. This is normal over long investment periods."
        else:
            message = f"Your simulation period included {total_crashes} significant market declines"
            if severe_crashes > 0:
                message += f" (including {severe_crashes} major crashes)"
            message += ". This demonstrates the importance of staying invested through market cycles."
            return message
    
    def generate_crash_education_summary(self, crashes: List[Dict[str, Any]]) -> str:
        """Generate educational summary about market crashes."""
        
        if not crashes:
            return """
Market crashes are a normal part of investing, though your simulation period was relatively stable. 
Historical data shows that markets experience significant declines (20%+ drops) roughly every 3-4 years, 
with major crashes (40%+ drops) occurring roughly once per decade. The key to successful long-term 
investing is staying disciplined during these downturns and continuing to invest regularly.
"""
        
        avg_recovery = None
        if crashes:
            recovery_times = [c.get('recovery_time_days') for c in crashes if c.get('recovery_time_days')]
            if recovery_times:
                avg_recovery = sum(recovery_times) // len(recovery_times)
        
        summary = f"""
Your simulation experienced {len(crashes)} significant market decline(s). This is actually quite normal - 
historical data shows major market corrections happen regularly. Key lessons:

1. **Markets Always Recover**: Every major crash in history has been followed by recovery and new highs.
2. **Time Is Your Friend**: Long-term investors who stayed invested through crashes were rewarded.
3. **Don't Panic Sell**: The biggest mistake is selling during crashes and missing the recovery.
4. **Keep Contributing**: Market downturns are actually opportunities to buy investments at lower prices.
"""
        
        if avg_recovery:
            summary += f"\n5. **Recovery Takes Time**: On average, your simulation's crashes took {avg_recovery} days to recover, showing patience is essential."
        
        return summary
    
    def generate_key_crash_insights(self, crashes: List[Dict], 
                                   crash_analyses: List[Dict]) -> List[str]:
        """
        Generate key insights from crash analysis for users.
        """
        
        insights = []
        
        if not crashes:
            insights.append("Your simulation period was relatively stable with no major market crashes detected.")
            return insights
        
        # Recovery time insights
        recovery_times = [c.get('recovery_time_days') for c in crashes if c.get('recovery_time_days')]
        if recovery_times:
            avg_recovery = sum(recovery_times) // len(recovery_times)
            insights.append(f"On average, market crashes in your simulation took {avg_recovery} days to recover, demonstrating markets' resilience.")
        
        # Severity insights
        severe_crashes = [c for c in crashes if c['severity'] >= 0.4]
        if severe_crashes:
            insights.append(f"Your simulation included {len(severe_crashes)} major crash(es) of 40%+, yet your portfolio strategy still delivered results.")
        
        # News-based insights
        crashes_with_news = [c for c in crash_analyses if 'news_analysis' in c]
        if crashes_with_news:
            insights.append(f"We analyzed news from {len(crashes_with_news)} recent crash(es) to help you understand what drove market movements.")
        
        # Educational insight
        insights.append("Each crash in your simulation represents a real market event that tested investor patience - staying invested through these periods is key to long-term success.")
        
        return insights
    
    async def get_simulation_crash_details(self, simulation_id: int, db) -> Dict[str, Any]:
        """
        Get detailed crash analysis for a specific simulation.
        
        Args:
            simulation_id: ID of the simulation to analyze
            db: Database session
            
        Returns:
            Detailed crash analysis for the simulation
        """
        
        try:
            from database import models
            
            # Get simulation from database
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            if not simulation:
                return {"error": "Simulation not found"}
            
            # Extract crash analysis from results
            crash_analysis = simulation.results.get("market_crash_analysis")
            
            if not crash_analysis:
                return {"message": "No crash analysis available for this simulation"}
            
            return {
                "simulation_id": simulation_id,
                "crash_analysis": crash_analysis,
                "analysis_date": datetime.now().isoformat(),
                "educational_resources": self._get_crash_educational_resources()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting simulation crash details: {e}")
            return {"error": str(e)}
    
    def _get_crash_educational_resources(self) -> List[Dict[str, str]]:
        """Get educational resources about market crashes."""
        
        return [
            {
                "title": "Understanding Market Volatility",
                "description": "Learn why markets go up and down and how to stay calm during turbulent times.",
                "key_points": "Market volatility is normal, diversification helps, time horizon matters"
            },
            {
                "title": "Dollar-Cost Averaging During Downturns",
                "description": "How continuing to invest during market crashes can improve long-term returns.",
                "key_points": "Buy more shares when prices are low, reduces average cost basis, smooths volatility"
            },
            {
                "title": "Historical Market Recovery Patterns",
                "description": "Every major market crash in history has been followed by recovery and new highs.",
                "key_points": "Markets always recover, patience is rewarded, selling during crashes locks in losses"
            }
        ]