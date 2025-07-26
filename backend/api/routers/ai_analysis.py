# api/routers/ai_analysis.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from services.portfolio_simulator import (
    simulate_portfolio, 
    get_shap_visualization, 
    analyze_simulation_with_news
)
from services.ai_analysis import AIAnalysisService
from services.news_analysis import NewsAnalysisService
from database.db import get_db
import os
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai-analysis"])

# Initialize AI service
ai_service = AIAnalysisService()

# Environment variables
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Configuration constants
MAX_SYMBOLS_PER_BATCH = 20
MAX_DAYS_BACK = 30
MIN_DAYS_BACK = 1
DEFAULT_DAYS_BACK = 7
DEFAULT_NEWS_LIMIT = 20
MAX_NEWS_LIMIT = 100

@router.post("/simulate")
async def create_portfolio_simulation(sim_input: dict, db: Session = Depends(get_db)):
    """
    Create a new portfolio simulation with enhanced WealthWise AI integration
    
    This endpoint now includes:
    - Goal-oriented portfolio optimization
    - SHAP explainable AI explanations  
    - Market regime detection
    - Enhanced educational AI summaries
    - Fallback protection for reliability
    """
    try:
        logger.info(f"🚀 Starting portfolio simulation for user with goal: {sim_input.get('goal', 'unknown')}")
        
        # Validate required inputs
        if not sim_input.get('target_value') or not sim_input.get('timeframe'):
            raise HTTPException(
                status_code=400, 
                detail="target_value and timeframe are required parameters"
            )
        
        # Run enhanced simulation with WealthWise integration
        result = await simulate_portfolio(sim_input, db)
        
        logger.info(f"✅ Simulation completed successfully. Enhanced: {result.get('wealthwise_enhanced', False)}")
        return result
        
    except ValueError as ve:
        logger.warning(f"⚠️ Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio simulation failed: {str(e)}")

@router.post("/analyze")
async def analyze_portfolio(portfolio_data: dict):
    """Analyze existing portfolio performance with news context"""
    try:
        logger.info("📊 Starting portfolio performance analysis")
        result = await ai_service.analyze_portfolio_performance(portfolio_data)
        return {
            "status": "success",
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Portfolio analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@router.post("/analyze-risk")
async def analyze_risk(portfolio_data: dict):
    """Analyze portfolio risk and allocation with market sentiment"""
    try:
        logger.info("⚖️ Starting portfolio risk analysis")
        result = await ai_service.analyze_risk_allocation(portfolio_data)
        return {
            "status": "success",
            "risk_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Risk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

@router.post("/explain-changes")
async def explain_changes(portfolio_data: dict, previous_data: Optional[dict] = None):
    """Explain portfolio changes over time with news context"""
    try:
        logger.info("🔄 Analyzing portfolio changes")
        result = await ai_service.explain_portfolio_changes(portfolio_data, previous_data)
        return {
            "status": "success",
            "explanation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Change explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Change explanation failed: {str(e)}")

# NEW ENHANCED ENDPOINTS WITH NEWS INTEGRATION AND WEALTHWISE FEATURES

@router.get("/simulation/{simulation_id}/shap-visualization")
async def get_simulation_shap_visualization(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Generate SHAP visualization for enhanced portfolio simulation
    
    This endpoint creates visual explanations of why the WealthWise AI
    made specific stock recommendations for a simulation.
    
    Args:
        simulation_id: ID of the simulation to visualize
        
    Returns:
        Path to generated SHAP visualization or error if unavailable
    """
    try:
        logger.info(f"🎨 Generating SHAP visualization for simulation {simulation_id}")
        
        viz_path = await get_shap_visualization(simulation_id, db)
        
        if viz_path:
            logger.info(f"✅ SHAP visualization created: {viz_path}")
            return {
                "status": "success",
                "visualization_path": viz_path,
                "simulation_id": simulation_id,
                "message": "SHAP explanation visualization generated successfully"
            }
        else:
            logger.warning(f"⚠️ SHAP visualization not available for simulation {simulation_id}")
            raise HTTPException(
                status_code=404, 
                detail="SHAP visualization not available. This simulation may not have been created with WealthWise enhanced features."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating SHAP visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP visualization: {str(e)}")

@router.get("/simulation/{simulation_id}/news-analysis")
async def get_simulation_news_analysis(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Analyze a simulation with current news context
    
    This combines the simulation results with current news analysis
    for the recommended stocks to provide updated insights.
    
    Args:
        simulation_id: ID of the simulation to analyze
        
    Returns:
        Analysis combining simulation and current news data
    """
    try:
        logger.info(f"📰 Analyzing simulation {simulation_id} with current news context")
        
        analysis = await analyze_simulation_with_news(simulation_id, db)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        logger.info(f"✅ News analysis completed for simulation {simulation_id}")
        return {
            "status": "success",
            **analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing simulation with news: {e}")
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")

@router.post("/analyze-with-news")
async def analyze_with_news_context(
    portfolio_data: dict, 
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
):
    """
    Comprehensive portfolio analysis with news sentiment and market events context
    
    Args:
        portfolio_data: Portfolio holdings data
        days_back: Number of days to look back for news (1-30)
    
    Returns:
        Enhanced analysis including sentiment, events, and AI insights
    """
    try:
        logger.info(f"📊 Starting enhanced portfolio analysis with {days_back} days of news context")
        
        # Validate portfolio data
        if not portfolio_data or not isinstance(portfolio_data, dict):
            raise HTTPException(status_code=400, detail="Invalid portfolio data provided")
        
        # Get enhanced analysis with news context
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back)
        
        logger.info("✅ Enhanced portfolio analysis with news context completed")
        return {
            'status': 'success',
            'portfolio_analysis': analysis,
            'parameters': {
                'days_back': days_back,
                'analysis_date': datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ News-enhanced analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"News-enhanced analysis failed: {str(e)}")

@router.post("/portfolio-news-summary")
async def get_portfolio_news_summary(
    portfolio_data: dict,
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news")
):
    """
    Get a comprehensive news summary for the entire portfolio with AI insights
    
    Args:
        portfolio_data: Portfolio holdings data
        days_back: Number of days to look back for news
    
    Returns:
        Portfolio-level news summary with sentiment, events, and recommendations
    """
    try:
        logger.info(f"📰 Generating portfolio news summary for {days_back} days")
        
        # Validate portfolio data
        if not portfolio_data or not isinstance(portfolio_data, dict):
            raise HTTPException(status_code=400, detail="Invalid portfolio data provided")
        
        symbols = ai_service.extract_symbols_from_portfolio(portfolio_data)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No valid symbols found in portfolio data")
        
        logger.info(f"📈 Analyzing {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Get comprehensive news analysis
        news_analysis = await ai_service._get_portfolio_news_analysis(symbols, days_back)
        
        # Generate AI summary of the news analysis
        if news_analysis and 'error' not in news_analysis:
            ai_summary_prompt = f"""
Summarize this portfolio news analysis for a beginner investor:

PORTFOLIO SYMBOLS: {', '.join(symbols)}
OVERALL SENTIMENT: {news_analysis.get('overall_sentiment', 0):.3f}
TOTAL ARTICLES: {news_analysis.get('total_articles', 0)}
MARKET EVENTS: {len(news_analysis.get('market_events', []))}

Provide a brief, educational summary covering:
1. What the overall news sentiment means
2. Key events affecting the portfolio
3. What investors should watch for
4. Educational insights about news impact

Keep it beginner-friendly and encouraging.
"""
            ai_summary = await ai_service._get_groq_response(ai_summary_prompt)
        else:
            ai_summary = "Unable to generate AI summary due to insufficient news data."
        
        logger.info("✅ Portfolio news summary generated successfully")
        return {
            'status': 'success',
            'portfolio_news_analysis': news_analysis,
            'ai_summary': ai_summary,
            'analysis_period_days': days_back,
            'symbols_analyzed': symbols,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio news summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio news summary failed: {str(e)}")

@router.get("/stock-sentiment/{symbol}")
async def get_stock_sentiment(
    symbol: str,
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news")
):
    """
    Get detailed sentiment analysis and AI insights for a specific stock symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL)
        days_back: Number of days to look back for news
    
    Returns:
        Detailed sentiment analysis, news events, and AI insights for the symbol
    """
    try:
        logger.info(f"📊 Analyzing sentiment for {symbol.upper()} over {days_back} days")
        
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        # Validate symbol format
        symbol_clean = symbol.upper().strip()
        if not symbol_clean or len(symbol_clean) > 10:
            raise HTTPException(status_code=400, detail="Invalid stock symbol format")
        
        async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
            # Get news for the symbol
            news_data = await news_service.get_market_news([symbol_clean], days_back)
            articles = news_data.get(symbol_clean, [])
            
            # Analyze sentiment
            sentiment_analysis = await news_service.analyze_sentiment(articles)
            
            # Detect events
            events = await news_service.get_market_events(articles)
            
            # Get price data
            price_data = await ai_service._get_price_data(symbol_clean, days_back)
            
            # Generate AI insights
            ai_insights = await ai_service._generate_stock_insights(
                symbol_clean, sentiment_analysis, events, price_data
            )
            
            logger.info(f"✅ Sentiment analysis completed for {symbol_clean}")
            return {
                'status': 'success',
                'symbol': symbol_clean,
                'analysis_period_days': days_back,
                'sentiment_analysis': sentiment_analysis,
                'market_events': events,
                'price_data': price_data,
                'ai_insights': ai_insights,
                'news_impact_score': ai_service._calculate_news_impact(sentiment_analysis, events, price_data),
                'analysis_date': datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stock sentiment analysis failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stock sentiment analysis failed: {str(e)}")

@router.get("/market-news")
async def get_general_market_news(
    category: str = Query(default="general", description="News category (general, forex, crypto, merger)"),
    limit: int = Query(default=DEFAULT_NEWS_LIMIT, ge=1, le=MAX_NEWS_LIMIT, description="Number of articles to return")
):
    """
    Get general market news with sentiment analysis and AI insights
    
    Args:
        category: News category to fetch
        limit: Maximum number of articles to return
    
    Returns:
        List of recent market news articles with sentiment analysis and AI summary
    """
    try:
        logger.info(f"📰 Fetching {category} market news (limit: {limit})")
        
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        # Validate category
        valid_categories = ['general', 'forex', 'crypto', 'merger']
        if category not in valid_categories:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            )
        
        async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
            # Get general market news
            articles = await news_service.get_general_market_news(category, limit)
            
            # Analyze overall sentiment
            sentiment_analysis = await news_service.analyze_sentiment(articles)
            
            # Detect events
            events = await news_service.get_market_events(articles)
            
            # Generate AI market summary
            ai_summary_prompt = f"""
Provide a brief market summary based on this news analysis:

NEWS CATEGORY: {category}
ARTICLES ANALYZED: {len(articles)}
OVERALL SENTIMENT: {sentiment_analysis.get('average_sentiment', 0):.3f}
SENTIMENT CATEGORY: {sentiment_analysis.get('sentiment_category', 'Neutral')}
MARKET EVENTS: {len(events)}

Summarize in 2-3 sentences what this means for investors, focusing on education.
"""
            ai_summary = await ai_service._get_groq_response(ai_summary_prompt)
            
            logger.info(f"✅ Market news analysis completed for {category} category")
            return {
                'status': 'success',
                'category': category,
                'articles_count': len(articles),
                'overall_sentiment': sentiment_analysis,
                'market_events': events,
                'ai_market_summary': ai_summary,
                'articles': articles[:limit],  # Return the articles
                'analysis_date': datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Market news fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market news fetch failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_symbols(
    symbols: List[str],
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news"),
    include_ai_insights: bool = Query(default=True, description="Include AI-generated insights")
):
    """
    Batch analyze multiple symbols for news sentiment, events, and AI insights
    
    Args:
        symbols: List of stock symbols to analyze
        days_back: Number of days to look back for news
        include_ai_insights: Whether to include AI-generated insights
    
    Returns:
        Analysis results for all symbols with optional AI insights
    """
    try:
        logger.info(f"🔄 Starting batch analysis for {len(symbols)} symbols")
        
        # Validate input
        if not symbols or not isinstance(symbols, list):
            raise HTTPException(status_code=400, detail="symbols must be a non-empty list")
        
        # Clean and validate symbols
        clean_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str) and symbol.strip():
                clean_symbol = symbol.upper().strip()
                if len(clean_symbol) <= 10:  # Reasonable symbol length
                    clean_symbols.append(clean_symbol)
        
        if not clean_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        # Limit to prevent abuse and API overload
        if len(clean_symbols) > MAX_SYMBOLS_PER_BATCH:
            raise HTTPException(
                status_code=400, 
                detail=f"Maximum {MAX_SYMBOLS_PER_BATCH} symbols allowed per batch. Received {len(clean_symbols)}"
            )
        
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        logger.info(f"📊 Analyzing symbols: {', '.join(clean_symbols)}")
        
        results = {}
        failed_symbols = []
        
        async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
            # Get news for all symbols in batch (more efficient)
            try:
                news_data = await news_service.get_market_news(clean_symbols, days_back)
            except Exception as e:
                logger.error(f"❌ Failed to fetch news data: {e}")
                raise HTTPException(status_code=503, detail="News service temporarily unavailable")
            
            # Process each symbol
            for symbol in clean_symbols:
                try:
                    logger.debug(f"Processing {symbol}...")
                    articles = news_data.get(symbol, [])
                    
                    # Analyze sentiment
                    sentiment_analysis = await news_service.analyze_sentiment(articles)
                    
                    # Detect events
                    events = await news_service.get_market_events(articles)
                    
                    # Get price data
                    try:
                        price_data = await ai_service._get_price_data(symbol, days_back)
                    except Exception as price_error:
                        logger.warning(f"⚠️ Price data unavailable for {symbol}: {price_error}")
                        price_data = {"error": "Price data unavailable", "symbol": symbol}
                    
                    # Calculate news impact score
                    news_impact_score = ai_service._calculate_news_impact(sentiment_analysis, events, price_data)
                    
                    result_data = {
                        'symbol': symbol,
                        'sentiment_analysis': sentiment_analysis,
                        'market_events': events,
                        'price_data': price_data,
                        'news_impact_score': news_impact_score,
                        'articles_count': len(articles)
                    }
                    
                    # Add AI insights if requested
                    if include_ai_insights:
                        try:
                            ai_insights = await ai_service._generate_stock_insights(
                                symbol, sentiment_analysis, events, price_data
                            )
                            result_data['ai_insights'] = ai_insights
                        except Exception as ai_error:
                            logger.warning(f"⚠️ AI insights failed for {symbol}: {ai_error}")
                            result_data['ai_insights'] = f"AI insights unavailable: {str(ai_error)}"
                    
                    results[symbol] = result_data
                    logger.debug(f"✅ {symbol} analysis complete")
                    
                except Exception as symbol_error:
                    logger.error(f"❌ Analysis failed for {symbol}: {symbol_error}")
                    failed_symbols.append(symbol)
                    results[symbol] = {
                        'symbol': symbol,
                        'error': str(symbol_error),
                        'status': 'failed'
                    }
        
        # Generate batch summary
        successful_analyses = len([r for r in results.values() if 'error' not in r])
        
        logger.info(f"✅ Batch analysis complete: {successful_analyses}/{len(clean_symbols)} successful")
        
        return {
            'status': 'success',
            'symbols_requested': len(symbols),
            'symbols_processed': len(clean_symbols),
            'symbols_successful': successful_analyses,
            'symbols_failed': len(failed_symbols),
            'failed_symbols': failed_symbols,
            'analysis_period_days': days_back,
            'included_ai_insights': include_ai_insights,
            'results': results,
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_articles': sum(r.get('articles_count', 0) for r in results.values() if 'error' not in r),
                'avg_sentiment': sum(r.get('sentiment_analysis', {}).get('average_sentiment', 0) for r in results.values() if 'error' not in r) / max(successful_analyses, 1),
                'total_events': sum(len(r.get('market_events', [])) for r in results.values() if 'error' not in r)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/portfolio-alerts")
async def generate_portfolio_alerts(
    portfolio_data: dict,
    alert_thresholds: Optional[dict] = None
):
    """
    Generate AI-powered alerts based on news sentiment and market events
    
    Args:
        portfolio_data: Portfolio holdings data
        alert_thresholds: Custom thresholds for alerts (optional)
    
    Returns:
        List of alerts with AI-generated explanations
    """
    try:
        logger.info("🚨 Generating portfolio alerts")
        
        # Validate portfolio data
        if not portfolio_data or not isinstance(portfolio_data, dict):
            raise HTTPException(status_code=400, detail="Invalid portfolio data provided")
        
        # Default alert thresholds
        default_thresholds = {
            'sentiment_threshold': 0.3,  # Alert if |sentiment| > 0.3
            'news_volume_threshold': 15,  # Alert if > 15 articles in period
            'high_impact_events': ['earnings', 'merger_acquisition', 'regulatory', 'lawsuit', 'management_change']
        }
        
        thresholds = {**default_thresholds, **(alert_thresholds or {})}
        
        symbols = ai_service.extract_symbols_from_portfolio(portfolio_data)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No valid symbols found in portfolio data")
        
        logger.info(f"📊 Generating alerts for {len(symbols)} portfolio holdings")
        
        # Get analysis for alerts
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back=7)
        
        alerts = []
        
        for symbol, data in analysis.items():
            if 'error' in data:
                logger.warning(f"⚠️ Skipping {symbol} due to analysis error: {data['error']}")
                continue
                
            sentiment_data = data.get('sentiment_analysis', {})
            events = data.get('market_events', [])
            price_data = data.get('price_data', {})
            
            symbol_alerts = []
            
            # Sentiment alerts
            avg_sentiment = sentiment_data.get('average_sentiment', 0)
            if abs(avg_sentiment) > thresholds['sentiment_threshold']:
                sentiment_type = "Very Positive" if avg_sentiment > 0 else "Very Negative"
                
                # Generate AI explanation for the alert
                try:
                    alert_explanation = await ai_service._get_groq_response(f"""
Explain in 1-2 sentences why a sentiment score of {avg_sentiment:.3f} for {symbol} 
should be considered a {sentiment_type.lower()} alert for a beginner investor.
Focus on what this means and what action they might consider.
""")
                except Exception:
                    alert_explanation = f"The news sentiment for {symbol} is {sentiment_type.lower()}, which may indicate significant market developments affecting the stock."
                
                symbol_alerts.append({
                    'type': 'sentiment',
                    'severity': 'high' if abs(avg_sentiment) > 0.5 else 'medium',
                    'message': f"{sentiment_type} news sentiment detected ({avg_sentiment:.3f})",
                    'ai_explanation': alert_explanation,
                    'value': avg_sentiment,
                    'threshold_used': thresholds['sentiment_threshold']
                })
            
            # News volume alerts
            news_count = sentiment_data.get('total_articles', 0)
            if news_count > thresholds['news_volume_threshold']:
                symbol_alerts.append({
                    'type': 'high_news_volume',
                    'severity': 'medium',
                    'message': f"High news activity detected ({news_count} articles)",
                    'ai_explanation': f"Increased news coverage often indicates significant developments or heightened market interest in {symbol}. This could signal upcoming volatility or important announcements.",
                    'value': news_count,
                    'threshold_used': thresholds['news_volume_threshold']
                })
            
            # Event-based alerts
            for event in events:
                event_types = event.get('event_types', [])
                if Any(event_type in thresholds['high_impact_events'] for event_type in event_types):
                    symbol_alerts.append({
                        'type': 'high_impact_event',
                        'severity': 'high',
                        'message': f"High-impact event detected: {', '.join(event_types)}",
                        'headline': event.get('headline', '')[:100] + "..." if len(event.get('headline', '')) > 100 else event.get('headline', ''),
                        'event_types': event_types,
                        'ai_explanation': f"Events like {', '.join(event_types)} can significantly impact {symbol}'s stock price and should be monitored closely for potential portfolio adjustments.",
                        'event_date': event.get('date', '')
                    })
            
            # Price movement alerts (if price data available)
            if price_data and 'error' not in price_data:
                price_change = price_data.get('price_change_percent', 0)
                if abs(price_change) > 5:  # 5% price change threshold
                    direction = "surge" if price_change > 0 else "drop"
                    symbol_alerts.append({
                        'type': 'significant_price_movement',
                        'severity': 'medium' if abs(price_change) < 10 else 'high',
                        'message': f"Significant price {direction}: {price_change:+.1f}%",
                        'ai_explanation': f"A {abs(price_change):.1f}% price {direction} in {symbol} may be related to recent news or market sentiment changes. Consider reviewing your position.",
                        'value': price_change
                    })
            
            if symbol_alerts:
                alerts.append({
                    'symbol': symbol,
                    'alert_count': len(symbol_alerts),
                    'highest_severity': 'high' if Any(alert['severity'] == 'high' for alert in symbol_alerts) else 'medium',
                    'alerts': symbol_alerts
                })
        
        # Sort by severity and alert count
        alerts.sort(key=lambda x: (
            1 if x['highest_severity'] == 'high' else 0,
            x['alert_count']
        ), reverse=True)
        
        # Generate overall portfolio alert summary
        total_alerts = sum(alert['alert_count'] for alert in alerts)
        high_severity_count = sum(1 for alert in alerts if alert['highest_severity'] == 'high')
        
        portfolio_summary = f"""
Portfolio Alert Summary: {total_alerts} total alerts detected across {len(alerts)} holdings.
{high_severity_count} high-severity alerts require immediate attention.
Consider reviewing positions with multiple alerts or high-impact events.
"""
        
        logger.info(f"✅ Generated {total_alerts} alerts for {len(alerts)} symbols")
        
        return {
            'status': 'success',
            'total_symbols_with_alerts': len(alerts),
            'total_alerts': total_alerts,
            'high_severity_alerts': high_severity_count,
            'thresholds_used': thresholds,
            'portfolio_summary': portfolio_summary.strip(),
            'alerts': alerts,
            'generated_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Alert generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")

@router.get("/health-check")
async def health_check():
    """Check if the AI analysis service and all dependencies are working"""
    try:
        logger.info("🔍 Running health check")
        
        health_status = {
            'status': 'healthy',
            'groq_configured': bool(ai_service.groq_api_key),
            'finnhub_configured': bool(FINNHUB_API_KEY),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check WealthWise availability
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            health_status['wealthwise_available'] = WEALTHWISE_AVAILABLE
            
            if WEALTHWISE_AVAILABLE:
                from ai_models.stock_model.explainable_ai import SHAPExplainer
                shap_explainer = SHAPExplainer()
                health_status['shap_model_trained'] = shap_explainer.is_available()
        except ImportError:
            health_status['wealthwise_available'] = False
            health_status['shap_model_trained'] = False
        
        # Test Finnhub connection if configured
        if FINNHUB_API_KEY:
            try:
                async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
                    test_news = await news_service.get_general_market_news("general", 1)
                    health_status['finnhub_connection'] = 'active' if test_news else 'inactive'
            except Exception as e:
                health_status['finnhub_connection'] = 'error'
                health_status['finnhub_error'] = str(e)
        else:
            health_status['finnhub_connection'] = 'not_configured'
        
        # Test AI service
        try:
            test_response = await ai_service._get_groq_response("Test message")
            health_status['ai_service'] = 'active' if test_response else 'inactive'
        except Exception as e:
            health_status['ai_service'] = 'error'
            health_status['ai_error'] = str(e)
        
        # Overall health determination
        critical_services = [
            health_status['groq_configured'],
            health_status['finnhub_connection'] in ['active', 'not_configured']
        ]
        
        if all(critical_services):
            health_status['overall'] = 'healthy'
        else:
            health_status['overall'] = 'degraded'
            health_status['status'] = 'degraded'
        
        logger.info(f"✅ Health check complete: {health_status['overall']}")
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'overall': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/config")
async def get_service_config():
    """Get current service configuration and capabilities"""
    try:
        logger.info("⚙️ Retrieving service configuration")
        
        # Check WealthWise status
        wealthwise_status = False
        shap_available = False
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            wealthwise_status = WEALTHWISE_AVAILABLE
            
            if WEALTHWISE_AVAILABLE:
                from ai_models.stock_model.explainable_ai import SHAPExplainer
                shap_explainer = SHAPExplainer()
                shap_available = shap_explainer.is_available()
        except ImportError:
            wealthwise_status = False
        
        config = {
            'service_info': {
                'name': 'WealthWise AI Analysis Service',
                'version': '2.0.0',
                'enhanced_features': wealthwise_status
            },
            'api_configuration': {
                'groq_configured': bool(ai_service.groq_api_key),
                'finnhub_configured': bool(FINNHUB_API_KEY),
                'model': getattr(ai_service, 'model', 'unknown')
            },
            'wealthwise_integration': {
                'available': wealthwise_status,
                'shap_model_trained': shap_available,
                'goal_optimization': wealthwise_status,
                'market_regime_detection': wealthwise_status
            },
            'limits_and_constraints': {
                'max_symbols_per_batch': MAX_SYMBOLS_PER_BATCH,
                'max_days_back': MAX_DAYS_BACK,
                'min_days_back': MIN_DAYS_BACK,
                'default_days_back': DEFAULT_DAYS_BACK,
                'max_news_limit': MAX_NEWS_LIMIT
            },
            'available_news_categories': ['general', 'forex', 'crypto', 'merger'],
            'supported_analysis_types': [
                'portfolio_simulation',
                'portfolio_summary',
                'performance_analysis', 
                'risk_analysis',
                'change_explanation',
                'news_sentiment',
                'market_events',
                'batch_analysis',
                'alerts',
                'shap_explanations' if shap_available else None
            ],
            'features': {
                'ai_insights': True,
                'news_integration': bool(FINNHUB_API_KEY),
                'sentiment_analysis': bool(FINNHUB_API_KEY),
                'market_events': bool(FINNHUB_API_KEY),
                'real_time_alerts': True,
                'educational_summaries': True,
                'goal_oriented_optimization': wealthwise_status,
                'explainable_ai': shap_available,
                'market_regime_analysis': wealthwise_status,
                'fallback_protection': True
            },
            'endpoints': {
                'portfolio_simulation': '/ai/simulate',
                'portfolio_analysis': '/ai/analyze',
                'risk_analysis': '/ai/analyze-risk',
                'change_explanation': '/ai/explain-changes',
                'news_enhanced_analysis': '/ai/analyze-with-news',
                'portfolio_news_summary': '/ai/portfolio-news-summary',
                'stock_sentiment': '/ai/stock-sentiment/{symbol}',
                'market_news': '/ai/market-news',
                'batch_analysis': '/ai/batch-analyze',
                'portfolio_alerts': '/ai/portfolio-alerts',
                'shap_visualization': '/ai/simulation/{id}/shap-visualization' if shap_available else None,
                'news_analysis': '/ai/simulation/{id}/news-analysis',
                'health_check': '/ai/health-check',
                'configuration': '/ai/config'
            },
            'example_usage': {
                'portfolio_simulation': {
                    'method': 'POST',
                    'endpoint': '/ai/simulate',
                    'sample_payload': {
                        "goal": "retirement planning",
                        "target_value": 100000,
                        "lump_sum": 10000,
                        "monthly": 500,
                        "timeframe": 15,
                        "risk_score": 65,
                        "risk_label": "Moderate Aggressive"
                    }
                },
                'batch_analysis': {
                    'method': 'POST',
                    'endpoint': '/ai/batch-analyze',
                    'sample_payload': {
                        "symbols": ["AAPL", "GOOGL", "MSFT"],
                        "days_back": 7,
                        "include_ai_insights": True
                    }
                },
                'portfolio_alerts': {
                    'method': 'POST',
                    'endpoint': '/ai/portfolio-alerts',
                    'sample_payload': {
                        "portfolio_data": {"stocks": ["AAPL", "TSLA"]},
                        "alert_thresholds": {
                            "sentiment_threshold": 0.4,
                            "news_volume_threshold": 20
                        }
                    }
                }
            }
        }
        
        # Remove None values from supported_analysis_types
        config['supported_analysis_types'] = [t for t in config['supported_analysis_types'] if t is not None]
        
        # Remove None endpoints
        config['endpoints'] = {k: v for k, v in config['endpoints'].items() if v is not None}
        
        logger.info("✅ Service configuration retrieved")
        return config
        
    except Exception as e:
        logger.error(f"❌ Error retrieving service configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {str(e)}")

# Additional utility endpoints for development and testing

@router.get("/status")
async def get_service_status():
    """Quick service status check - lighter than health-check"""
    try:
        # Check WealthWise availability
        wealthwise_available = False
        try:
            from services.portfolio_simulator import WEALTHWISE_AVAILABLE
            wealthwise_available = WEALTHWISE_AVAILABLE
        except ImportError:
            pass
        
        return {
            'service': 'AI Analysis Router',
            'status': 'online',
            'wealthwise_enhanced': wealthwise_available,
            'endpoints_available': len([
                route for route in router.routes 
                if hasattr(route, 'path') and route.path.startswith('/ai/')
            ]),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'service': 'AI Analysis Router',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/endpoints")
async def list_all_endpoints():
    """List all available endpoints with their methods and descriptions"""
    try:
        endpoints_info = []
        
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                # Extract endpoint information
                endpoint_info = {
                    'path': route.path,
                    'methods': list(route.methods),
                    'name': getattr(route, 'name', 'unnamed'),
                    'summary': getattr(route, 'summary', ''),
                }
                
                # Try to get docstring
                if hasattr(route, 'endpoint') and route.endpoint:
                    docstring = route.endpoint.__doc__
                    if docstring:
                        # Get first line of docstring as description
                        description = docstring.strip().split('\n')[0]
                        endpoint_info['description'] = description
                
                endpoints_info.append(endpoint_info)
        
        # Sort by path
        endpoints_info.sort(key=lambda x: x['path'])
        
        return {
            'service': 'AI Analysis Router',
            'total_endpoints': len(endpoints_info),
            'endpoints': endpoints_info,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list endpoints: {str(e)}")

# Error handlers and middleware could be added here if needed

# For development: endpoint to clear caches or reset state
@router.post("/dev/reset")
async def development_reset():
    """Development endpoint to reset service state (only for testing)"""
    try:
        # Only allow in development mode
        import os
        if os.getenv("ENVIRONMENT", "production").lower() != "development":
            raise HTTPException(status_code=403, detail="Development endpoint not available in production")
        
        logger.info("🔄 Development reset requested")
        
        # Reset any caches or temporary state here
        # This is a placeholder for development use
        
        return {
            'status': 'success',
            'message': 'Development state reset completed',
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Development reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Background task examples for future enhancements
async def background_portfolio_analysis(portfolio_data: dict, user_id: str):
    """Background task for long-running portfolio analysis"""
    try:
        logger.info(f"🔄 Starting background analysis for user {user_id}")
        
        # Placeholder for background analysis logic
        # This could include:
        # - Deep market analysis
        # - Historical performance calculations
        # - Risk scenario modeling
        # - Notification generation
        
        logger.info(f"✅ Background analysis completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for user {user_id}: {e}")

# Optional: WebSocket endpoint for real-time updates
# This would require additional FastAPI WebSocket setup
"""
@router.websocket("/ws/portfolio-updates")
async def websocket_portfolio_updates(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send real-time portfolio updates
            # This could include:
            # - Live price updates
            # - News alerts
            # - Sentiment changes
            # - Market regime shifts
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
"""