# api/routers/ai_analysis.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from services.portfolio_simulator import simulate_portfolio
from services.ai_analysis import AIAnalysisService
from services.news_analysis import NewsAnalysisService
from database.database import get_db
import os
from datetime import datetime
from typing import Optional, List

router = APIRouter(prefix="/ai", tags=["ai-analysis"])

# Initialize AI service
ai_service = AIAnalysisService()

# Environment variables
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

@router.post("/simulate")
async def create_portfolio_simulation(sim_input: dict, db: Session = Depends(get_db)):
    """Create a new portfolio simulation with AI summary"""
    try:
        # Run the simulation (this calls your existing portfolio_simulator.py)
        result = simulate_portfolio(sim_input, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_portfolio(portfolio_data: dict):
    """Analyze existing portfolio performance with news context"""
    try:
        result = await ai_service.analyze_portfolio_performance(portfolio_data)
        return {
            "status": "success",
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@router.post("/analyze-risk")
async def analyze_risk(portfolio_data: dict):
    """Analyze portfolio risk and allocation with market sentiment"""
    try:
        result = await ai_service.analyze_risk_allocation(portfolio_data)
        return {
            "status": "success",
            "risk_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

@router.post("/explain-changes")
async def explain_changes(portfolio_data: dict, previous_data: Optional[dict] = None):
    """Explain portfolio changes over time with news context"""
    try:
        result = await ai_service.explain_portfolio_changes(portfolio_data, previous_data)
        return {
            "status": "success",
            "explanation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change explanation failed: {str(e)}")

# NEW ENHANCED ENDPOINTS WITH NEWS INTEGRATION

@router.post("/analyze-with-news")
async def analyze_with_news_context(
    portfolio_data: dict, 
    days_back: int = Query(default=7, ge=1, le=30, description="Number of days to look back for news"),
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
        # Get enhanced analysis with news context
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back)
        
        return {
            'status': 'success',
            'portfolio_analysis': analysis,
            'parameters': {
                'days_back': days_back,
                'analysis_date': datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News-enhanced analysis failed: {str(e)}")

@router.post("/portfolio-news-summary")
async def get_portfolio_news_summary(
    portfolio_data: dict,
    days_back: int = Query(default=7, ge=1, le=30, description="Number of days to look back for news")
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
        symbols = ai_service.extract_symbols_from_portfolio(portfolio_data)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No valid symbols found in portfolio data")
        
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
        
        return {
            'status': 'success',
            'portfolio_news_analysis': news_analysis,
            'ai_summary': ai_summary,
            'analysis_period_days': days_back,
            'symbols_analyzed': symbols,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio news summary failed: {str(e)}")

@router.get("/stock-sentiment/{symbol}")
async def get_stock_sentiment(
    symbol: str,
    days_back: int = Query(default=7, ge=1, le=30, description="Number of days to look back for news")
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
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
            # Get news for the symbol
            news_data = await news_service.get_market_news([symbol.upper()], days_back)
            articles = news_data.get(symbol.upper(), [])
            
            # Analyze sentiment
            sentiment_analysis = await news_service.analyze_sentiment(articles)
            
            # Detect events
            events = await news_service.get_market_events(articles)
            
            # Get price data
            price_data = await ai_service._get_price_data(symbol.upper(), days_back)
            
            # Generate AI insights
            ai_insights = await ai_service._generate_stock_insights(
                symbol.upper(), sentiment_analysis, events, price_data
            )
            
            return {
                'status': 'success',
                'symbol': symbol.upper(),
                'analysis_period_days': days_back,
                'sentiment_analysis': sentiment_analysis,
                'market_events': events,
                'price_data': price_data,
                'ai_insights': ai_insights,
                'news_impact_score': ai_service._calculate_news_impact(sentiment_analysis, events, price_data),
                'analysis_date': datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock sentiment analysis failed: {str(e)}")

@router.get("/market-news")
async def get_general_market_news(
    category: str = Query(default="general", description="News category (general, forex, crypto, merger)"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of articles to return")
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
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
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
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market news fetch failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_symbols(
    symbols: List[str],
    days_back: int = Query(default=7, ge=1, le=30, description="Number of days to look back for news"),
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
        # Limit to prevent abuse
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed per batch")
        
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        results = {}
        
        async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
            # Get news for all symbols
            news_data = await news_service.get_market_news([s.upper() for s in symbols], days_back)
            
            for symbol in symbols:
                symbol_upper = symbol.upper()
                try:
                    articles = news_data.get(symbol_upper, [])
                    
                    # Analyze sentiment
                    sentiment_analysis = await news_service.analyze_sentiment(articles)
                    
                    # Detect events
                    events = await news_service.get_market_events(articles)
                    
                    # Get price data
                    price_data = await ai_service._get_price_data(symbol_upper, days_back)
                    
                    result_data = {
                        'sentiment_analysis': sentiment_analysis,
                        'market_events': events,
                        'price_data': price_data,
                        'news_impact_score': ai_service._calculate_news_impact(sentiment_analysis, events, price_data)
                    }
                    
                    # Add AI insights if requested
                    if include_ai_insights:
                        ai_insights = await ai_service._generate_stock_insights(
                            symbol_upper, sentiment_analysis, events, price_data
                        )
                        result_data['ai_insights'] = ai_insights
                    
                    results[symbol_upper] = result_data
                    
                except Exception as symbol_error:
                    results[symbol_upper] = {'error': str(symbol_error)}
        
        return {
            'status': 'success',
            'symbols_analyzed': len(results),
            'analysis_period_days': days_back,
            'results': results,
            'analysis_date': datetime.now().isoformat()
        }
        
    except Exception as e:
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
        # Default alert thresholds
        default_thresholds = {
            'sentiment_threshold': 0.3,  # Alert if |sentiment| > 0.3
            'news_volume_threshold': 15,  # Alert if > 15 articles in period
            'high_impact_events': ['earnings', 'merger_acquisition', 'regulatory']
        }
        
        thresholds = {**default_thresholds, **(alert_thresholds or {})}
        
        symbols = ai_service.extract_symbols_from_portfolio(portfolio_data)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No valid symbols found in portfolio data")
        
        # Get analysis for alerts
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back=7)
        
        alerts = []
        
        for symbol, data in analysis.items():
            if 'error' in data:
                continue
                
            sentiment_data = data.get('sentiment_analysis', {})
            events = data.get('market_events', [])
            
            symbol_alerts = []
            
            # Sentiment alerts
            avg_sentiment = sentiment_data.get('average_sentiment', 0)
            if abs(avg_sentiment) > thresholds['sentiment_threshold']:
                sentiment_type = "Very Positive" if avg_sentiment > 0 else "Very Negative"
                
                # Generate AI explanation for the alert
                alert_explanation = await ai_service._get_groq_response(f"""
Explain in 1-2 sentences why a sentiment score of {avg_sentiment:.3f} for {symbol} 
should be considered a {sentiment_type.lower()} alert for a beginner investor.
""")
                
                symbol_alerts.append({
                    'type': 'sentiment',
                    'severity': 'high' if abs(avg_sentiment) > 0.5 else 'medium',
                    'message': f"{sentiment_type} news sentiment detected ({avg_sentiment:.3f})",
                    'ai_explanation': alert_explanation,
                    'value': avg_sentiment
                })
            
            # News volume alerts
            news_count = sentiment_data.get('total_articles', 0)
            if news_count > thresholds['news_volume_threshold']:
                symbol_alerts.append({
                    'type': 'high_news_volume',
                    'severity': 'medium',
                    'message': f"High news activity detected ({news_count} articles)",
                    'ai_explanation': f"Increased news coverage often indicates significant developments or market interest in {symbol}.",
                    'value': news_count
                })
            
            # Event-based alerts
            for event in events:
                event_types = event.get('event_types', [])
                if any(event_type in thresholds['high_impact_events'] for event_type in event_types):
                    symbol_alerts.append({
                        'type': 'high_impact_event',
                        'severity': 'high',
                        'message': f"High-impact event detected: {', '.join(event_types)}",
                        'headline': event.get('headline', ''),
                        'event_types': event_types,
                        'ai_explanation': f"Events like {', '.join(event_types)} can significantly impact {symbol}'s stock price and should be monitored closely."
                    })
            
            if symbol_alerts:
                alerts.append({
                    'symbol': symbol,
                    'alert_count': len(symbol_alerts),
                    'alerts': symbol_alerts
                })
        
        # Sort by severity and alert count
        alerts.sort(key=lambda x: (
            sum(1 for alert in x['alerts'] if alert['severity'] == 'high'),
            x['alert_count']
        ), reverse=True)
        
        return {
            'status': 'success',
            'total_symbols_with_alerts': len(alerts),
            'total_alerts': sum(alert['alert_count'] for alert in alerts),
            'thresholds_used': thresholds,
            'alerts': alerts,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert generation failed: {str(e)}")

@router.get("/health-check")
async def health_check():
    """Check if the AI analysis service and all dependencies are working"""
    try:
        health_status = {
            'status': 'healthy',
            'groq_configured': bool(ai_service.groq_api_key),
            'finnhub_configured': bool(FINNHUB_API_KEY),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test Finnhub connection if configured
        if FINNHUB_API_KEY:
            try:
                async with NewsAnalysisService(FINNHUB_API_KEY) as news_service:
                    test_news = await news_service.get_general_market_news("general", 1)
                    health_status['finnhub_connection'] = 'active' if test_news else 'inactive'
            except Exception:
                health_status['finnhub_connection'] = 'error'
        else:
            health_status['finnhub_connection'] = 'not_configured'
        
        return health_status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/config")
async def get_service_config():
    """Get current service configuration and capabilities"""
    return {
        'groq_configured': bool(ai_service.groq_api_key),
        'finnhub_configured': bool(FINNHUB_API_KEY),
        'model': ai_service.model,
        'max_symbols_per_batch': 20,
        'max_days_back': 30,
        'available_news_categories': ['general', 'forex', 'crypto', 'merger'],
        'supported_analysis_types': [
            'portfolio_summary',
            'performance_analysis', 
            'risk_analysis',
            'change_explanation',
            'news_sentiment',
            'market_events',
            'batch_analysis',
            'alerts'
        ],
        'features': {
            'ai_insights': True,
            'news_integration': True,
            'sentiment_analysis': True,
            'market_events': True,
            'real_time_alerts': True,
            'educational_summaries': True
        }
    }