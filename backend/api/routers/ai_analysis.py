# api/routers/ai_analysis.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
# 🔧 FIXED: Removed problematic imports from old portfolio simulator
# from services.portfolio_simulator import (
#     simulate_portfolio, 
#     get_shap_visualization, 
#     analyze_simulation_with_news
# )
from services.ai_analysis import AIAnalysisService
from database.db import get_db 
from database.models import Simulation
import os
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter( tags=["ai-analysis"])

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
    Create a new portfolio simulation with enhanced AI integration
    
    This endpoint automatically detects and uses the best available simulator:
    - Enhanced modular simulator (when available)
    - Standard portfolio simulator (fallback)
    
    Features include:
    - Goal-oriented portfolio optimization
    - Smart goal calculations (fixes 0% return issue)
    - Market crash detection (when enhanced simulator available)
    - SHAP explainable AI explanations (when available)
    - Enhanced educational AI summaries
    """
    try:
        logger.info(f"🚀 Starting portfolio simulation for user with goal: {sim_input.get('goal', 'unknown')}")
        
        # Validate required inputs
        if not sim_input.get('target_value') or not sim_input.get('timeframe'):
            raise HTTPException(
                status_code=400, 
                detail="target_value and timeframe are required parameters"
            )
        
        # Try to use enhanced modular simulator first
        try:
            from services.portfolio_simulator.main import simulate_portfolio
            logger.info("🎯 Using enhanced modular portfolio simulator")
            result = await simulate_portfolio(sim_input, db)
            
        except ImportError:
            # Fall back to standard simulator
            logger.info("📊 Using standard portfolio simulator")
            from services.portfolio_simulator import simulate_portfolio as standard_simulate
            result = await standard_simulate(sim_input, db)
        
        enhanced_used = result.get('wealthwise_enhanced', False)
        logger.info(f"✅ Simulation completed successfully. Enhanced: {enhanced_used}")
        
        return {
            **result,
            "simulator_type": "Enhanced Modular" if enhanced_used else "Standard",
            "api_endpoint": "/ai/simulate"
        }
        
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

# ENHANCED ENDPOINTS WITH MODULAR SIMULATOR INTEGRATION

@router.get("/simulation/{simulation_id}/shap-visualization")
async def get_simulation_shap_visualization(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Generate SHAP visualization for enhanced portfolio simulation
    
    This endpoint creates visual explanations of why the AI made specific 
    stock recommendations for a simulation (requires enhanced modular simulator).
    """
    try:
        logger.info(f"🎨 Generating SHAP visualization for simulation {simulation_id}")
        
        # Try to use enhanced modular simulator
        try:
            from services.portfolio_simulator.main import generate_shap_visualization
            viz_path = await generate_shap_visualization(simulation_id, db)
            
            if viz_path:
                logger.info(f"✅ SHAP visualization created: {viz_path}")
                return {
                    "status": "success",
                    "visualization_path": viz_path,
                    "simulation_id": simulation_id,
                    "message": "SHAP explanation visualization generated successfully"
                }
            else:
                raise HTTPException(
                    status_code=404, 
                    detail="SHAP visualization not available for this simulation"
                )
                
        except ImportError:
            logger.warning("⚠️ Enhanced modular simulator not available")
            raise HTTPException(
                status_code=503,
                detail="SHAP visualizations require the enhanced modular simulator. This feature will be available when the enhanced simulator is deployed."
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
    """
    try:
        logger.info(f"📰 Analyzing simulation {simulation_id} with current news context")
        
        # Try to use enhanced modular simulator
        try:
            from services.portfolio_simulator.main import get_simulation_crash_analysis
            analysis = await get_simulation_crash_analysis(simulation_id, db)
            
            if "error" in analysis:
                raise HTTPException(status_code=404, detail=analysis["error"])
            
            logger.info(f"✅ Enhanced news analysis completed for simulation {simulation_id}")
            return {
                "status": "success",
                **analysis
            }
            
        except ImportError:
            # Fallback: Get simulation and do basic news analysis
            logger.info("📊 Using basic news analysis (enhanced simulator not available)")
            
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            # Extract portfolio data and do basic analysis
            results = simulation.results or {}
            stocks_picked = results.get("stocks_picked", [])
            
            if not stocks_picked:
                raise HTTPException(status_code=400, detail="No portfolio data found in simulation")
            
            # Use AI service for basic analysis
            portfolio_data = {"stocks": [stock.get("symbol", "") for stock in stocks_picked]}
            basic_analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back=7)
            
            return {
                "status": "success",
                "simulation_id": simulation_id,
                "analysis": basic_analysis,
                "note": "Basic news analysis provided (enhanced features available when modular simulator is deployed)"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing simulation with news: {e}")
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")

@router.get("/simulation/{simulation_id}/crash-analysis")
async def get_simulation_crash_analysis(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Get market crash analysis for a simulation (enhanced simulator feature)
    """
    try:
        logger.info(f"📉 Getting crash analysis for simulation {simulation_id}")
        
        # Try to use enhanced modular simulator
        try:
            from services.portfolio_simulator.main import get_simulation_crash_analysis
            crash_analysis = await get_simulation_crash_analysis(simulation_id, db)
            
            if "error" in crash_analysis:
                raise HTTPException(status_code=404, detail=crash_analysis["error"])
            
            return {
                "status": "success",
                **crash_analysis
            }
            
        except ImportError:
            logger.warning("⚠️ Enhanced modular simulator not available")
            return {
                "simulation_id": simulation_id,
                "message": "Market crash analysis requires the enhanced modular simulator",
                "basic_info": "Market volatility is a normal part of long-term investing. Historical data shows that markets recover from downturns over time.",
                "status": "enhanced_feature_not_available"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting crash analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Crash analysis failed: {str(e)}")

@router.post("/analyze-with-news")
async def analyze_with_news_context(
    portfolio_data: dict, 
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
):
    """
    Comprehensive portfolio analysis with news sentiment and market events context
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
    """
    try:
        logger.info(f"📊 Analyzing sentiment for {symbol.upper()} over {days_back} days")
        
        if not FINNHUB_API_KEY:
            raise HTTPException(status_code=500, detail="Finnhub API key not configured")
        
        # Validate symbol format
        symbol_clean = symbol.upper().strip()
        if not symbol_clean or len(symbol_clean) > 10:
            raise HTTPException(status_code=400, detail="Invalid stock symbol format")
        
        # Try to import news analysis service
        try:
            from services.news_analysis import NewsAnalysisService
            
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
                
        except ImportError:
            logger.warning("⚠️ News analysis service not available")
            raise HTTPException(
                status_code=503,
                detail="News analysis service not available. Ensure news_analysis module is properly configured."
            )
            
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
        
        try:
            from services.news_analysis import NewsAnalysisService
            
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
                
        except ImportError:
            logger.warning("⚠️ News analysis service not available")
            raise HTTPException(
                status_code=503,
                detail="News analysis service not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Market news fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market news fetch failed: {str(e)}")

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
        
        # Check enhanced modular simulator availability
        try:
            from services.portfolio_simulator.main import simulate_portfolio
            health_status['enhanced_simulator_available'] = True
            
            # Check for SHAP availability
            try:
                from services.portfolio_simulator.main import generate_shap_visualization
                health_status['shap_visualization_available'] = True
            except ImportError:
                health_status['shap_visualization_available'] = False
                
        except ImportError:
            health_status['enhanced_simulator_available'] = False
            health_status['shap_visualization_available'] = False
        
        # Check news analysis service
        try:
            from services.news_analysis import NewsAnalysisService
            health_status['news_analysis_available'] = True
        except ImportError:
            health_status['news_analysis_available'] = False
        
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
            health_status['ai_service'] == 'active'
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

@router.get("/status")
async def get_service_status():
    """Quick service status check - lighter than health-check"""
    try:
        # Check enhanced simulator availability
        enhanced_available = False
        try:
            from services.portfolio_simulator.main import simulate_portfolio
            enhanced_available = True
        except ImportError:
            pass
        
        return {
            'service': 'AI Analysis Router',
            'status': 'online',
            'enhanced_simulator': enhanced_available,
            'simulator_type': 'Enhanced Modular' if enhanced_available else 'Standard',
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