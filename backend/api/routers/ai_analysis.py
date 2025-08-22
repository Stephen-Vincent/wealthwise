# api/routers/ai_analysis.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
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

router = APIRouter(tags=["ai-analysis"])

# Import the AI analysis service (this should remain the same)
try:
    from backend.services.portfolio_simulator.ai_analysis import AIAnalysisService
    ai_service = AIAnalysisService()
    AI_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI Analysis service not available: {e}")
    AI_ANALYSIS_AVAILABLE = False
    ai_service = None

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
    """
    try:
        logger.info(f"Starting portfolio simulation for user with goal: {sim_input.get('goal', 'unknown')}")
        
        # Validate required inputs
        if not sim_input.get('target_value') or not sim_input.get('timeframe'):
            raise HTTPException(
                status_code=400, 
                detail="target_value and timeframe are required parameters"
            )
        
        # Try to use new modular portfolio simulator first
        try:
            from services.portfolio_simulator import simulate_portfolio_workflow
            logger.info("Using new modular portfolio simulator")
            result = await simulate_portfolio_workflow(sim_input, db)
            
        except ImportError:
            logger.info("New modular simulator not available, trying standard import")
            try:
                # Try the old import structure
                from services.portfolio_simulator import simulate_portfolio
                logger.info("Using legacy portfolio simulator")
                result = await simulate_portfolio(sim_input, db)
            except ImportError:
                # Final fallback - create a basic response
                logger.error("No portfolio simulator available")
                raise HTTPException(
                    status_code=503,
                    detail="Portfolio simulation service is not available. Please ensure the portfolio simulator module is properly installed."
                )
        
        enhanced_used = result.get('enhanced_features_enabled', {}).get('ai_recommendations', False)
        logger.info(f"Simulation completed successfully. Enhanced: {enhanced_used}")
        
        return {
            **result,
            "simulator_type": "Enhanced Modular" if enhanced_used else "Standard",
            "api_endpoint": "/ai/simulate"
        }
        
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio simulation failed: {str(e)}")

@router.post("/analyze")
async def analyze_portfolio(portfolio_data: dict):
    """Analyze existing portfolio performance with news context"""
    if not AI_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Analysis service not available")
    
    try:
        logger.info("Starting portfolio performance analysis")
        result = await ai_service.analyze_portfolio_performance(portfolio_data)
        return {
            "status": "success",
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@router.post("/analyze-risk")
async def analyze_risk(portfolio_data: dict):
    """Analyze portfolio risk and allocation with market sentiment"""
    if not AI_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Analysis service not available")
    
    try:
        logger.info("Starting portfolio risk analysis")
        result = await ai_service.analyze_risk_allocation(portfolio_data)
        return {
            "status": "success",
            "risk_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Risk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")

@router.post("/explain-changes")
async def explain_changes(portfolio_data: dict, previous_data: Optional[dict] = None):
    """Explain portfolio changes over time with news context"""
    if not AI_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Analysis service not available")
    
    try:
        logger.info("Analyzing portfolio changes")
        result = await ai_service.explain_portfolio_changes(portfolio_data, previous_data)
        return {
            "status": "success",
            "explanation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Change explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Change explanation failed: {str(e)}")

# ENHANCED ENDPOINTS WITH MODULAR SIMULATOR INTEGRATION

@router.get("/simulation/{simulation_id}/chart-data")
async def get_simulation_chart_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Get chart data for a simulation (works with both old and new simulators)
    """
    try:
        logger.info(f"Getting chart data for simulation {simulation_id}")
        
        # Try new modular simulator first
        try:
            from services.portfolio_simulator import get_simulation_charts
            chart_data = await get_simulation_charts(simulation_id, db)
            return chart_data
            
        except ImportError:
            # Fallback: Get simulation from database and create basic chart data
            logger.info("Using fallback chart data generation")
            
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            results = simulation.results or {}
            
            return {
                "simulation_id": simulation_id,
                "basic_data": {
                    "starting_value": results.get("starting_value", 0),
                    "ending_value": results.get("end_value", 0),
                    "total_return": results.get("return", 0),
                    "stocks_picked": results.get("stocks_picked", [])
                },
                "note": "Enhanced chart data available when modular simulator is deployed"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chart data: {str(e)}")

@router.get("/simulation/{simulation_id}/shap-visualization")
async def get_simulation_shap_visualization(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Generate SHAP visualization for enhanced portfolio simulation
    """
    try:
        logger.info(f"Generating SHAP visualization for simulation {simulation_id}")
        
        # Try to use new modular simulator
        try:
            from services.portfolio_simulator.visualization_service import VisualizationService
            from services.portfolio_simulator.database_service import DatabaseService
            
            db_service = DatabaseService()
            viz_service = VisualizationService()
            
            # Get simulation data
            simulation = db_service.get_simulation(db, simulation_id)
            if not simulation:
                raise HTTPException(status_code=404, detail="Simulation not found")
            
            results = simulation.results or {}
            shap_explanation = results.get("shap_explanation")
            
            if not shap_explanation:
                raise HTTPException(
                    status_code=404, 
                    detail="SHAP explanation not available for this simulation"
                )
            
            # Generate visualization paths
            visualization_paths = results.get("visualization_paths", {})
            
            return {
                "status": "success",
                "visualization_paths": visualization_paths,
                "simulation_id": simulation_id,
                "has_shap": bool(shap_explanation),
                "message": "SHAP explanation data available"
            }
                
        except ImportError:
            logger.warning("Enhanced modular simulator not available")
            raise HTTPException(
                status_code=503,
                detail="SHAP visualizations require the enhanced modular simulator. This feature will be available when the enhanced simulator is deployed."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SHAP visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP visualization: {str(e)}")

@router.get("/simulation/{simulation_id}/news-analysis")
async def get_simulation_news_analysis(
    simulation_id: int, 
    db: Session = Depends(get_db)
):
    """
    Analyze a simulation with current news context
    """
    if not AI_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Analysis service not available")
    
    try:
        logger.info(f"Analyzing simulation {simulation_id} with current news context")
        
        # Get simulation from database
        simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        # Extract portfolio data
        results = simulation.results or {}
        stocks_picked = results.get("stocks_picked", [])
        
        if not stocks_picked:
            raise HTTPException(status_code=400, detail="No portfolio data found in simulation")
        
        # Create portfolio data for analysis
        portfolio_data = {"stocks": [stock.get("symbol", "") for stock in stocks_picked]}
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back=7)
        
        return {
            "status": "success",
            "simulation_id": simulation_id,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing simulation with news: {e}")
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")

@router.post("/analyze-with-news")
async def analyze_with_news_context(
    portfolio_data: dict, 
    days_back: int = Query(default=DEFAULT_DAYS_BACK, ge=MIN_DAYS_BACK, le=MAX_DAYS_BACK, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
):
    """
    Comprehensive portfolio analysis with news sentiment and market events context
    """
    if not AI_ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Analysis service not available")
    
    try:
        logger.info(f"Starting enhanced portfolio analysis with {days_back} days of news context")
        
        # Validate portfolio data
        if not portfolio_data or not isinstance(portfolio_data, dict):
            raise HTTPException(status_code=400, detail="Invalid portfolio data provided")
        
        # Get enhanced analysis with news context
        analysis = await ai_service.analyze_portfolio_with_context(portfolio_data, days_back)
        
        logger.info("Enhanced portfolio analysis with news context completed")
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
        logger.error(f"News-enhanced analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"News-enhanced analysis failed: {str(e)}")

@router.get("/health-check")
async def health_check():
    """Check if the AI analysis service and all dependencies are working"""
    try:
        logger.info("Running health check")
        
        health_status = {
            'status': 'healthy',
            'ai_analysis_configured': AI_ANALYSIS_AVAILABLE,
            'groq_configured': bool(ai_service.groq_api_key) if ai_service else False,
            'finnhub_configured': bool(FINNHUB_API_KEY),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check portfolio simulator availability
        try:
            from services.portfolio_simulator import PortfolioSimulatorService
            health_status['modular_simulator_available'] = True
            
            # Check for specific features
            try:
                from services.portfolio_simulator import get_simulation_charts
                health_status['chart_data_available'] = True
            except ImportError:
                health_status['chart_data_available'] = False
                
        except ImportError:
            health_status['modular_simulator_available'] = False
            health_status['chart_data_available'] = False
            
            # Check for legacy simulator
            try:
                from services.portfolio_simulator import simulate_portfolio
                health_status['legacy_simulator_available'] = True
            except ImportError:
                health_status['legacy_simulator_available'] = False
        
        # Test AI service if available
        if AI_ANALYSIS_AVAILABLE and ai_service:
            try:
                test_response = await ai_service._get_groq_response("Test message")
                health_status['ai_service'] = 'active' if test_response else 'inactive'
            except Exception as e:
                health_status['ai_service'] = 'error'
                health_status['ai_error'] = str(e)
        else:
            health_status['ai_service'] = 'unavailable'
        
        # Overall health determination
        critical_services = [
            health_status.get('ai_analysis_configured', False),
            health_status.get('modular_simulator_available', False) or health_status.get('legacy_simulator_available', False)
        ]
        
        if all(critical_services):
            health_status['overall'] = 'healthy'
        else:
            health_status['overall'] = 'degraded'
            health_status['status'] = 'degraded'
        
        logger.info(f"Health check complete: {health_status['overall']}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'overall': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/status")
async def get_service_status():
    """Quick service status check"""
    try:
        # Check simulator availability
        simulator_status = "none"
        try:
            from services.portfolio_simulator import PortfolioSimulatorService
            simulator_status = "modular"
        except ImportError:
            try:
                from services.portfolio_simulator import simulate_portfolio
                simulator_status = "legacy"
            except ImportError:
                pass
        
        return {
            'service': 'AI Analysis Router',
            'status': 'online',
            'ai_analysis_available': AI_ANALYSIS_AVAILABLE,
            'simulator_status': simulator_status,
            'endpoints_available': len([
                route for route in router.routes 
                if hasattr(route, 'path')
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