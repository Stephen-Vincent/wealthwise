from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database.db import get_db
from database import models
import os
import logging
from typing import Dict, Any, Optional
import io
import base64
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SHAP Visualization"])

# Try to import from new modular portfolio simulator first
try:
    from services.portfolio_simulator import (
        VisualizationService,
        ChartDataGenerator,
        get_simulation_charts
    )
    MODULAR_SIMULATOR_AVAILABLE = True
    logger.info("New modular portfolio simulator visualization functions loaded successfully")
except ImportError:
    MODULAR_SIMULATOR_AVAILABLE = False
    logger.info("New modular simulator not available, will use fallback methods")

# Try legacy portfolio simulator imports
try:
    from services.portfolio_simulator import (
        get_visualization_engine,
        get_simulation_visualizations,
        regenerate_shap_visualization,
        create_custom_visualization
    )
    LEGACY_SIMULATOR_AVAILABLE = True
    logger.info("Legacy portfolio simulator functions loaded successfully")
except ImportError:
    LEGACY_SIMULATOR_AVAILABLE = False
    logger.warning("Legacy portfolio simulator not available")

# Check overall availability
PORTFOLIO_SIMULATOR_AVAILABLE = MODULAR_SIMULATOR_AVAILABLE or LEGACY_SIMULATOR_AVAILABLE

@router.get("/simulation/{simulation_id}/explanation")
async def get_shap_explanation_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get SHAP explanation data for frontend consumption"""
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation")

        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available for this simulation")

        # Extract feature contributions safely
        feature_contributions = shap_explanation.get("feature_contributions", {})
        
        return {
            "simulation_id": simulation_id,
            "shap_data": {
                "portfolio_quality_score": shap_explanation.get("portfolio_quality_score", 0),
                "confidence_score": shap_explanation.get("confidence_score", 0),
                "feature_contributions": feature_contributions,
                "human_readable_explanation": shap_explanation.get("human_readable_explanation", {}),
                "methodology": shap_explanation.get("methodology", "SHAP-based explainable AI"),
                "base_value": shap_explanation.get("base_value", 50.0),
                "transparency_metrics": shap_explanation.get("transparency_metrics", {}),
            },
            "portfolio_info": {
                "stocks": [s.get("symbol", "") for s in results.get("stocks_picked", [])],
                "risk_score": results.get("risk_score", 0),
                "risk_label": results.get("risk_label", "Unknown"),
                "target_value": simulation.target_value,
                "final_value": results.get("end_value", 0),
                "target_achieved": results.get("target_reached", False),
            },
            "goal_analysis": results.get("goal_analysis", {}),
            "market_regime": results.get("market_regime", {}),
            "metadata": {
                "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
                "methodology": results.get("methodology", "Standard simulation"),
                "has_visualizations": bool(results.get("visualization_paths")),
                "simulator_type": "modular" if MODULAR_SIMULATOR_AVAILABLE else "legacy" if LEGACY_SIMULATOR_AVAILABLE else "none"
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SHAP explanation: {str(e)}")

@router.get("/simulation/{simulation_id}/chart-data")
async def get_chart_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get raw chart data for React components using the best available method"""
    try:
        # Try new modular simulator first
        if MODULAR_SIMULATOR_AVAILABLE:
            try:
                return await get_simulation_charts(simulation_id, db)
            except Exception as e:
                logger.warning(f"Modular simulator chart data failed: {e}, falling back to legacy method")
        
        # Fall back to legacy chart data generation
        return await get_legacy_chart_data(simulation_id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_legacy_chart_data(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Legacy chart data generation method"""
    simulation = db.query(models.Simulation).filter(
        models.Simulation.id == simulation_id
    ).first()

    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")

    results = simulation.results or {}
    shap_explanation = results.get("shap_explanation", {})
    stocks_picked = results.get("stocks_picked", [])
    goal_analysis = results.get("goal_analysis", {})
    market_regime = results.get("market_regime", {})

    # Prepare structured chart data for React components
    chart_data = {
        "portfolio_composition": [
            {
                "symbol": stock.get("symbol", ""),
                "name": stock.get("name", stock.get("symbol", "")),
                "allocation": float(stock.get("allocation", 0)),
                "sector": stock.get("sector", "Unknown"),
                "value": float(stock.get("allocation", 0)) * float(results.get("starting_value", 100000))
            }
            for stock in stocks_picked
        ],
        "factor_importance": [
            {
                "factor": factor,
                "importance": float(importance),
                "impact": "positive" if float(importance) > 0 else "negative",
                "description": shap_explanation.get("human_readable_explanation", {}).get(factor, f"Impact of {factor}")
            }
            for factor, importance in shap_explanation.get("feature_contributions", {}).items()
        ],
        "shap_waterfall": {
            "base_value": float(shap_explanation.get("base_value", 50.0)),
            "final_value": float(shap_explanation.get("portfolio_quality_score", 0)),
            "contributions": [
                {
                    "feature": feature,
                    "contribution": float(contrib),
                    "cumulative": float(shap_explanation.get("base_value", 50.0)) + sum(
                        float(c) for f, c in list(shap_explanation.get("feature_contributions", {}).items())[:i+1]
                    )
                }
                for i, (feature, contrib) in enumerate(shap_explanation.get("feature_contributions", {}).items())
            ]
        },
        "risk_return_analysis": {
            "portfolio": {
                "expected_return": float(results.get("return", 0.08)),
                "volatility": estimate_volatility_from_risk_score(results.get("risk_score", 50)),
                "sharpe_ratio": calculate_sharpe_ratio(
                    float(results.get("return", 0.08)),
                    estimate_volatility_from_risk_score(results.get("risk_score", 50))
                ),
                "label": "Your Portfolio"
            },
            "benchmark": generate_benchmark_data(),
            "efficient_frontier": generate_efficient_frontier_data()
        },
        "market_regime": {
            "current_regime": market_regime.get("regime", "Normal"),
            "confidence": float(market_regime.get("confidence", 0.5)),
            "characteristics": market_regime.get("characteristics", {}),
            "historical_performance": market_regime.get("historical_performance", {}),
            "regime_probabilities": market_regime.get("regime_probabilities", {
                "Bull Market": 0.3,
                "Bear Market": 0.2, 
                "Normal": 0.4,
                "Volatile": 0.1
            })
        },
        "goal_analysis": {
            "target_value": float(simulation.target_value or 0),
            "current_value": float(results.get("end_value", 0)),
            "starting_value": float(results.get("starting_value", 0)),
            "progress_percentage": calculate_progress_percentage(
                float(results.get("starting_value", 0)),
                float(results.get("end_value", 0)),
                float(simulation.target_value or 0)
            ),
            "target_achieved": bool(results.get("target_reached", False)),
            "time_horizon": getattr(simulation, 'timeframe', 5),
            "monthly_required": goal_analysis.get("monthly_required", 0),
            "probability_of_success": goal_analysis.get("success_probability", 0.5)
        }
    }

    return {
        "success": True,
        "simulation_id": simulation_id,
        "chart_data": chart_data,
        "metadata": {
            "generated_at": simulation.created_at.isoformat() if simulation.created_at else None,
            "methodology": results.get("methodology", "Standard simulation"),
            "wealthwise_enhanced": bool(results.get("wealthwise_enhanced", False)),
            "has_shap_explanation": bool(shap_explanation),
            "portfolio_size": len(stocks_picked),
            "data_source": "legacy_method"
        }
    }

@router.get("/simulation/{simulation_id}/enhanced-data")
async def get_enhanced_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get enhanced portfolio data with historical performance"""
    try:
        # Try modular simulator first
        if MODULAR_SIMULATOR_AVAILABLE:
            try:
                from services.portfolio_simulator.database_service import DatabaseService
                db_service = DatabaseService()
                enhanced_data = await db_service.get_enhanced_portfolio_data(simulation_id, db)
                if enhanced_data.get("success"):
                    return enhanced_data
            except Exception as e:
                logger.warning(f"Modular enhanced data failed: {e}, using fallback")
        
        # Fallback to legacy enhanced data generation
        return await get_legacy_enhanced_data(simulation_id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_legacy_enhanced_data(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Legacy enhanced data generation"""
    simulation = db.query(models.Simulation).filter(
        models.Simulation.id == simulation_id
    ).first()

    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")

    results = simulation.results or {}
    stocks_picked = results.get("stocks_picked", [])
    
    # Generate enhanced timeline data
    enhanced_data = {
        "portfolio_timeline": generate_portfolio_timeline(simulation, results),
        "individual_stock_performance": [
            {
                "symbol": stock.get("symbol", ""),
                "name": stock.get("name", ""),
                "allocation": float(stock.get("allocation", 0)),
                "historical_returns": generate_mock_returns(stock.get("symbol", "")),
                "risk_metrics": {
                    "beta": generate_mock_beta(),
                    "volatility": generate_mock_volatility(),
                    "max_drawdown": generate_mock_drawdown()
                },
                "sector": stock.get("sector", "Unknown"),
                "market_cap": stock.get("market_cap", "Unknown")
            }
            for stock in stocks_picked
        ],
        "correlation_matrix": generate_correlation_matrix([s.get("symbol", "") for s in stocks_picked]),
        "sector_allocation": calculate_sector_allocation(stocks_picked),
        "risk_decomposition": {
            "systematic_risk": 0.6,
            "idiosyncratic_risk": 0.4,
            "risk_attribution": [
                {
                    "stock": stock.get("symbol", ""),
                    "contribution": float(stock.get("allocation", 0)) * generate_mock_risk_contribution()
                }
                for stock in stocks_picked
            ]
        },
        "performance_attribution": {
            "asset_allocation": 0.4,
            "security_selection": 0.3,
            "interaction": 0.2,
            "timing": 0.1
        }
    }

    return {
        "success": True,
        "simulation_id": simulation_id,
        "enhanced_data": enhanced_data,
        "metadata": {
            "data_points": len(enhanced_data["portfolio_timeline"]),
            "stocks_analyzed": len(stocks_picked),
            "analysis_date": simulation.created_at.isoformat() if simulation.created_at else None,
            "data_source": "legacy_enhanced"
        }
    }

@router.get("/simulation/{simulation_id}/visualizations")
async def get_all_simulation_visualizations(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get all available visualizations for a simulation"""
    try:
        # Try new modular simulator first
        if MODULAR_SIMULATOR_AVAILABLE:
            try:
                viz_service = VisualizationService()
                # Implementation would depend on the actual VisualizationService API
                # For now, return metadata about available visualizations
                simulation = db.query(models.Simulation).filter(
                    models.Simulation.id == simulation_id
                ).first()
                
                if not simulation:
                    raise HTTPException(status_code=404, detail="Simulation not found")
                
                results = simulation.results or {}
                visualization_paths = results.get("visualization_paths", {})
                
                return {
                    "simulation_id": simulation_id,
                    "visualizations": {
                        chart_type: {
                            "exists": True,
                            "path": path,
                            "type": "modular_generated"
                        }
                        for chart_type, path in visualization_paths.items()
                    },
                    "available_charts": list(visualization_paths.keys()),
                    "service_status": "modular_available"
                }
            except Exception as e:
                logger.warning(f"Modular visualization service failed: {e}")
        
        # Try legacy simulator
        if LEGACY_SIMULATOR_AVAILABLE:
            try:
                result = await get_simulation_visualizations(simulation_id, db)
                if "error" not in result:
                    return result
            except Exception as e:
                logger.warning(f"Legacy visualization service failed: {e}")
        
        # Fallback
        return await get_fallback_visualizations(simulation_id, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation visualizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/simulation/{simulation_id}/chart/{chart_type}")
async def serve_visualization_file(
    simulation_id: int, 
    chart_type: str, 
    db: Session = Depends(get_db)
):
    """Serve a specific visualization file"""
    try:
        # Check if we have any visualization capabilities
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            return await create_fallback_visualization(simulation_id, chart_type, db)
        
        # Try to get existing visualization file
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        results = simulation.results or {}
        visualization_paths = results.get("visualization_paths", {})
        
        # Check if file exists
        if chart_type in visualization_paths:
            file_path = Path(visualization_paths[chart_type])
            if file_path.exists():
                logger.info(f"Serving existing visualization: {file_path}")
                return FileResponse(
                    path=str(file_path),
                    media_type="image/png",
                    filename=f"simulation_{simulation_id}_{chart_type}.png"
                )
        
        # Generate visualization if not found
        return await create_fallback_visualization(simulation_id, chart_type, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving visualization: {e}")
        return await create_fallback_visualization(simulation_id, chart_type, db)

@router.get("/simulation/{simulation_id}/chart/{chart_type}/image")
async def get_chart_as_base64(
    simulation_id: int, 
    chart_type: str, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Serve visualization as base64 encoded image data"""
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        results = simulation.results or {}
        visualization_paths = results.get("visualization_paths", {})
        
        # Try to get existing file first
        if chart_type in visualization_paths:
            file_path = Path(visualization_paths[chart_type])
            
            if file_path.exists():
                try:
                    with open(file_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                    
                    return {
                        "success": True,
                        "chart_type": chart_type,
                        "image_data": f"data:image/png;base64,{image_data}",
                        "simulation_id": simulation_id,
                        "method": "existing_file"
                    }
                except Exception as file_error:
                    logger.warning(f"Failed to read existing file: {file_error}")
        
        # Generate chart in memory if file doesn't exist
        return await generate_chart_as_base64(simulation_id, chart_type, db)
        
    except Exception as e:
        logger.error(f"Error serving chart as base64: {e}")
        return await generate_chart_as_base64(simulation_id, chart_type, db)

async def generate_chart_as_base64(simulation_id: int, chart_type: str, db: Session) -> Dict[str, Any]:
    """Generate chart in memory and return as base64"""
    try:
        # Use the fallback visualization method that creates charts in memory
        response = await create_fallback_visualization(simulation_id, chart_type, db)
        
        # Convert the response content to base64
        if hasattr(response, 'body'):
            image_bytes = response.body
        else:
            # Extract bytes from Response object
            image_bytes = response._content if hasattr(response, '_content') else b''
        
        if not image_bytes:
            raise Exception("No image data generated")
        
        image_data = base64.b64encode(image_bytes).decode()
        
        return {
            "success": True,
            "chart_type": chart_type,
            "image_data": f"data:image/png;base64,{image_data}",
            "simulation_id": simulation_id,
            "method": "generated_in_memory"
        }
        
    except Exception as e:
        logger.error(f"Error generating chart as base64: {e}")
        return {
            "success": False,
            "error": str(e),
            "chart_type": chart_type,
            "simulation_id": simulation_id
        }

@router.get("/simulation/{simulation_id}/charts/all")
async def get_all_charts_as_base64(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get all available charts as base64 data"""
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        charts = {}
        errors = []
        
        # Define available chart types
        chart_types = ["shap_explanation", "portfolio_composition", "risk_return_analysis", "factor_importance", "market_regime"]
        
        for chart_type in chart_types:
            try:
                chart_result = await get_chart_as_base64(simulation_id, chart_type, db)
                if chart_result.get("success"):
                    charts[chart_type] = chart_result["image_data"]
                else:
                    errors.append(f"{chart_type}: {chart_result.get('error', 'Unknown error')}")
            except Exception as e:
                errors.append(f"{chart_type}: {str(e)}")
        
        return {
            "simulation_id": simulation_id,
            "charts": charts,
            "chart_count": len(charts),
            "errors": errors if errors else None,
            "success": len(charts) > 0,
            "available_simulators": {
                "modular": MODULAR_SIMULATOR_AVAILABLE,
                "legacy": LEGACY_SIMULATOR_AVAILABLE
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting all charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulation/{simulation_id}/regenerate-shap")
async def regenerate_shap_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Regenerate SHAP visualization"""
    try:
        # Try legacy simulator first (if it has this function)
        if LEGACY_SIMULATOR_AVAILABLE:
            try:
                result = await regenerate_shap_visualization(simulation_id, db)
                return result
            except Exception as e:
                logger.warning(f"Legacy SHAP regeneration failed: {e}")
        
        # Fallback: generate new SHAP chart
        chart_result = await generate_chart_as_base64(simulation_id, "shap_explanation", db)
        
        if chart_result.get("success"):
            return {
                "success": True,
                "simulation_id": simulation_id,
                "message": "SHAP visualization regenerated successfully",
                "chart_data": chart_result["image_data"],
                "method": "fallback_generation"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to regenerate SHAP visualization")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating SHAP visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check the health of visualization services"""
    try:
        health_status = {
            "service": "shap_visualization",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "simulators": {
                "modular_available": MODULAR_SIMULATOR_AVAILABLE,
                "legacy_available": LEGACY_SIMULATOR_AVAILABLE,
                "any_available": PORTFOLIO_SIMULATOR_AVAILABLE
            }
        }
        
        # Test matplotlib availability for fallback charts
        try:
            import matplotlib.pyplot as plt
            health_status["matplotlib_available"] = True
        except ImportError:
            health_status["matplotlib_available"] = False
            health_status["status"] = "degraded"
        
        # Check visualization directory
        viz_dir = Path("static/visualizations")
        health_status["visualization_directory"] = {
            "exists": viz_dir.exists(),
            "writable": viz_dir.is_dir() and os.access(viz_dir, os.W_OK) if viz_dir.exists() else False
        }
        
        if not health_status["visualization_directory"]["writable"]:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "service": "shap_visualization",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# [Rest of the helper functions remain the same...]
def estimate_volatility_from_risk_score(risk_score: int) -> float:
    """Estimate portfolio volatility based on risk score"""
    return 0.05 + (risk_score / 100) * 0.25

def calculate_sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float = 0.03) -> float:
    """Calculate Sharpe ratio"""
    if volatility == 0:
        return 0
    return (expected_return - risk_free_rate) / volatility

def calculate_progress_percentage(start_value: float, current_value: float, target_value: float) -> float:
    """Calculate progress towards goal as percentage"""
    if target_value <= start_value:
        return 100.0 if current_value >= target_value else 0.0
    
    progress = (current_value - start_value) / (target_value - start_value) * 100
    return max(0.0, min(100.0, progress))

def generate_benchmark_data() -> list:
    """Generate benchmark comparison data"""
    return [
        {"name": "S&P 500", "return": 0.10, "volatility": 0.16, "sharpe": 0.44},
        {"name": "NASDAQ", "return": 0.12, "volatility": 0.20, "sharpe": 0.45},
        {"name": "Total Market", "return": 0.09, "volatility": 0.15, "sharpe": 0.40},
    ]

def generate_efficient_frontier_data() -> list:
    """Generate efficient frontier data points"""
    return [
        {"return": 0.04 + i * 0.01, "volatility": 0.08 + i * 0.02}
        for i in range(15)
    ]

def generate_portfolio_timeline(simulation, results: dict) -> list:
    """Generate portfolio performance timeline"""
    start_value = float(results.get("starting_value", 100000))
    end_value = float(results.get("end_value", start_value))
    time_horizon = getattr(simulation, 'timeframe', 5)
    
    timeline = []
    for month in range(time_horizon * 12 + 1):
        # Simple linear interpolation with some noise
        progress = month / (time_horizon * 12)
        base_value = start_value + (end_value - start_value) * progress
        
        # Add some realistic market volatility
        noise = np.random.normal(0, base_value * 0.02) if month > 0 else 0
        value = max(base_value + noise, start_value * 0.5)  # Prevent unrealistic losses
        
        timeline.append({
            "date": f"2024-{(month % 12) + 1:02d}-01",
            "value": round(value, 2),
            "cumulative_return": round((value - start_value) / start_value * 100, 2)
        })
    
    return timeline

def generate_mock_returns(symbol: str) -> list:
    """Generate mock historical returns for a stock"""
    np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
    returns = np.random.normal(0.08, 0.15, 60)  # 5 years of monthly returns
    return [round(r, 4) for r in returns.tolist()]

def generate_mock_beta() -> float:
    return round(np.random.normal(1.0, 0.3), 2)

def generate_mock_volatility() -> float:
    return round(np.random.uniform(0.15, 0.35), 3)

def generate_mock_drawdown() -> float:
    return round(np.random.uniform(-0.15, -0.05), 3)

def generate_mock_risk_contribution() -> float:
    return round(np.random.uniform(0.8, 1.2), 3)

def generate_correlation_matrix(symbols: list) -> dict:
    """Generate correlation matrix for stocks"""
    n = len(symbols)
    if n == 0:
        return {}
    
    # Generate symmetric correlation matrix
    correlations = {}
    for i, sym1 in enumerate(symbols):
        correlations[sym1] = {}
        for j, sym2 in enumerate(symbols):
            if i == j:
                correlations[sym1][sym2] = 1.0
            elif sym2 in correlations:
                correlations[sym1][sym2] = correlations[sym2][sym1]
            else:
                # Generate realistic correlation (typically between 0.2-0.8 for stocks)
                corr = round(np.random.uniform(0.2, 0.8), 3)
                correlations[sym1][sym2] = corr
    
    return correlations

def calculate_sector_allocation(stocks_picked: list) -> dict:
    """Calculate sector allocation from stocks"""
    sector_allocation = {}
    
    for stock in stocks_picked:
        sector = stock.get("sector", "Unknown")
        allocation = float(stock.get("allocation", 0))
        
        if sector in sector_allocation:
            sector_allocation[sector] += allocation
        else:
            sector_allocation[sector] = allocation
    
    return sector_allocation

async def get_fallback_visualizations(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Fallback visualization metadata when service unavailable"""
    return {
        "simulation_id": simulation_id,
        "visualizations": {
            "shap_explanation": {"exists": True, "path": "", "type": "generated"},
            "portfolio_composition": {"exists": True, "path": "", "type": "generated"},
            "risk_return_analysis": {"exists": True, "path": "", "type": "generated"},
            "factor_importance": {"exists": True, "path": "", "type": "generated"},
            "market_regime": {"exists": True, "path": "", "type": "generated"}
        },
        "available_charts": ["shap_explanation", "portfolio_composition", "risk_return_analysis", "factor_importance", "market_regime"],
        "service_status": "fallback_mode"
    }

# Fallback visualization function using matplotlib
async def create_fallback_visualization(simulation_id: int, chart_type: str, db: Session) -> Response:
    """Create a fallback visualization when main visualization engine is not available"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        simulation = db.query(models.Simulation).filter(
           models.Simulation.id == simulation_id
       ).first()

        if not simulation:
           raise HTTPException(status_code=404, detail="Simulation not found")

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))

        results = simulation.results or {}
       
        if chart_type in ["shap_explanation", "shap_waterfall"]:
           # Create SHAP waterfall fallback
           shap_explanation = results.get("shap_explanation", {})
           feature_contributions = shap_explanation.get("feature_contributions", {})
           
           if feature_contributions:
               features = list(feature_contributions.keys())
               values = [float(v) for v in feature_contributions.values()]
               colors = ['green' if v > 0 else 'red' for v in values]
               
               bars = ax.barh(features, values, color=colors, alpha=0.7)
               ax.set_title(f'SHAP Feature Importance - Simulation {simulation_id}', 
                           fontsize=16, fontweight='bold')
               ax.set_xlabel('Impact on Portfolio Quality', fontsize=14)
               ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
               
               # Add value labels
               for bar, value in zip(bars, values):
                   width = bar.get_width()
                   ax.text(width + (0.01 if width >= 0 else -0.01),
                          bar.get_y() + bar.get_height()/2,
                          f'{value:.3f}', ha='left' if width >= 0 else 'right',
                          va='center', fontweight='bold')
           else:
               ax.text(0.5, 0.5, 'SHAP explanation data not available', 
                      ha='center', va='center', transform=ax.transAxes, fontsize=16)
               ax.set_title(f'SHAP Analysis - Simulation {simulation_id}')
               
        elif chart_type == "portfolio_composition":
           # Create portfolio composition fallback
           stocks_picked = results.get("stocks_picked", [])
           
           if stocks_picked:
               symbols = [stock.get("symbol", "") for stock in stocks_picked]
               allocations = [float(stock.get("allocation", 0)) for stock in stocks_picked]
               
               colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
               wedges, texts, autotexts = ax.pie(allocations, labels=symbols, autopct='%1.1f%%', 
                                                colors=colors, startangle=90)
               ax.set_title(f'Portfolio Composition - Simulation {simulation_id}', 
                           fontsize=16, fontweight='bold')
           else:
               ax.text(0.5, 0.5, 'Portfolio data not available', 
                      ha='center', va='center', transform=ax.transAxes, fontsize=16)
               ax.set_title(f'Portfolio Composition - Simulation {simulation_id}')
               
        elif chart_type == "risk_return_analysis":
           # Create risk-return scatter plot
           portfolio_return = float(results.get("return", 0.08))
           risk_score = results.get("risk_score", 50)
           volatility = estimate_volatility_from_risk_score(risk_score)
           
           # Plot portfolio point
           ax.scatter(volatility, portfolio_return, s=200, c='red', alpha=0.8, 
                     label='Your Portfolio', edgecolors='black', linewidth=2)
           
           # Add benchmark points
           benchmarks = generate_benchmark_data()
           for benchmark in benchmarks:
               ax.scatter(benchmark["volatility"], benchmark["return"], 
                         s=100, alpha=0.6, label=benchmark["name"])
           
           ax.set_xlabel('Volatility (Risk)', fontsize=14)
           ax.set_ylabel('Expected Return', fontsize=14)
           ax.set_title(f'Risk-Return Analysis - Simulation {simulation_id}', 
                       fontsize=16, fontweight='bold')
           ax.legend()
           ax.grid(True, alpha=0.3)
           
        elif chart_type == "factor_importance":
           # Create factor importance chart
           shap_explanation = results.get("shap_explanation", {})
           feature_contributions = shap_explanation.get("feature_contributions", {})
           
           if feature_contributions:
               factors = list(feature_contributions.keys())
               importance = [abs(float(v)) for v in feature_contributions.values()]
               colors = plt.cm.viridis(np.linspace(0, 1, len(factors)))
               
               bars = ax.bar(factors, importance, color=colors, alpha=0.8)
               ax.set_title(f'Factor Importance - Simulation {simulation_id}', 
                           fontsize=16, fontweight='bold')
               ax.set_ylabel('Absolute Importance', fontsize=14)
               plt.xticks(rotation=45, ha='right')
               
               # Add value labels
               for bar, value in zip(bars, importance):
                   height = bar.get_height()
                   ax.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
           else:
               ax.text(0.5, 0.5, 'Factor importance data not available', 
                      ha='center', va='center', transform=ax.transAxes, fontsize=16)
               ax.set_title(f'Factor Importance - Simulation {simulation_id}')
               
        elif chart_type == "market_regime":
           # Create market regime visualization
           market_regime = results.get("market_regime", {})
           
           if market_regime:
               regime_probs = market_regime.get("regime_probabilities", {
                   "Bull Market": 0.3, "Bear Market": 0.2, "Normal": 0.4, "Volatile": 0.1
               })
               
               regimes = list(regime_probs.keys())
               probabilities = list(regime_probs.values())
               colors = ['green', 'red', 'blue', 'orange'][:len(regimes)]
               
               bars = ax.bar(regimes, probabilities, color=colors, alpha=0.7)
               ax.set_title(f'Market Regime Analysis - Simulation {simulation_id}', 
                           fontsize=16, fontweight='bold')
               ax.set_ylabel('Probability', fontsize=14)
               ax.set_ylim(0, 1)
               
               # Highlight current regime
               current_regime = market_regime.get("regime", "Normal")
               if current_regime in regimes:
                   idx = regimes.index(current_regime)
                   bars[idx].set_edgecolor('black')
                   bars[idx].set_linewidth(3)
               
               # Add value labels
               for bar, value in zip(bars, probabilities):
                   height = bar.get_height()
                   ax.text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
           else:
               ax.text(0.5, 0.5, 'Market regime data not available', 
                      ha='center', va='center', transform=ax.transAxes, fontsize=16)
               ax.set_title(f'Market Regime Analysis - Simulation {simulation_id}')
               
        else:
           ax.text(0.5, 0.5, f'Visualization for {chart_type} not available', 
                  ha='center', va='center', transform=ax.transAxes, fontsize=16)
           ax.set_title(f'{chart_type.title()} - Simulation {simulation_id}')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close()

        return Response(content=buf.getvalue(), media_type="image/png")
       
    except Exception as e:
        logger.error(f"Error creating fallback visualization: {e}")
       # Create minimal error image
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error generating {chart_type}\nfor simulation {simulation_id}', 
                  ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Visualization Error')
           
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
           
        return Response(content=buf.getvalue(), media_type="image/png")
    except:
           raise HTTPException(status_code=500, detail="Failed to create visualization")

def generate_chart_colors(num_colors: int) -> list:
   """Generate a list of colors for charts"""
   base_colors = [
       '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
   ]
   
   if num_colors <= len(base_colors):
       return base_colors[:num_colors]
   
   # Generate additional colors if needed
   import colorsys
   additional_colors = []
   for i in range(num_colors - len(base_colors)):
       hue = i / max(1, (num_colors - len(base_colors)))
       r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
       additional_colors.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
   
   return base_colors + additional_colors

@router.get("/simulation/{simulation_id}/chart-data-legacy")
async def get_chart_data_for_frontend(
   simulation_id: int,
   db: Session = Depends(get_db)
) -> Dict[str, Any]:
   """Get structured chart data for frontend rendering (legacy format)"""
   try:
       simulation = db.query(models.Simulation).filter(
           models.Simulation.id == simulation_id
       ).first()

       if not simulation:
           raise HTTPException(status_code=404, detail="Simulation not found")

       results = simulation.results or {}
       shap_explanation = results.get("shap_explanation", {})

       # Extract feature contributions safely
       feature_contributions = shap_explanation.get("feature_contributions", {})
       
       # Prepare chart data
       chart_data = {
           "portfolio_score": {
               "value": float(shap_explanation.get("portfolio_quality_score", 0)),
               "max_value": 100,
               "label": "Portfolio Quality Score",
           },
           "feature_importance": {
               "labels": list(feature_contributions.keys()),
               "values": [float(v) for v in feature_contributions.values()],
               "colors": generate_chart_colors(len(feature_contributions)),
           },
           "explanations": shap_explanation.get("human_readable_explanation", {}),
           "portfolio_data": {
               "stocks": [
                   {
                       "symbol": stock.get("symbol", ""),
                       "allocation": float(stock.get("allocation", 0)),
                       "name": stock.get("name", "")
                   }
                   for stock in results.get("stocks_picked", [])
               ],
               "performance": {
                   "starting_value": float(results.get("starting_value", 0)),
                   "ending_value": float(results.get("end_value", 0)),
                   "total_return": float(results.get("return", 0)),
                   "target_achieved": bool(results.get("target_reached", False))
               }
           },
           "metadata": {
               "methodology": shap_explanation.get("methodology", "SHAP Explainable AI"),
               "confidence": float(shap_explanation.get("confidence_score", 0)),
               "base_value": float(shap_explanation.get("base_value", 50.0)),
               "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
               "wealthwise_enhanced": bool(results.get("wealthwise_enhanced", False)),
               "simulator_available": {
                   "modular": MODULAR_SIMULATOR_AVAILABLE,
                   "legacy": LEGACY_SIMULATOR_AVAILABLE
               }
           },
       }

       return chart_data

   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error formatting chart data: {e}")
       raise HTTPException(status_code=500, detail=f"Failed to format chart data: {str(e)}")