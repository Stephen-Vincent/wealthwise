# api/routers/shap_visualization.py

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database.db import get_db
from database import models
import os
import logging
from typing import Dict, Any
import io

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SHAP Visualization"])

# Import your actual VisualizationEngine
try:
    from ai_models.stock_model.explainable_ai.visualization import VisualizationEngine
    VISUALIZATION_AVAILABLE = True
    logger.info("✅ VisualizationEngine loaded successfully")
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logger.warning(f"⚠️ VisualizationEngine not available: {e}")

@router.get("/simulation/{simulation_id}/explanation")
async def get_shap_explanation_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get SHAP explanation data for frontend consumption
    """
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

        # Extract feature contributions/importance
        feature_contributions = shap_explanation.get("feature_contributions", {})
        feature_importance = shap_explanation.get("feature_importance", {})
        
        # Use whichever is available
        features_data = feature_contributions or feature_importance

        return {
            "simulation_id": simulation_id,
            "shap_data": {
                "portfolio_quality_score": shap_explanation.get("portfolio_quality_score"),
                "confidence_score": shap_explanation.get("confidence_score"),
                "feature_importance": features_data,
                "feature_contributions": feature_contributions,
                "human_readable_explanation": shap_explanation.get("human_readable_explanation", {}),
                "methodology": shap_explanation.get("methodology", "SHAP-based explainable AI"),
                "base_value": shap_explanation.get("base_value", []),
                "transparency_metrics": shap_explanation.get("transparency_metrics", {}),
            },
            "portfolio_info": {
                "stocks": [s.get("symbol") for s in results.get("stocks_picked", [])],
                "risk_score": results.get("risk_score"),
                "risk_label": results.get("risk_label"),
                "target_value": simulation.target_value,
                "final_value": results.get("end_value"),
            },
            "goal_analysis": results.get("goal_analysis", {}),
            "market_regime": results.get("market_regime", {}),
            "metadata": {
                "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
                "methodology": results.get("methodology", "Standard simulation"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SHAP explanation: {str(e)}")

@router.get("/simulation/{simulation_id}/shap-chart")
async def get_shap_waterfall_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Generate SHAP waterfall chart using VisualizationEngine
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation")

        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available")

        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Visualization engine not available")

        # Create directory if it doesn't exist
        os.makedirs("./static/visualizations", exist_ok=True)

        # Use your actual VisualizationEngine
        viz_engine = VisualizationEngine()
        chart_path = f"./static/visualizations/shap_waterfall_{simulation_id}.png"
        
        result = viz_engine.create_shap_waterfall_chart(shap_explanation, chart_path)
        
        if "saved" in result:
            return {
                "chart_url": f"/static/visualizations/shap_waterfall_{simulation_id}.png",
                "message": "SHAP waterfall chart generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {result}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SHAP waterfall chart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")

@router.get("/simulation/{simulation_id}/portfolio-chart")
async def get_portfolio_composition_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Generate portfolio composition chart using VisualizationEngine
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        stocks_picked = results.get("stocks_picked", [])

        if not stocks_picked:
            raise HTTPException(status_code=404, detail="No portfolio data available")

        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Visualization engine not available")

        # Create directory if it doesn't exist
        os.makedirs("./static/visualizations", exist_ok=True)

        # Extract stocks and weights
        stocks = [stock["symbol"] for stock in stocks_picked]
        weights = {stock["symbol"]: stock["allocation"] for stock in stocks_picked}

        # Use your actual VisualizationEngine
        viz_engine = VisualizationEngine()
        chart_path = f"./static/visualizations/portfolio_composition_{simulation_id}.png"
        
        result = viz_engine.create_portfolio_composition_chart(stocks, weights, chart_path)
        
        if "saved" in result:
            return {
                "chart_url": f"/static/visualizations/portfolio_composition_{simulation_id}.png",
                "message": "Portfolio composition chart generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {result}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating portfolio composition chart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")

@router.get("/simulation/{simulation_id}/risk-return-chart")
async def get_risk_return_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Generate risk-return scatter plot using VisualizationEngine
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        
        # Extract or estimate portfolio metrics
        portfolio_return = results.get("return", 0.08)  # Default 8%
        risk_score = simulation.risk_score or 50
        
        # Estimate volatility based on risk score (rough approximation)
        estimated_volatility = 0.05 + (risk_score / 100) * 0.20  # 5-25% volatility range
        
        portfolio_metrics = {
            "expected_return": portfolio_return,
            "volatility": estimated_volatility,
            "sharpe_ratio": portfolio_return / estimated_volatility if estimated_volatility > 0 else 0
        }

        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Visualization engine not available")

        # Create directory if it doesn't exist
        os.makedirs("./static/visualizations", exist_ok=True)

        # Use your actual VisualizationEngine
        viz_engine = VisualizationEngine()
        chart_path = f"./static/visualizations/risk_return_{simulation_id}.png"
        
        result = viz_engine.create_risk_return_scatter(portfolio_metrics, None, chart_path)
        
        if "saved" in result:
            return {
                "chart_url": f"/static/visualizations/risk_return_{simulation_id}.png",
                "message": "Risk-return chart generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {result}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating risk-return chart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")

@router.get("/simulation/{simulation_id}/market-regime-chart")
async def get_market_regime_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Generate market regime visualization using VisualizationEngine
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        market_regime = results.get("market_regime")

        if not market_regime:
            raise HTTPException(status_code=404, detail="Market regime data not available")

        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Visualization engine not available")

        # Create directory if it doesn't exist
        os.makedirs("./static/visualizations", exist_ok=True)

        # Use your actual VisualizationEngine
        viz_engine = VisualizationEngine()
        chart_path = f"./static/visualizations/market_regime_{simulation_id}.png"
        
        result = viz_engine.create_market_regime_visualization(market_regime, chart_path)
        
        if "saved" in result:
            return {
                "chart_url": f"/static/visualizations/market_regime_{simulation_id}.png",
                "message": "Market regime chart generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {result}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating market regime chart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")

@router.get("/simulation/{simulation_id}/all-charts")
async def generate_all_charts(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Generate all available charts for a simulation
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        if not VISUALIZATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Visualization engine not available")

        results = simulation.results or {}
        charts = {}
        errors = {}

        # Generate SHAP waterfall chart
        if results.get("shap_explanation"):
            try:
                chart_result = await get_shap_waterfall_chart(simulation_id, db)
                charts["shap_waterfall"] = chart_result["chart_url"]
            except Exception as e:
                errors["shap_waterfall"] = str(e)

        # Generate portfolio composition chart
        if results.get("stocks_picked"):
            try:
                chart_result = await get_portfolio_composition_chart(simulation_id, db)
                charts["portfolio_composition"] = chart_result["chart_url"]
            except Exception as e:
                errors["portfolio_composition"] = str(e)

        # Generate risk-return chart
        try:
            chart_result = await get_risk_return_chart(simulation_id, db)
            charts["risk_return"] = chart_result["chart_url"]
        except Exception as e:
            errors["risk_return"] = str(e)

        # Generate market regime chart
        if results.get("market_regime"):
            try:
                chart_result = await get_market_regime_chart(simulation_id, db)
                charts["market_regime"] = chart_result["chart_url"]
            except Exception as e:
                errors["market_regime"] = str(e)

        return {
            "simulation_id": simulation_id,
            "charts": charts,
            "errors": errors if errors else None,
            "total_charts": len(charts),
            "message": f"Generated {len(charts)} charts successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating all charts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate charts: {str(e)}")

@router.get("/simulation/{simulation_id}/visualization")
async def get_shap_visualization_image(
    simulation_id: int,
    db: Session = Depends(get_db),
    chart_type: str = "waterfall"
) -> Response:
    """
    Get visualization image directly (for backward compatibility)
    """
    try:
        if chart_type == "waterfall":
            result = await get_shap_waterfall_chart(simulation_id, db)
            chart_path = f"./static/visualizations/shap_waterfall_{simulation_id}.png"
        elif chart_type == "portfolio":
            result = await get_portfolio_composition_chart(simulation_id, db)
            chart_path = f"./static/visualizations/portfolio_composition_{simulation_id}.png"
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type. Use 'waterfall' or 'portfolio'")

        if os.path.exists(chart_path):
            return FileResponse(chart_path, media_type="image/png", filename=f"{chart_type}_{simulation_id}.png")
        else:
            raise HTTPException(status_code=404, detail="Chart file not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving visualization image: {e}")
        return await create_fallback_shap_visualization(simulation_id, chart_type, db)

@router.get("/simulation/{simulation_id}/chart-data")
async def get_shap_chart_data(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get structured chart data for frontend rendering
    """
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation", {})

        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available")

        # Get feature importance/contributions
        feature_importance = (
            shap_explanation.get("feature_contributions") or 
            shap_explanation.get("feature_importance") or 
            {}
        )

        chart_data = {
            "portfolio_score": {
                "value": shap_explanation.get("portfolio_quality_score", 0),
                "max_value": 100,
                "label": "Portfolio Quality Score",
            },
            "feature_importance": {
                "labels": list(feature_importance.keys()),
                "values": list(feature_importance.values()),
                "colors": generate_chart_colors(len(feature_importance)),
            },
            "explanations": shap_explanation.get("human_readable_explanation", {}),
            "metadata": {
                "methodology": shap_explanation.get("methodology", "SHAP Explainable AI"),
                "confidence": shap_explanation.get("confidence_score", 0),
                "base_value": shap_explanation.get("base_value", []),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
            },
        }

        return chart_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error formatting SHAP chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to format chart data: {str(e)}")

# Helper functions
async def create_fallback_shap_visualization(simulation_id: int, chart_type: str, db: Session) -> Response:
    """
    Create a fallback visualization when VisualizationEngine is not available
    """
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
        fig, ax = plt.subplots(figsize=(10, 6))

        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation", {})
        
        feature_importance = (
            shap_explanation.get("feature_contributions") or 
            shap_explanation.get("feature_importance") or 
            {}
        )

        if feature_importance:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax.barh(features, values, color=colors, alpha=0.7)
            ax.set_title(f'SHAP Feature Importance - Simulation {simulation_id}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=12)
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
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'SHAP Analysis - Simulation {simulation_id}')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error creating fallback visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to create visualization")

def generate_chart_colors(num_colors: int) -> list:
    """Generate a list of colors for charts"""
    base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if num_colors <= len(base):
        return base[:num_colors]
    
    # Generate additional colors if needed
    import colorsys
    extra = []
    for i in range(num_colors - len(base)):
        hue = i / max(1, (num_colors - len(base)))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        extra.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
    return base + extra