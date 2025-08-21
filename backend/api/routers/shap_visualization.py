# api/routers/shap_visualization.py

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

logger = logging.getLogger(__name__)
router = APIRouter(tags=["SHAP Visualization"])

# Import from your portfolio simulator service
try:
    from services.portfolio_simulator import (
        get_visualization_engine,
        get_simulation_visualizations,
        regenerate_shap_visualization,
        create_custom_visualization,
        get_simulation_chart_data,
        get_enhanced_portfolio_data
    )
    PORTFOLIO_SIMULATOR_AVAILABLE = True
    logger.info("Portfolio simulator visualization functions loaded successfully")
except ImportError as e:
    PORTFOLIO_SIMULATOR_AVAILABLE = False
    logger.warning(f"Portfolio simulator not available: {e}")

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
    """Get raw chart data for React components"""
    try:
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Portfolio simulator service not available")
        
        result = await get_simulation_chart_data(simulation_id, db)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/simulation/{simulation_id}/enhanced-data")
async def get_enhanced_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get enhanced portfolio data with historical performance"""
    try:
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Portfolio simulator service not available")
        
        result = await get_enhanced_portfolio_data(simulation_id, db)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/simulation/{simulation_id}/visualizations")
async def get_all_simulation_visualizations(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get all available visualizations for a simulation using the portfolio simulator service"""
    try:
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Portfolio simulator service not available")
        
        result = await get_simulation_visualizations(simulation_id, db)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
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
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            return await create_fallback_visualization(simulation_id, chart_type, db)
        
        # Get visualization metadata
        viz_data = await get_simulation_visualizations(simulation_id, db)
        
        if "error" in viz_data:
            raise HTTPException(status_code=404, detail=viz_data["error"])
        
        visualizations = viz_data.get("visualizations", {})
        
        if chart_type not in visualizations:
            raise HTTPException(status_code=404, detail=f"Chart type '{chart_type}' not found")
        
        viz_info = visualizations[chart_type]
        
        if not viz_info.get("exists", False):
            return await create_fallback_visualization(simulation_id, chart_type, db)
        
        file_path = Path(viz_info["path"])
        if not file_path.exists():
            return await create_fallback_visualization(simulation_id, chart_type, db)
        
        logger.info(f"Serving visualization: {file_path}")
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            filename=f"simulation_{simulation_id}_{chart_type}.png"
        )
        
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
    """Serve visualization as base64 encoded image data - works on all hosting platforms"""
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
        
        results = simulation.results or {}
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
            "success": len(charts) > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting all charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulation/{simulation_id}/regenerate-shap")
async def regenerate_shap_chart(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Regenerate SHAP visualization using the portfolio simulator service"""
    try:
        if not PORTFOLIO_SIMULATOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Portfolio simulator service not available")
        
        result = await regenerate_shap_visualization(simulation_id, db)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating SHAP visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulation/{simulation_id}/generate-chart/{chart_type}")
async def generate_specific_chart(
    simulation_id: int,
    chart_type: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Generate a specific chart type on demand"""
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        results = simulation.results or {}
        
        # Get or initialize visualization engine
        viz_engine = get_visualization_engine()
        if not viz_engine:
            # Return as base64 instead of failing
            chart_result = await generate_chart_as_base64(simulation_id, chart_type, db)
            return chart_result

        # Ensure directory exists
        viz_dir = Path("static/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        chart_path = viz_dir / f"sim_{simulation_id}_{chart_type}_generated.png"
        
        # Generate based on chart type
        if chart_type == "shap_explanation":
            shap_explanation = results.get("shap_explanation")
            if not shap_explanation:
                raise HTTPException(status_code=404, detail="SHAP explanation not available")
            
            result = viz_engine.create_shap_waterfall_chart(shap_explanation, str(chart_path))
            
        elif chart_type == "portfolio_composition":
            stocks_picked = results.get("stocks_picked", [])
            if not stocks_picked:
                raise HTTPException(status_code=404, detail="Portfolio data not available")
            
            stocks = [stock["symbol"] for stock in stocks_picked]
            weights = {stock["symbol"]: stock["allocation"] for stock in stocks_picked}
            
            result = viz_engine.create_portfolio_composition_chart(stocks, weights, str(chart_path))
            
        elif chart_type == "risk_return_analysis":
            # Calculate portfolio metrics
            portfolio_return = results.get("return", 0.08)
            risk_score = simulation.risk_score or 50
            estimated_volatility = 0.05 + (risk_score / 100) * 0.20
            
            portfolio_metrics = {
                "expected_return": float(portfolio_return),
                "volatility": float(estimated_volatility),
                "sharpe_ratio": float(portfolio_return / estimated_volatility) if estimated_volatility > 0 else 0
            }
            
            result = viz_engine.create_risk_return_scatter(portfolio_metrics, [], str(chart_path))
            
        elif chart_type == "market_regime":
            market_regime = results.get("market_regime")
            if not market_regime:
                raise HTTPException(status_code=404, detail="Market regime data not available")
            
            result = viz_engine.create_market_regime_visualization(market_regime, str(chart_path))
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown chart type: {chart_type}")
        
        if "saved" in result.lower():
            return {
                "success": True,
                "chart_path": str(chart_path),
                "chart_url": f"/static/visualizations/{chart_path.name}",
                "message": f"{chart_type} chart generated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Chart generation failed: {result}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating {chart_type} chart: {e}")
        # Fallback to base64 generation
        return await generate_chart_as_base64(simulation_id, chart_type, db)

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
            },
        }

        return chart_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error formatting chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to format chart data: {str(e)}")

# Fallback visualization function
async def create_fallback_visualization(simulation_id: int, chart_type: str, db: Session) -> Response:
    """Create a fallback visualization when main visualization engine is not available"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np

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