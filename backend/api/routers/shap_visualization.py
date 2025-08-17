# api/routers/shap_visualization.py

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database.db import get_db
from database import models

import os
import logging
from typing import Dict, Any, Optional
import json
import base64
import io

# --- SHAP extraction helpers -------------------------------------------------
from typing import Iterable

def _find_shap_in(obj: Any) -> Optional[Dict[str, Any]]:
    """Deep-search for SHAP-like explanation dictionaries.
    Looks for common keys (shap_explanation, shap_data, explanation).
    Returns the first dict-like match found, otherwise None.
    """
    if obj is None:
        return None
    # Direct dict hit
    if isinstance(obj, dict):
        # Preferred keys
        for key in ("shap_explanation", "shap_data", "explanation"):
            val = obj.get(key)
            if isinstance(val, dict):
                return val
        # Deep scan
        for v in obj.values():
            found = _find_shap_in(v)
            if found is not None:
                return found
        return None
    # List/tuple deep scan
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_shap_in(v)
            if found is not None:
                return found
        return None
    # Fallback: try __dict__ of ORM objects
    if hasattr(obj, "__dict__"):
        return _find_shap_in({k: v for k, v in obj.__dict__.items() if not k.startswith("_")})
    return None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/shap", tags=["SHAP Visualization"])

@router.get("/simulation/{simulation_id}/explanation")
async def get_shap_explanation_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get SHAP explanation data for a simulation.
    
    Returns the raw SHAP explanation data that can be used
    to create visualizations on the frontend.
    """
    try:
        # Get simulation from database
        print(f"Fetching SHAP explanation for simulation ID: {simulation_id}")
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        print(f"Simulation fetched: {simulation}")
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        results = simulation.results or {}
        print(f"Results field from simulation: {results}")

        # Try multiple locations + deep search for SHAP data
        shap_explanation = (
            _find_shap_in(results)
            or _find_shap_in(getattr(simulation, "shap_explanation", None))
            or _find_shap_in({k: v for k, v in simulation.__dict__.items() if not k.startswith("_")})
        )
        print(f"Resolved SHAP explanation: {shap_explanation}")

        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available for this simulation")

        # Return structured SHAP data for frontend visualization
        print("Returning SHAP explanation response")
        return {
            "simulation_id": simulation_id,
            "shap_data": {
                "portfolio_quality_score": shap_explanation.get("portfolio_quality_score"),
                "confidence_score": shap_explanation.get("confidence_score"),
                "feature_importance": shap_explanation.get("feature_importance", {}),
                "human_readable_explanation": shap_explanation.get("human_readable_explanation", {}),
                "methodology": shap_explanation.get("methodology", "SHAP-based explainable AI"),
                "shap_values": shap_explanation.get("shap_values", []),
            },
            "portfolio_info": {
                "stocks": [stock.get("symbol") for stock in results.get("stocks_picked", [])],
                # Risk score sits on the simulation, not inside results
                "risk_score": getattr(simulation, "risk_score", None),
                "expected_return": results.get("portfolio_metrics", {}).get("expected_return"),
                "volatility": results.get("portfolio_metrics", {}).get("volatility"),
            },
            "goal_analysis": results.get("goal_analysis", {}),
            "market_regime": results.get("market_regime", {}),
            "metadata": {
                # wealthwise_enhanced is a top-level field on your payload
                "wealthwise_enhanced": getattr(simulation, "wealthwise_enhanced", False),
                "has_shap_explanations": getattr(simulation, "has_shap_explanations", False),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting SHAP explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SHAP explanation: {str(e)}")

@router.get("/simulation/{simulation_id}/visualization")
async def get_shap_visualization_image(
    simulation_id: int,
    db: Session = Depends(get_db),
    chart_type: str = "waterfall"
) -> Response:
    """
    Generate and return SHAP visualization as an image.
    
    Args:
        simulation_id: ID of the simulation
        chart_type: Type of chart ('waterfall', 'bar', 'summary')
    
    Returns:
        PNG image of the SHAP visualization
    """
    try:
        # Get simulation
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        results = simulation.results or {}
        shap_explanation = (
            _find_shap_in(results)
            or _find_shap_in(getattr(simulation, "shap_explanation", None))
            or _find_shap_in({k: v for k, v in simulation.__dict__.items() if not k.startswith("_")})
        )
        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available for this simulation")
        
        # Generate SHAP visualization
        try:
            from ai_models.stock_model.explainable_ai.visualization import SHAPVisualizer
            visualizer = SHAPVisualizer()
            
            # Create visualization based on chart type
            if chart_type == "waterfall":
                image_path = visualizer.create_waterfall_chart(
                    shap_explanation, 
                    save_path=f"./static/temp/shap_waterfall_{simulation_id}.png"
                )
            elif chart_type == "bar":
                image_path = visualizer.create_feature_importance_chart(
                    shap_explanation,
                    save_path=f"./static/temp/shap_bar_{simulation_id}.png"
                )
            elif chart_type == "summary":
                image_path = visualizer.create_summary_chart(
                    shap_explanation,
                    save_path=f"./static/temp/shap_summary_{simulation_id}.png"
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid chart type")
            
            # Return the image file
            if os.path.exists(image_path):
                return FileResponse(
                    image_path,
                    media_type="image/png",
                    filename=f"shap_{chart_type}_{simulation_id}.png"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate visualization")
                
        except ImportError:
            # Fallback: create a simple visualization using matplotlib
            return await create_fallback_shap_visualization(shap_explanation, simulation_id, chart_type)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SHAP visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {str(e)}")

@router.get("/simulation/{simulation_id}/chart-data")
async def get_shap_chart_data(
    simulation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get SHAP data formatted for frontend charting libraries (Chart.js, D3, etc.)
    
    This endpoint provides the data in a format that's easy to use with
    popular JavaScript charting libraries.
    """
    try:
        # Get simulation
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        results = simulation.results or {}
        shap_explanation = (
            _find_shap_in(results)
            or _find_shap_in(getattr(simulation, "shap_explanation", None))
            or _find_shap_in({k: v for k, v in simulation.__dict__.items() if not k.startswith("_")})
        )
        if not shap_explanation:
            raise HTTPException(status_code=404, detail="SHAP explanation not available")
        
        # Format data for Chart.js
        feature_importance = shap_explanation.get("feature_importance", {})
        
        chart_data = {
            "portfolio_score": {
                "value": shap_explanation.get("portfolio_quality_score", 0),
                "max_value": 100,
                "label": "Portfolio Quality Score"
            },
            "feature_importance": {
                "labels": list(feature_importance.keys()),
                "values": list(feature_importance.values()),
                "colors": generate_chart_colors(len(feature_importance))
            },
            "explanations": shap_explanation.get("human_readable_explanation", {}),
            "metadata": {
                "methodology": shap_explanation.get("methodology", "SHAP Explainable AI"),
                "confidence": shap_explanation.get("confidence_score", 0),
                "wealthwise_enhanced": getattr(simulation, "wealthwise_enhanced", False),
                "has_shap_explanations": getattr(simulation, "has_shap_explanations", False),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
            }
        }
        
        return chart_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error formatting SHAP chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to format chart data: {str(e)}")

@router.get("/simulations/compare")
async def compare_shap_explanations(
    simulation_ids: str,  # Comma-separated simulation IDs
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Compare SHAP explanations across multiple simulations.
    
    This is useful for showing users how different risk profiles
    or goals lead to different AI recommendations.
    """
    try:
        # Parse simulation IDs
        sim_ids = [int(id.strip()) for id in simulation_ids.split(",")]
        
        if len(sim_ids) > 5:  # Limit to 5 simulations for performance
            raise HTTPException(status_code=400, detail="Too many simulations (max 5)")
        
        comparisons = []
        
        for sim_id in sim_ids:
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == sim_id
            ).first()
            
            if simulation:
                results = simulation.results or {}
                shap_explanation = results.get("shap_explanation", {})
                
                comparison_data = {
                    "simulation_id": sim_id,
                    "name": simulation.name,
                    "risk_score": simulation.risk_score,
                    "portfolio_quality": shap_explanation.get("portfolio_quality_score", 0),
                    "key_factors": list(shap_explanation.get("feature_importance", {}).keys())[:3],
                    "stocks": [stock.get("symbol") for stock in results.get("stocks_picked", [])],
                    "expected_return": results.get("portfolio_metrics", {}).get("expected_return", 0)
                }
                comparisons.append(comparison_data)
        
        return {
            "comparisons": comparisons,
            "summary": {
                "total_simulations": len(comparisons),
                "avg_portfolio_quality": sum(c["portfolio_quality"] for c in comparisons) / len(comparisons) if comparisons else 0,
                "risk_range": {
                    "min": min(c["risk_score"] for c in comparisons) if comparisons else 0,
                    "max": max(c["risk_score"] for c in comparisons) if comparisons else 0
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing SHAP explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare explanations: {str(e)}")

# Helper functions

async def create_fallback_shap_visualization(shap_explanation: Dict, simulation_id: int, chart_type: str) -> Response:
    """
    Create a simple SHAP visualization using matplotlib when advanced visualization isn't available.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get feature importance data
        feature_importance = shap_explanation.get("feature_importance", {})
        
        if feature_importance:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            
            # Create bar chart
            bars = ax.bar(features, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(features)])
            
            ax.set_title(f'SHAP Feature Importance - Simulation {simulation_id}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Importance Score', fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        else:
            # No data available
            ax.text(0.5, 0.5, 'SHAP explanation data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'SHAP Analysis - Simulation {simulation_id}')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png"
        )
        
    except Exception as e:
        logger.error(f"Error creating fallback visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to create visualization")

def generate_chart_colors(num_colors: int) -> list:
    """Generate a list of colors for charts."""
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]
    else:
        # Generate additional colors if needed
        import colorsys
        additional_colors = []
        for i in range(num_colors - len(base_colors)):
            hue = i / (num_colors - len(base_colors))
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            additional_colors.append(hex_color)
        
        return base_colors + additional_colors