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

@router.get("/simulation/{simulation_id}/explanation")
async def get_shap_explanation_data(
    simulation_id: int, 
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
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

        portfolio_metrics = results.get("portfolio_metrics") or {}

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
                "stocks": [s.get("symbol") for s in results.get("stocks_picked", [])],
                "risk_score": results.get("risk_score"),
                "expected_return": portfolio_metrics.get("expected_return"),
                "volatility": portfolio_metrics.get("volatility"),
            },
            "goal_analysis": results.get("goal_analysis", {}),
            "market_regime": results.get("market_regime", {}),
            "metadata": {
                "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
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

        try:
            from ai_models.stock_model.explainable_ai.visualization import SHAPVisualizer
            visualizer = SHAPVisualizer()

            if chart_type == "waterfall":
                image_path = visualizer.create_waterfall_chart(
                    shap_explanation, save_path=f"./static/temp/shap_waterfall_{simulation_id}.png"
                )
            elif chart_type == "bar":
                image_path = visualizer.create_feature_importance_chart(
                    shap_explanation, save_path=f"./static/temp/shap_bar_{simulation_id}.png"
                )
            elif chart_type == "summary":
                image_path = visualizer.create_summary_chart(
                    shap_explanation, save_path=f"./static/temp/shap_summary_{simulation_id}.png"
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid chart type")

            if os.path.exists(image_path):
                return FileResponse(image_path, media_type="image/png", filename=f"shap_{chart_type}_{simulation_id}.png")
            else:
                raise HTTPException(status_code=500, detail="Failed to generate visualization")

        except ImportError:
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

        feature_importance = shap_explanation.get("feature_importance", {})

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
                "methodology": "SHAP Explainable AI",
                "confidence": shap_explanation.get("confidence_score", 0),
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
            },
        }

        return chart_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error formatting SHAP chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to format chart data: {str(e)}")

@router.get("/simulations/compare")
async def compare_shap_explanations(
    simulation_ids: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    try:
        sim_ids = [int(s.strip()) for s in simulation_ids.split(",") if s.strip()]
        if len(sim_ids) > 5:
            raise HTTPException(status_code=400, detail="Too many simulations (max 5)")

        comparisons = []
        for sim_id in sim_ids:
            simulation = db.query(models.Simulation).filter(models.Simulation.id == sim_id).first()
            if not simulation:
                continue
            results = simulation.results or {}
            shap_explanation = results.get("shap_explanation", {})
            portfolio_metrics = results.get("portfolio_metrics") or {}

            comparisons.append({
                "simulation_id": sim_id,
                "name": simulation.name,
                "risk_score": simulation.risk_score,
                "portfolio_quality": shap_explanation.get("portfolio_quality_score", 0),
                "key_factors": list(shap_explanation.get("feature_importance", {}).keys())[:3],
                "stocks": [s.get("symbol") for s in results.get("stocks_picked", [])],
                "expected_return": portfolio_metrics.get("expected_return", 0),
            })

        return {
            "comparisons": comparisons,
            "summary": {
                "total_simulations": len(comparisons),
                "avg_portfolio_quality": (sum(c["portfolio_quality"] for c in comparisons) / len(comparisons)) if comparisons else 0,
                "risk_range": {
                  "min": min((c["risk_score"] for c in comparisons), default=0),
                  "max": max((c["risk_score"] for c in comparisons), default=0),
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing SHAP explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare explanations: {str(e)}")


# Helper functions
async def create_fallback_shap_visualization(shap_explanation: Dict, simulation_id: int, chart_type: str) -> Response:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))

        feature_importance = shap_explanation.get("feature_importance", {})
        if feature_importance:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            bars = ax.bar(features, values)
            ax.set_title(f'SHAP Feature Importance - Simulation {simulation_id}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Importance Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'SHAP explanation data not available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
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
    base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if num_colors <= len(base):
        return base[:num_colors]
    import colorsys
    extra = []
    for i in range(num_colors - len(base)):
        hue = i / max(1, (num_colors - len(base)))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        extra.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
    return base + extra