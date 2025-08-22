"""
Database service for the Portfolio Simulator Service.

Handles DB operations: save, fetch, list, delete, and formatting of simulation results.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func  # ✅ needed for aggregates

from database import models
from .config import get_config
from .exceptions import DatabaseError
from .validators import InputValidator

__all__ = ["DatabaseService", "SimulationResultsFormatter"]

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Handles all database operations for portfolio simulations.
    """

    def __init__(self, validator: Optional[InputValidator] = None):
        self.config = get_config()
        self.validator = validator or InputValidator()

    # === New simple API used by main_service.py =================================

    def save_simulation(
        self,
        db: Session,
        user_id: Optional[int],
        name: str,
        input_payload: Dict[str, Any],
        result_payload: Dict[str, Any],
    ) -> models.Simulation:
        """
        Persist a simulation using a compact interface.

        Parameters mirror the orchestrator:
          - user_id: owner
          - name: display name (e.g., goal)
          - input_payload: original request
          - result_payload: simulation output already shaped for UI

        Returns the created Simulation row.
        """
        try:
            logger.info("Saving simulation to database (compact API)")

            # Derive a few convenient top-level columns from payloads
            target_value = float(input_payload.get("target_value", 0) or 0)
            lump_sum = float(input_payload.get("lump_sum", 0) or 0)
            monthly = float(input_payload.get("monthly", 0) or 0)
            timeframe = int(input_payload.get("timeframe", 0) or 0)
            income_bracket = (input_payload.get("income_bracket") or "medium")

            # Risk info (prefer detailed when present)
            risk_score = int(
                (input_payload.get("risk_score")
                 or result_payload.get("portfolio_metrics", {}).get("risk_score")
                 or 0)
            )
            risk_label = (
                input_payload.get("risk_label")
                or input_payload.get("detailed_risk_profile", {}).get("risk_level")
                or result_payload.get("recommendation", {}).get("risk_label")
                or "Moderate"
            )

            # Determine if target was achieved based on result payload
            target_achieved = bool(result_payload.get("target_achieved", False))

            # AI summary optional (if you have one upstream)
            ai_summary = (result_payload.get("ai_summary") or "")

            sim = models.Simulation(
                user_id=user_id,
                name=name or "Investment Simulation",
                goal=input_payload.get("goal", "wealth building"),
                target_value=target_value,
                lump_sum=lump_sum,
                monthly=monthly,
                timeframe=timeframe,
                target_achieved=target_achieved,
                income_bracket=income_bracket,
                risk_score=risk_score,
                risk_label=risk_label,
                ai_summary=ai_summary,
                results=result_payload,  # store the rich JSON blob
            )

            db.add(sim)
            db.commit()
            db.refresh(sim)
            logger.info("Simulation saved successfully with ID %s", sim.id)
            return sim

        except SQLAlchemyError as e:
            logger.error("Database error saving simulation: %s", e)
            db.rollback()
            raise DatabaseError(
                f"Failed to save simulation: {str(e)}",
                operation="insert",
                table="simulations",
            )
        except Exception as e:
            logger.error("Unexpected error saving simulation: %s", e)
            db.rollback()
            raise DatabaseError(
                f"Unexpected error saving simulation: {str(e)}",
                operation="insert",
                table="simulations",
            )

    # === Legacy API kept for backwards compatibility ============================

    def save_simulation_legacy(
        self,
        db: Session,
        simulation_input: Dict[str, Any],
        user_data: Dict[str, Any],
        simulation_results: Dict[str, Any],
        ai_summary: str,
        stocks_data: List[Dict[str, Any]],
        risk_score: int,
        risk_label: str,
        shap_explanation: Optional[Dict[str, Any]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
    ) -> models.Simulation:
        """
        Legacy save method kept for older callers.
        """
        try:
            logger.info("Saving simulation to database (legacy API)")

            enhanced_results = self._prepare_simulation_results(
                simulation_results,
                stocks_data,
                shap_explanation,
                visualization_paths,
                risk_score,
                risk_label,
            )

            end_value = simulation_results.get("end_value", 0)
            target_value = user_data.get("target_value", 0)
            target_achieved = end_value >= target_value

            sim = models.Simulation(
                user_id=simulation_input.get("user_id"),
                name=user_data.get("goal", "Investment Simulation"),
                goal=user_data.get("goal", "wealth building"),
                target_value=target_value,
                lump_sum=user_data.get("lump_sum", 0),
                monthly=user_data.get("monthly", 0),
                timeframe=user_data.get("timeframe", 10),
                target_achieved=target_achieved,
                income_bracket=user_data.get("income_bracket", "medium"),
                risk_score=risk_score,
                risk_label=risk_label,
                ai_summary=ai_summary,
                results=enhanced_results,
            )

            db.add(sim)
            db.commit()
            db.refresh(sim)
            logger.info("Simulation saved successfully with ID %s", sim.id)
            return sim

        except SQLAlchemyError as e:
            logger.error("Database error saving simulation (legacy): %s", e)
            db.rollback()
            raise DatabaseError(
                f"Failed to save simulation: {str(e)}",
                operation="insert",
                table="simulations",
            )
        except Exception as e:
            logger.error("Unexpected error saving simulation (legacy): %s", e)
            db.rollback()
            raise DatabaseError(
                f"Unexpected error saving simulation: {str(e)}",
                operation="insert",
                table="simulations",
            )

    # === Updates / retrieval ====================================================

    def update_simulation_visualizations(
        self, db: Session, simulation_id: int, visualization_paths: Dict[str, str]
    ) -> None:
        try:
            sim = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            if not sim:
                raise DatabaseError(
                    f"Simulation with ID {simulation_id} not found",
                    operation="update",
                    table="simulations",
                )

            sim.results = sim.results or {}
            sim.results["visualization_paths"] = visualization_paths
            sim.results["has_visualizations"] = bool(visualization_paths)
            sim.results["visualization_count"] = len(visualization_paths)

            db.commit()
            logger.info("Updated visualizations for simulation %s", simulation_id)

        except SQLAlchemyError as e:
            logger.error("DB error updating visualization paths: %s", e)
            db.rollback()
            raise DatabaseError(
                f"Failed to update visualization paths: {str(e)}",
                operation="update",
                table="simulations",
            )

    def get_simulation(self, db: Session, simulation_id: int) -> Optional[models.Simulation]:
        try:
            return (
                db.query(models.Simulation)
                .filter(models.Simulation.id == simulation_id)
                .first()
            )
        except SQLAlchemyError as e:
            logger.error("DB error retrieving simulation: %s", e)
            raise DatabaseError(
                f"Failed to retrieve simulation: {str(e)}",
                operation="select",
                table="simulations",
            )

    def get_user_simulations(
        self, db: Session, user_id: int, limit: int = 10, offset: int = 0
    ) -> List[models.Simulation]:
        try:
            return (
                db.query(models.Simulation)
                .filter(models.Simulation.user_id == user_id)
                .order_by(models.Simulation.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        except SQLAlchemyError as e:
            logger.error("DB error retrieving user simulations: %s", e)
            raise DatabaseError(
                f"Failed to retrieve user simulations: {str(e)}",
                operation="select",
                table="simulations",
            )

    def delete_simulation(self, db: Session, simulation_id: int) -> bool:
        try:
            sim = (
                db.query(models.Simulation)
                .filter(models.Simulation.id == simulation_id)
                .first()
            )
            if not sim:
                return False
            db.delete(sim)
            db.commit()
            logger.info("Deleted simulation %s", simulation_id)
            return True
        except SQLAlchemyError as e:
            logger.error("DB error deleting simulation: %s", e)
            db.rollback()
            raise DatabaseError(
                f"Failed to delete simulation: {str(e)}",
                operation="delete",
                table="simulations",
            )

    def get_simulation_statistics(self, db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
        try:
            q = db.query(models.Simulation)
            if user_id:
                q = q.filter(models.Simulation.user_id == user_id)

            total = q.count()
            successful = q.filter(models.Simulation.target_achieved.is_(True)).count()

            if total > 0:
                avg_target = (
                    db.query(func.avg(models.Simulation.target_value))
                    .filter(models.Simulation.user_id == user_id if user_id else True)
                    .scalar()
                    or 0
                )
                avg_timeframe = (
                    db.query(func.avg(models.Simulation.timeframe))
                    .filter(models.Simulation.user_id == user_id if user_id else True)
                    .scalar()
                    or 0
                )
                success_rate = (successful / total) * 100
            else:
                avg_target = 0
                avg_timeframe = 0
                success_rate = 0

            return {
                "total_simulations": total,
                "successful_simulations": successful,
                "success_rate": round(success_rate, 1),
                "average_target_value": round(float(avg_target), 2),
                "average_timeframe": round(float(avg_timeframe), 1),
            }

        except SQLAlchemyError as e:
            logger.error("DB error getting statistics: %s", e)
            raise DatabaseError(
                f"Failed to get simulation statistics: {str(e)}",
                operation="select",
                table="simulations",
            )

    # === Helpers for legacy serializer =========================================

    def _prepare_simulation_results(
        self,
        simulation_results: Dict[str, Any],
        stocks_data: List[Dict[str, Any]],
        shap_explanation: Optional[Dict[str, Any]],
        visualization_paths: Optional[Dict[str, str]],
        risk_score: int,
        risk_label: str,
    ) -> Dict[str, Any]:
        enhanced = {
            "stocks_picked": self._serialize_stocks_data(stocks_data),
            "starting_value": float(simulation_results.get("starting_value", 0)),
            "end_value": float(simulation_results.get("ending_value", 0)),
            "total_return": float(simulation_results.get("total_return", 0)),
            "annualized_return": float(simulation_results.get("annualized_return", 0)),
            "volatility": float(simulation_results.get("volatility", 0)),
            "sharpe_ratio": float(simulation_results.get("sharpe_ratio", 0)),
            "max_drawdown": float(simulation_results.get("max_drawdown", 0)),
            "total_contributed": float(simulation_results.get("total_contributed", 0)),
            "profit_loss": float(simulation_results.get("profit_loss", 0)),
            "risk_score": risk_score,
            "risk_label": risk_label,
        }

        timeline_data = simulation_results.get("timeline_data", [])
        contribution_data = simulation_results.get("contribution_data", [])
        if timeline_data or contribution_data:
            enhanced["timeline"] = {
                "portfolio": self._serialize_timeline_data(timeline_data),
                "contributions": self._serialize_timeline_data(contribution_data),
            }

        asset_breakdown = simulation_results.get("asset_breakdown", {})
        if asset_breakdown:
            enhanced["breakdown"] = {str(k): float(v) for k, v in asset_breakdown.items()}

        if shap_explanation:
            enhanced["shap_explanation"] = self._serialize_shap_data(shap_explanation)
            enhanced["has_shap_explanations"] = True
        else:
            enhanced["has_shap_explanations"] = False

        if visualization_paths:
            enhanced["visualization_paths"] = visualization_paths
            enhanced["has_visualizations"] = True
            enhanced["visualization_count"] = len(visualization_paths)
        else:
            enhanced["has_visualizations"] = False
            enhanced["visualization_count"] = 0

        enhanced["metadata"] = {
            "created_timestamp": datetime.now().isoformat(),
            "methodology": "Enhanced AI-powered simulation with SHAP explanations",
            "version": "2.0",
            "enhanced_features": {
                "ai_recommendations": True,
                "shap_explanations": bool(shap_explanation),
                "visualizations": bool(visualization_paths),
                "goal_optimization": True,
            },
        }
        return enhanced

    def _serialize_stocks_data(self, stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for stock in stocks_data:
            out.append(
                {
                    "symbol": str(stock.get("symbol", "")),
                    "name": str(stock.get("name", "")),
                    "allocation": float(stock.get("allocation", 0)),
                    "explanation": str(stock.get("explanation", "")),
                }
            )
        return out

    def _serialize_timeline_data(self, timeline_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in timeline_data:
            row: Dict[str, Any] = {}
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    row[k] = float(v)
                else:
                    row[k] = str(v)
            out.append(row)
        return out

    def _serialize_shap_data(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in shap_explanation.items():
            if k == "feature_contributions" and isinstance(v, dict):
                out[k] = {str(f): float(c) for f, c in v.items()}
            elif k == "human_readable_explanation" and isinstance(v, dict):
                out[k] = {str(f): str(txt) for f, txt in v.items()}
            elif isinstance(v, (int, float)):
                out[k] = float(v)
            elif isinstance(v, dict):
                out[k] = self._serialize_nested_dict(v)
            else:
                out[k] = str(v)
        return out

    def _serialize_nested_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
            elif isinstance(v, dict):
                out[str(k)] = self._serialize_nested_dict(v)
            elif isinstance(v, list):
                out[str(k)] = [
                    self._serialize_nested_dict(i) if isinstance(i, dict) else str(i) for i in v
                ]
            else:
                out[str(k)] = str(v)
        return out


class SimulationResultsFormatter:
    """
    Formats simulation results for API responses (when reading back from DB).
    """

    def __init__(self):
        self.config = get_config()

    def format_simulation_response(self, simulation: models.Simulation) -> Dict[str, Any]:
        results = simulation.results or {}

        response = {
            "id": simulation.id,
            "user_id": simulation.user_id,
            "name": simulation.name,
            "goal": simulation.goal,
            "target_value": simulation.target_value,
            "lump_sum": simulation.lump_sum,
            "monthly": simulation.monthly,
            "timeframe": simulation.timeframe,
            "target_achieved": simulation.target_achieved,
            "income_bracket": simulation.income_bracket,
            "risk_score": simulation.risk_score,
            "risk_label": simulation.risk_label,
            "ai_summary": simulation.ai_summary,
            "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
        }

        # FIXED: Map nested data to top-level fields for frontend
        portfolio_metrics = results.get("portfolio_metrics", {})
        
        response.update({
            "results": results,
            "has_shap_explanations": results.get("shap_explanation") is not None,
            "has_visualizations": results.get("has_visualizations", False),
            "visualization_count": results.get("visualization_count", 0),
            "methodology": results.get("metadata", {}).get("methodology", "Standard simulation"),
            "enhanced_features": results.get("metadata", {}).get("enhanced_features", {}),
            
            # Map nested data to top-level fields for frontend
            "final_balance": portfolio_metrics.get("ending_value", 0),
            "stocks": results.get("stocks_picked", []),
            "breakdown": {
                stock["symbol"]: stock["allocation"] 
                for stock in results.get("stocks_picked", [])
            },
            "timeline": portfolio_metrics.get("timeline_data", []),
            "volatility": portfolio_metrics.get("volatility", 0),
            "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0),
            "total_return": portfolio_metrics.get("total_return", 0),
        })

        performance_metrics = self._extract_performance_metrics(results, portfolio_metrics)
        response["performance_metrics"] = performance_metrics

        if results.get("shap_explanation"):
            response["shap_summary"] = self._extract_shap_summary(results["shap_explanation"])

        if results.get("visualization_paths"):
            response["available_visualizations"] = list(results["visualization_paths"].keys())
        else:
            response["available_visualizations"] = []

        return response

    def format_simulation_list(self, simulations: List[models.Simulation]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sim in simulations:
            # Extract portfolio metrics for list view
            portfolio_metrics = sim.results.get("portfolio_metrics", {}) if sim.results else {}
            
            summary = {
                "id": sim.id,
                "name": sim.name,
                "goal": sim.goal,
                "target_value": sim.target_value,
                "target_achieved": sim.target_achieved,
                "risk_label": sim.risk_label,
                "created_at": sim.created_at.isoformat() if sim.created_at else None,
                "has_shap_explanations": sim.results.get("shap_explanation") is not None if sim.results else False,
                "has_visualizations": sim.results.get("has_visualizations", False) if sim.results else False,
            }
            if sim.results:
                summary.update({
                    "end_value": portfolio_metrics.get("ending_value", 0),
                    "total_return": portfolio_metrics.get("total_return", 0),
                    "profit_loss": portfolio_metrics.get("profit_loss", 0),
                })
            out.append(summary)
        return out

    def _extract_performance_metrics(self, results: Dict[str, Any], portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Use portfolio_metrics first, fall back to top-level results
        return {
            "starting_value": portfolio_metrics.get("starting_value", results.get("starting_value", 0)),
            "ending_value": portfolio_metrics.get("ending_value", results.get("end_value", 0)),
            "total_return": portfolio_metrics.get("total_return", results.get("total_return", 0)),
            "annualized_return": portfolio_metrics.get("annualized_return", results.get("annualized_return", 0)),
            "volatility": portfolio_metrics.get("volatility", results.get("volatility", 0)),
            "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", results.get("sharpe_ratio", 0)),
            "max_drawdown": portfolio_metrics.get("max_drawdown", results.get("max_drawdown", 0)),
            "total_contributed": portfolio_metrics.get("total_contributed", results.get("total_contributed", 0)),
            "profit_loss": portfolio_metrics.get("profit_loss", results.get("profit_loss", 0)),
        }

    def _extract_shap_summary(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "portfolio_quality_score": shap_explanation.get("portfolio_quality_score", 0),
            "top_factors": self._get_top_shap_factors(shap_explanation),
            "overall_explanation": shap_explanation.get("human_readable_explanation", {}).get("overall", ""),
        }

    def _get_top_shap_factors(self, shap_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        feats = shap_explanation.get("feature_contributions", {})
        sorted_f = sorted(feats.items(), key=lambda x: abs(float(x[1])), reverse=True)
        out: List[Dict[str, Any]] = []
        for feature, contribution in sorted_f[:3]:
            out.append(
                {
                    "feature": feature,
                    "contribution": float(contribution),
                    "impact": "positive" if float(contribution) > 0 else "negative",
                }
            )
        return out