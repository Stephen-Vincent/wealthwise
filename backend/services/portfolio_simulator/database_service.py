"""
Database service for the Portfolio Simulator Service.

This module handles all database operations including saving simulations,
retrieving results, and managing simulation data with proper error handling.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database import models
from .config import get_config
from .exceptions import DatabaseError
from .validators import InputValidator

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Handles all database operations for portfolio simulations.
    
    This service provides methods to save simulation results, retrieve
    historical data, and manage simulation metadata with proper error handling.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the database service.
        
        Args:
            validator: Input validator instance
        """
        self.config = get_config()
        self.validator = validator or InputValidator()
    
    def save_simulation(
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
        visualization_paths: Optional[Dict[str, str]] = None
    ) -> models.Simulation:
        """
        Save a complete simulation to the database.
        
        Args:
            db: Database session
            simulation_input: Original input parameters
            user_data: User profile data
            simulation_results: Complete simulation results
            ai_summary: AI-generated summary
            stocks_data: Stock allocation data
            risk_score: Risk tolerance score
            risk_label: Risk profile label
            shap_explanation: SHAP explanation data (optional)
            visualization_paths: Paths to generated visualizations (optional)
            
        Returns:
            Saved simulation model instance
            
        Raises:
            DatabaseError: If saving fails
        """
        try:
            logger.info("Saving simulation to database")
            
            # Prepare enhanced results dictionary
            enhanced_results = self._prepare_simulation_results(
                simulation_results, stocks_data, shap_explanation, 
                visualization_paths, risk_score, risk_label
            )
            
            # Determine if target was achieved
            end_value = simulation_results.get("end_value", 0)
            target_value = user_data.get("target_value", 0)
            target_achieved = end_value >= target_value
            
            # Create simulation model
            simulation = models.Simulation(
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
                results=enhanced_results
            )
            
            # Save to database
            db.add(simulation)
            db.commit()
            db.refresh(simulation)
            
            logger.info(f"Simulation saved successfully with ID: {simulation.id}")
            return simulation
            
        except SQLAlchemyError as e:
            logger.error(f"Database error saving simulation: {str(e)}")
            db.rollback()
            raise DatabaseError(
                f"Failed to save simulation: {str(e)}",
                operation="insert",
                table="simulations"
            )
        except Exception as e:
            logger.error(f"Unexpected error saving simulation: {str(e)}")
            db.rollback()
            raise DatabaseError(
                f"Unexpected error saving simulation: {str(e)}",
                operation="insert",
                table="simulations"
            )
    
    def update_simulation_visualizations(
        self,
        db: Session,
        simulation_id: int,
        visualization_paths: Dict[str, str]
    ) -> None:
        """
        Update visualization paths for an existing simulation.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to update
            visualization_paths: Dictionary of visualization paths
            
        Raises:
            DatabaseError: If update fails
        """
        try:
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            if not simulation:
                raise DatabaseError(
                    f"Simulation with ID {simulation_id} not found",
                    operation="update",
                    table="simulations"
                )
            
            # Update results with new visualization paths
            if simulation.results:
                simulation.results["visualization_paths"] = visualization_paths
                simulation.results["has_visualizations"] = bool(visualization_paths)
                simulation.results["visualization_count"] = len(visualization_paths)
            else:
                simulation.results = {
                    "visualization_paths": visualization_paths,
                    "has_visualizations": bool(visualization_paths),
                    "visualization_count": len(visualization_paths)
                }
            
            db.commit()
            logger.info(f"Updated visualizations for simulation {simulation_id}")
            
        except SQLAlchemyError as e:
            logger.error(f"Database error updating visualization paths: {str(e)}")
            db.rollback()
            raise DatabaseError(
                f"Failed to update visualization paths: {str(e)}",
                operation="update",
                table="simulations"
            )
    
    def get_simulation(self, db: Session, simulation_id: int) -> Optional[models.Simulation]:
        """
        Retrieve a simulation by ID.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to retrieve
            
        Returns:
            Simulation model instance or None if not found
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            return simulation
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving simulation: {str(e)}")
            raise DatabaseError(
                f"Failed to retrieve simulation: {str(e)}",
                operation="select",
                table="simulations"
            )
    
    def get_user_simulations(
        self, 
        db: Session, 
        user_id: int, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[models.Simulation]:
        """
        Retrieve simulations for a specific user.
        
        Args:
            db: Database session
            user_id: ID of the user
            limit: Maximum number of simulations to return
            offset: Number of simulations to skip
            
        Returns:
            List of simulation model instances
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            simulations = db.query(models.Simulation).filter(
                models.Simulation.user_id == user_id
            ).order_by(
                models.Simulation.created_at.desc()
            ).limit(limit).offset(offset).all()
            
            return simulations
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving user simulations: {str(e)}")
            raise DatabaseError(
                f"Failed to retrieve user simulations: {str(e)}",
                operation="select",
                table="simulations"
            )
    
    def delete_simulation(self, db: Session, simulation_id: int) -> bool:
        """
        Delete a simulation by ID.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to delete
            
        Returns:
            True if deletion was successful, False if simulation not found
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            if not simulation:
                return False
            
            db.delete(simulation)
            db.commit()
            
            logger.info(f"Deleted simulation {simulation_id}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting simulation: {str(e)}")
            db.rollback()
            raise DatabaseError(
                f"Failed to delete simulation: {str(e)}",
                operation="delete",
                table="simulations"
            )
    
    def get_simulation_statistics(self, db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics about simulations.
        
        Args:
            db: Database session
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary with simulation statistics
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            query = db.query(models.Simulation)
            
            if user_id:
                query = query.filter(models.Simulation.user_id == user_id)
            
            total_simulations = query.count()
            successful_simulations = query.filter(
                models.Simulation.target_achieved == True
            ).count()
            
            # Calculate average metrics
            if total_simulations > 0:
                avg_target_value = db.query(
                    db.func.avg(models.Simulation.target_value)
                ).filter(
                    models.Simulation.user_id == user_id if user_id else True
                ).scalar() or 0
                
                avg_timeframe = db.query(
                    db.func.avg(models.Simulation.timeframe)
                ).filter(
                    models.Simulation.user_id == user_id if user_id else True
                ).scalar() or 0
                
                success_rate = (successful_simulations / total_simulations) * 100
            else:
                avg_target_value = 0
                avg_timeframe = 0
                success_rate = 0
            
            return {
                "total_simulations": total_simulations,
                "successful_simulations": successful_simulations,
                "success_rate": round(success_rate, 1),
                "average_target_value": round(avg_target_value, 2),
                "average_timeframe": round(avg_timeframe, 1)
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting statistics: {str(e)}")
            raise DatabaseError(
                f"Failed to get simulation statistics: {str(e)}",
                operation="select",
                table="simulations"
            )
    
    def _prepare_simulation_results(
        self,
        simulation_results: Dict[str, Any],
        stocks_data: List[Dict[str, Any]],
        shap_explanation: Optional[Dict[str, Any]],
        visualization_paths: Optional[Dict[str, str]],
        risk_score: int,
        risk_label: str
    ) -> Dict[str, Any]:
        """
        Prepare and enhance simulation results for database storage.
        
        Args:
            simulation_results: Raw simulation results
            stocks_data: Stock allocation data
            shap_explanation: SHAP explanation data
            visualization_paths: Visualization file paths
            risk_score: Risk tolerance score
            risk_label: Risk profile label
            
        Returns:
            Enhanced results dictionary ready for JSON serialization
        """
        # Start with base results
        enhanced_results = {
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
            "risk_label": risk_label
        }
        
        # Add timeline data if available
        timeline_data = simulation_results.get("timeline_data", [])
        contribution_data = simulation_results.get("contribution_data", [])
        
        if timeline_data or contribution_data:
            enhanced_results["timeline"] = {
                "portfolio": self._serialize_timeline_data(timeline_data),
                "contributions": self._serialize_timeline_data(contribution_data)
            }
        
        # Add asset breakdown
        asset_breakdown = simulation_results.get("asset_breakdown", {})
        if asset_breakdown:
            enhanced_results["breakdown"] = {
                str(k): float(v) for k, v in asset_breakdown.items()
            }
        
        # Add SHAP explanation if available
        if shap_explanation:
            enhanced_results["shap_explanation"] = self._serialize_shap_data(shap_explanation)
            enhanced_results["has_shap_explanations"] = True
        else:
            enhanced_results["has_shap_explanations"] = False
        
        # Add visualization paths if available
        if visualization_paths:
            enhanced_results["visualization_paths"] = visualization_paths
            enhanced_results["has_visualizations"] = True
            enhanced_results["visualization_count"] = len(visualization_paths)
        else:
            enhanced_results["has_visualizations"] = False
            enhanced_results["visualization_count"] = 0
        
        # Add metadata
        enhanced_results["metadata"] = {
            "created_timestamp": datetime.now().isoformat(),
            "methodology": "Enhanced AI-powered simulation with SHAP explanations",
            "version": "2.0",
            "enhanced_features": {
                "ai_recommendations": True,
                "shap_explanations": bool(shap_explanation),
                "visualizations": bool(visualization_paths),
                "goal_optimization": True
            }
        }
        
        return enhanced_results
    
    def _serialize_stocks_data(self, stocks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serialize stocks data for JSON storage."""
        serialized = []
        
        for stock in stocks_data:
            serialized_stock = {
                "symbol": str(stock.get("symbol", "")),
                "name": str(stock.get("name", "")),
                "allocation": float(stock.get("allocation", 0)),
                "explanation": str(stock.get("explanation", ""))
            }
            serialized.append(serialized_stock)
        
        return serialized
    
    def _serialize_timeline_data(self, timeline_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serialize timeline data for JSON storage."""
        serialized = []
        
        for entry in timeline_data:
            serialized_entry = {}
            
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    serialized_entry[key] = float(value)
                else:
                    serialized_entry[key] = str(value)
            
            serialized.append(serialized_entry)
        
        return serialized
    
    def _serialize_shap_data(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize SHAP explanation data for JSON storage."""
        serialized = {}
        
        for key, value in shap_explanation.items():
            if key == "feature_contributions" and isinstance(value, dict):
                # Serialize feature contributions
                serialized[key] = {
                    str(feature): float(contribution)
                    for feature, contribution in value.items()
                }
            elif key == "human_readable_explanation" and isinstance(value, dict):
                # Serialize explanations
                serialized[key] = {
                    str(feature): str(explanation)
                    for feature, explanation in value.items()
                }
            elif isinstance(value, (int, float)):
                serialized[key] = float(value)
            elif isinstance(value, dict):
                # Recursively serialize nested dictionaries
                serialized[key] = self._serialize_nested_dict(value)
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def _serialize_nested_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize nested dictionary data."""
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                serialized[str(key)] = float(value)
            elif isinstance(value, dict):
                serialized[str(key)] = self._serialize_nested_dict(value)
            elif isinstance(value, list):
                serialized[str(key)] = [
                    self._serialize_nested_dict(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
            else:
                serialized[str(key)] = str(value)
        
        return serialized


class SimulationResultsFormatter:
    """
    Formats simulation results for API responses.
    
    This class handles the conversion of database simulation records
    into properly formatted API responses with all necessary data.
    """
    
    def __init__(self):
        """Initialize the results formatter."""
        self.config = get_config()
    
    def format_simulation_response(self, simulation: models.Simulation) -> Dict[str, Any]:
        """
        Format a simulation database record for API response.
        
        Args:
            simulation: Simulation model instance
            
        Returns:
            Formatted simulation data
        """
        results = simulation.results or {}
        
        # Basic simulation information
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
            "created_at": simulation.created_at.isoformat() if simulation.created_at else None
        }
        
        # Enhanced results data
        response.update({
            "results": results,
            "has_shap_explanations": results.get("has_shap_explanations", False),
            "has_visualizations": results.get("has_visualizations", False),
            "visualization_count": results.get("visualization_count", 0),
            "methodology": results.get("metadata", {}).get("methodology", "Standard simulation"),
            "enhanced_features": results.get("metadata", {}).get("enhanced_features", {})
        })
        
        # Performance metrics
        performance_metrics = self._extract_performance_metrics(results)
        response["performance_metrics"] = performance_metrics
        
        # SHAP explanation summary
        if results.get("shap_explanation"):
            response["shap_summary"] = self._extract_shap_summary(results["shap_explanation"])
        
        # Visualization information
        if results.get("visualization_paths"):
            response["available_visualizations"] = list(results["visualization_paths"].keys())
        else:
            response["available_visualizations"] = []
        
        return response
    
    def format_simulation_list(self, simulations: List[models.Simulation]) -> List[Dict[str, Any]]:
        """
        Format a list of simulations for API response.
        
        Args:
            simulations: List of simulation model instances
            
        Returns:
            List of formatted simulation summaries
        """
        formatted_list = []
        
        for simulation in simulations:
            # Create summary version with key information only
            summary = {
                "id": simulation.id,
                "name": simulation.name,
                "goal": simulation.goal,
                "target_value": simulation.target_value,
                "target_achieved": simulation.target_achieved,
                "risk_label": simulation.risk_label,
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
                "has_shap_explanations": simulation.results.get("has_shap_explanations", False) if simulation.results else False,
                "has_visualizations": simulation.results.get("has_visualizations", False) if simulation.results else False
            }
            
            # Add basic performance metrics
            if simulation.results:
                summary.update({
                    "end_value": simulation.results.get("end_value", 0),
                    "total_return": simulation.results.get("total_return", 0),
                    "profit_loss": simulation.results.get("profit_loss", 0)
                })
            
            formatted_list.append(summary)
        
        return formatted_list
    
    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format performance metrics."""
        return {
            "starting_value": results.get("starting_value", 0),
            "ending_value": results.get("end_value", 0),
            "total_return": results.get("total_return", 0),
            "annualized_return": results.get("annualized_return", 0),
            "volatility": results.get("volatility", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "total_contributed": results.get("total_contributed", 0),
            "profit_loss": results.get("profit_loss", 0)
        }
    
    def _extract_shap_summary(self, shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SHAP explanation summary."""
        return {
            "portfolio_quality_score": shap_explanation.get("portfolio_quality_score", 0),
            "top_factors": self._get_top_shap_factors(shap_explanation),
            "overall_explanation": shap_explanation.get("human_readable_explanation", {}).get("overall", "")
        }
    
    def _get_top_shap_factors(self, shap_explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top contributing factors from SHAP explanation."""
        feature_contributions = shap_explanation.get("feature_contributions", {})
        
        # Sort by absolute contribution
        sorted_factors = sorted(
            feature_contributions.items(),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )
        
        # Return top 3 factors
        top_factors = []
        for feature, contribution in sorted_factors[:3]:
            top_factors.append({
                "feature": feature,
                "contribution": float(contribution),
                "impact": "positive" if float(contribution) > 0 else "negative"
            })
        
        return top_factors