"""
Database Manager Module

This module handles all database operations with proper JSON serialization
and enhanced error handling for complex simulation data.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from database import models

logger = logging.getLogger(__name__)

class SerializationManager:
    """
    Handles serialization of complex objects for database storage.
    
    This class manages the conversion of NumPy arrays, pandas objects,
    and other non-serializable objects to JSON-compatible formats.
    """
    
    @staticmethod
    def serialize_for_json(data: Any) -> Any:
        """
        Recursively convert objects to JSON-compatible types.
        
        Handles NumPy arrays, pandas objects, datetime objects, and complex nested structures.
        """
        
        # Handle None values
        if data is None:
            return None
        
        # Handle NumPy types
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.str_):
            return str(data)
        
        # Handle pandas types
        elif isinstance(data, pd.Series):
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        
        # Handle datetime objects
        elif isinstance(data, datetime):
            return data.isoformat()
        
        # Handle complex numbers
        elif isinstance(data, complex):
            return {"real": data.real, "imag": data.imag}
        
        # Handle dictionaries
        elif isinstance(data, dict):
            return {str(key): SerializationManager.serialize_for_json(value) for key, value in data.items()}
        
        # Handle lists and tuples
        elif isinstance(data, (list, tuple)):
            return [SerializationManager.serialize_for_json(item) for item in data]
        
        # Handle sets
        elif isinstance(data, set):
            return list(data)
        
        # Handle custom objects with __dict__
        elif hasattr(data, '__dict__'):
            return SerializationManager.serialize_for_json(data.__dict__)
        
        # Handle objects with a to_dict method
        elif hasattr(data, 'to_dict'):
            return SerializationManager.serialize_for_json(data.to_dict())
        
        # Return as-is for basic JSON-serializable types
        elif isinstance(data, (str, int, float, bool)):
            return data
        
        # For anything else, try to convert to string as fallback
        else:
            try:
                return str(data)
            except Exception:
                return f"<non-serializable: {type(data).__name__}>"
    
    @staticmethod
    def clean_shap_explanation(shap_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Specifically clean SHAP explanation data for JSON serialization.
        """
        if not shap_data:
            return {}
        
        try:
            cleaned_shap = {}
            
            # Handle common SHAP fields
            for key, value in shap_data.items():
                if key == 'shap_values':
                    # SHAP values are typically NumPy arrays
                    cleaned_shap[key] = SerializationManager.serialize_for_json(value)
                elif key == 'feature_importance':
                    # Feature importance scores
                    cleaned_shap[key] = SerializationManager.serialize_for_json(value)
                elif key == 'expected_value':
                    # Expected value (baseline)
                    cleaned_shap[key] = float(value) if value is not None else None
                elif key == 'feature_names':
                    # Feature names should be strings
                    cleaned_shap[key] = [str(name) for name in value] if value else []
                elif key == 'human_readable_explanation':
                    # Text explanations
                    cleaned_shap[key] = {str(k): str(v) for k, v in value.items()} if value else {}
                elif key == 'portfolio_quality_score':
                    # Quality score
                    cleaned_shap[key] = float(value) if value is not None else None
                elif key == 'confidence_score':
                    # Confidence score
                    cleaned_shap[key] = float(value) if value is not None else None
                else:
                    # Generic cleaning for other fields
                    cleaned_shap[key] = SerializationManager.serialize_for_json(value)
            
            return cleaned_shap
            
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning SHAP explanation: {e}")
            return {"error": f"SHAP data cleaning failed: {str(e)}"}
    
    @staticmethod
    def test_json_serialization(data: Any, description: str = "data") -> bool:
        """
        Test if data can be JSON serialized.
        """
        try:
            json.dumps(data)
            logger.debug(f"✅ {description} is JSON serializable")
            return True
        except Exception as e:
            logger.error(f"❌ {description} is NOT JSON serializable: {e}")
            return False

class DatabaseManager:
    """
    Manages all database operations for portfolio simulations.
    
    Features:
    - Enhanced serialization handling
    - Robust error recovery
    - Support for both basic and enhanced simulations
    """
    
    def __init__(self):
        """Initialize the database manager."""
        self.serialization = SerializationManager()
        logger.info("💾 DatabaseManager initialized")
    
    async def save_enhanced_simulation(self, db, sim_input, user_data, risk_profile, 
                                     ai_summary, stocks_picked, simulation_results, 
                                     goal_analysis, recommendation_result, 
                                     shap_explanations=None):
        """
        Save enhanced simulation with proper SHAP data handling.
        """
        try:
            logger.info("💾 Saving enhanced simulation with advanced features")
            
            # Ensure we have SHAP explanations
            if not shap_explanations:
                # Try to extract from recommendation_result
                if 'shap_explanations' in recommendation_result:
                    shap_explanations = recommendation_result['shap_explanations']
                elif 'shap_explanation' in recommendation_result:
                    shap_explanations = recommendation_result['shap_explanation']
                else:
                    shap_explanations = {}
            
            # Clean simulation results for database storage
            cleaned_results = self.serialization_manager.clean_for_database(simulation_results)
            
            # Prepare comprehensive results structure
            enhanced_results = {
                # Core simulation data
                'portfolio_performance': cleaned_results.get('portfolio_performance', {}),
                'target_reached': cleaned_results.get('target_reached', False),
                'final_value': cleaned_results.get('final_value', 0),
                'total_return': cleaned_results.get('total_return', 0),
                'annualized_return': cleaned_results.get('annualized_return', 0),
                
                # Enhanced features
                'goal_analysis': goal_analysis,
                'market_crash_analysis': cleaned_results.get('market_crash_analysis', {}),
                'stocks_picked': stocks_picked,
                
                # 🔍 CRITICAL: Include SHAP explanations
                'shap_explanations': shap_explanations,
                'shap_explanation': shap_explanations,  # Also save as singular for compatibility
                
                # Portfolio recommendations
                'portfolio_recommendations': {
                    **recommendation_result,
                    'shap_explanations': shap_explanations  # Ensure it's also here
                },
                
                # Metadata
                'wealthwise_enhanced': True,
                'methodology': 'Enhanced WealthWise Portfolio Simulation with SHAP',
                'simulation_timestamp': datetime.utcnow().isoformat(),
                'has_shap_data': bool(shap_explanations)
            }
            
            # Log what we're saving
            logger.info(f"💾 Saving SHAP data: {bool(shap_explanations)}")
            if shap_explanations:
                logger.info(f"🔍 SHAP data type: {type(shap_explanations)}")
                if isinstance(shap_explanations, dict):
                    logger.info(f"🔍 SHAP keys: {list(shap_explanations.keys())}")
            
            # Create simulation model
            from database import models
            
            simulation = models.Simulation(
                user_id=sim_input['user_id'],
                name=sim_input['name'],
                goal=sim_input['goal'],
                target_value=sim_input['target_value'],
                lump_sum=sim_input['lump_sum'],
                monthly=sim_input['monthly'],
                timeframe=sim_input['timeframe'],
                target_achieved=enhanced_results['target_reached'],
                income_bracket=sim_input['income_bracket'],
                risk_score=sim_input['risk_score'],
                risk_label=sim_input['risk_label'],
                ai_summary=ai_summary,
                results=enhanced_results,  # This includes SHAP
                created_at=datetime.utcnow()
            )
            
            db.add(simulation)
            db.commit()
            db.refresh(simulation)
            
            logger.info(f"✅ Enhanced simulation saved successfully (ID: {simulation.id})")
            
            # Verify SHAP data was saved
            if simulation.results and 'shap_explanations' in simulation.results:
                logger.info("✅ SHAP explanations confirmed saved to database")
            else:
                logger.warning("⚠️ SHAP explanations may not have been saved properly")
            
            return simulation
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced simulation: {e}")
            db.rollback()
        
            # Try to save a basic version if enhanced save fails
            return await self._save_basic_fallback(
                db, sim_input, user_data, risk_profile, ai_summary, 
                stocks_picked, simulation_results, str(e)
            )
    
    async def save_basic_simulation(self, db, sim_input: Dict[str, Any], 
                                  user_data: Dict[str, Any], risk_profile: Dict[str, Any],
                                  ai_summary: str, stocks_picked: List[Dict], 
                                  simulation_results: Dict[str, Any]) -> models.Simulation:
        """
        Save basic simulation without enhanced features.
        
        Args:
            db: Database session
            sim_input: Original simulation input
            user_data: Processed user data
            risk_profile: Risk assessment results
            ai_summary: AI-generated summary
            stocks_picked: Selected stocks with allocations
            simulation_results: Portfolio simulation results
            
        Returns:
            Saved simulation model instance
        """
        
        try:
            logger.info("💾 Saving basic simulation")
            
            target_reached = simulation_results["end_value"] >= user_data["target_value"]
            
            # Create basic results object
            basic_results = {
                "name": user_data["goal"],
                "stocks_picked": stocks_picked,
                "starting_value": simulation_results["starting_value"],
                "end_value": simulation_results["end_value"],
                "return": simulation_results["portfolio_return"],
                "target_reached": target_reached,
                "risk_score": risk_profile["score"],
                "risk_label": risk_profile["label"],
                "timeline": simulation_results["timeline"],
                "simulation_metadata": simulation_results.get("simulation_metadata"),
                "methodology": "Basic portfolio simulation",
                "created_timestamp": datetime.now().isoformat()
            }
            
            # Clean the results for database storage
            cleaned_results = self._clean_simulation_results_for_db(basic_results)
            
            # Create simulation model
            simulation = models.Simulation(
                user_id=sim_input.get("user_id"),
                name=user_data["goal"],
                goal=user_data["goal"],
                target_value=user_data["target_value"],
                lump_sum=user_data["lump_sum"],
                monthly=user_data["monthly"],
                timeframe=user_data["timeframe"],
                target_achieved=target_reached,
                income_bracket=user_data["income_bracket"],
                risk_score=risk_profile["score"],
                risk_label=risk_profile["label"],
                ai_summary=ai_summary,
                results=cleaned_results
            )
            
            # Save to database
            db.add(simulation)
            db.commit()
            db.refresh(simulation)
            
            logger.info(f"✅ Basic simulation saved successfully (ID: {simulation.id})")
            return simulation
            
        except Exception as e:
            logger.error(f"❌ Error saving basic simulation: {str(e)}")
            db.rollback()
            raise
    
    def _clean_simulation_results_for_db(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all simulation results before saving to database.
        
        This is the main function to call before saving any simulation results
        that might contain NumPy arrays or other non-serializable objects.
        """
        try:
            logger.info("🧹 Cleaning simulation results for database storage")
            
            # Create a deep copy to avoid modifying the original
            import copy
            cleaned_results = copy.deepcopy(results)
            
            # Special handling for known problematic fields
            if 'shap_explanation' in cleaned_results:
                cleaned_results['shap_explanation'] = self.serialization.clean_shap_explanation(
                    cleaned_results['shap_explanation']
                )
            
            if 'goal_analysis' in cleaned_results:
                cleaned_results['goal_analysis'] = self.serialization.serialize_for_json(
                    cleaned_results['goal_analysis']
                )
            
            if 'recommendation_result' in cleaned_results:
                cleaned_results['recommendation_result'] = self.serialization.serialize_for_json(
                    cleaned_results['recommendation_result']
                )
            
            if 'market_crash_analysis' in cleaned_results:
                cleaned_results['market_crash_analysis'] = self.serialization.serialize_for_json(
                    cleaned_results['market_crash_analysis']
                )
            
            if 'stocks_picked' in cleaned_results:
                # Clean stock allocation data
                cleaned_stocks = []
                for stock in cleaned_results['stocks_picked']:
                    cleaned_stock = {
                        'symbol': str(stock.get('symbol', '')),
                        'name': str(stock.get('name', '')),
                        'allocation': float(stock.get('allocation', 0)),
                        'explanation': str(stock.get('explanation', ''))
                    }
                    cleaned_stocks.append(cleaned_stock)
                cleaned_results['stocks_picked'] = cleaned_stocks
            
            if 'timeline' in cleaned_results:
                # Clean timeline data
                timeline = cleaned_results['timeline']
                if isinstance(timeline, dict):
                    for key, values in timeline.items():
                        if isinstance(values, list):
                            cleaned_timeline = []
                            for item in values:
                                if isinstance(item, dict):
                                    cleaned_item = {
                                        'date': str(item.get('date', '')),
                                        'value': float(item.get('value', 0))
                                    }
                                    cleaned_timeline.append(cleaned_item)
                                else:
                                    cleaned_timeline.append(self.serialization.serialize_for_json(item))
                            timeline[key] = cleaned_timeline
            
            # Apply general serialization to the entire structure
            cleaned_results = self.serialization.serialize_for_json(cleaned_results)
            
            # Test that the result is actually JSON serializable
            json.dumps(cleaned_results)
            
            logger.info("✅ Simulation results successfully cleaned for database")
            return cleaned_results
            
        except Exception as e:
            logger.error(f"❌ Failed to clean simulation results: {e}")
            return self._create_fallback_results(results, e)
    
    def _create_fallback_results(self, original_results: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        Create a safe fallback structure when serialization fails.
        """
        
        try:
            # Extract basic safe values
            fallback_results = {
                "basic_info": {
                    "starting_value": float(original_results.get("starting_value", 0)),
                    "end_value": float(original_results.get("end_value", 0)),
                    "portfolio_return": float(original_results.get("return", 0)),
                    "target_reached": bool(original_results.get("target_reached", False))
                },
                "portfolio_summary": {
                    "stocks": [str(stock.get('symbol', '')) for stock in original_results.get('stocks_picked', [])],
                    "num_stocks": len(original_results.get('stocks_picked', []))
                },
                "metadata": {
                    "enhanced_features": bool(original_results.get("wealthwise_enhanced", False)),
                    "methodology": str(original_results.get("methodology", "Unknown")),
                    "serialization_note": f"Full results not serializable: {str(error)}"
                },
                "error_info": {
                    "original_error": str(error),
                    "fallback_used": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Test the fallback is serializable
            json.dumps(fallback_results)
            logger.warning("⚠️ Using fallback results structure due to serialization error")
            return fallback_results
            
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback serialization failed: {fallback_error}")
            # Return absolute minimum
            return {
                "status": "serialization_failed",
                "error": str(error),
                "fallback_error": str(fallback_error),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _save_basic_fallback(self, db, sim_input: Dict[str, Any], 
                                 user_data: Dict[str, Any], risk_profile: Dict[str, Any],
                                 ai_summary: str, stocks_picked: List[Dict], 
                                 simulation_results: Dict[str, Any], error: str) -> models.Simulation:
        """
        Save a basic simulation when enhanced save fails.
        """
        
        try:
            logger.warning("🔄 Attempting to save basic simulation after enhanced save failed")
            
            target_reached = simulation_results["end_value"] >= user_data["target_value"]
            
            basic_results = {
                "name": user_data["goal"],
                "stocks_picked": [
                    {
                        "symbol": str(stock.get("symbol", "")),
                        "name": str(stock.get("name", "")),
                        "allocation": float(stock.get("allocation", 0))
                    }
                    for stock in stocks_picked
                ],
                "starting_value": float(simulation_results["starting_value"]),
                "end_value": float(simulation_results["end_value"]),
                "return": float(simulation_results["portfolio_return"]),
                "target_reached": target_reached,
                "risk_score": risk_profile["score"],
                "risk_label": risk_profile["label"],
                "enhanced_save_failed": True,
                "error_message": error,
                "methodology": "Basic fallback simulation"
            }
            
            basic_simulation = models.Simulation(
                user_id=sim_input.get("user_id"),
                name=user_data["goal"],
                goal=user_data["goal"],
                target_value=user_data["target_value"],
                lump_sum=user_data["lump_sum"],
                monthly=user_data["monthly"],
                timeframe=user_data["timeframe"],
                target_achieved=target_reached,
                income_bracket=user_data["income_bracket"],
                risk_score=risk_profile["score"],
                risk_label=risk_profile["label"],
                ai_summary=ai_summary,
                results=basic_results
            )
            
            db.add(basic_simulation)
            db.commit()
            db.refresh(basic_simulation)
            
            logger.warning(f"⚠️ Saved basic fallback simulation (ID: {basic_simulation.id})")
            return basic_simulation
            
        except Exception as basic_error:
            logger.error(f"❌ Even basic fallback simulation save failed: {basic_error}")
            db.rollback()
            raise
    
    def get_simulation_by_id(self, db, simulation_id: int) -> Optional[models.Simulation]:
        """
        Retrieve a simulation by ID.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to retrieve
            
        Returns:
            Simulation model instance or None if not found
        """
        
        try:
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            if simulation:
                logger.info(f"✅ Retrieved simulation ID: {simulation_id}")
            else:
                logger.warning(f"⚠️ Simulation not found: {simulation_id}")
            
            return simulation
            
        except Exception as e:
            logger.error(f"❌ Error retrieving simulation {simulation_id}: {e}")
            return None
    
    def get_user_simulations(self, db, user_id: str, limit: int = 10) -> List[models.Simulation]:
        """
        Get all simulations for a specific user.
        
        Args:
            db: Database session
            user_id: User identifier
            limit: Maximum number of simulations to return
            
        Returns:
            List of simulation model instances
        """
        
        try:
            simulations = db.query(models.Simulation).filter(
                models.Simulation.user_id == user_id
            ).order_by(models.Simulation.created_at.desc()).limit(limit).all()
            
            logger.info(f"✅ Retrieved {len(simulations)} simulations for user {user_id}")
            return simulations
            
        except Exception as e:
            logger.error(f"❌ Error retrieving simulations for user {user_id}: {e}")
            return []
    
    def update_simulation_results(self, db, simulation_id: int, 
                                updated_results: Dict[str, Any]) -> bool:
        """
        Update simulation results (e.g., add crash analysis later).
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to update
            updated_results: New results data to merge
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            simulation = self.get_simulation_by_id(db, simulation_id)
            
            if not simulation:
                logger.error(f"❌ Cannot update - simulation {simulation_id} not found")
                return False
            
            # Merge updated results with existing results
            current_results = simulation.results or {}
            current_results.update(updated_results)
            
            # Clean the merged results
            cleaned_results = self._clean_simulation_results_for_db(current_results)
            
            # Update the simulation
            simulation.results = cleaned_results
            db.commit()
            
            logger.info(f"✅ Updated simulation {simulation_id} with new results")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating simulation {simulation_id}: {e}")
            db.rollback()
            return False
    
    def delete_simulation(self, db, simulation_id: int, user_id: str = None) -> bool:
        """
        Delete a simulation.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to delete
            user_id: Optional user ID for authorization check
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            query = db.query(models.Simulation).filter(models.Simulation.id == simulation_id)
            
            # Add user authorization check if provided
            if user_id:
                query = query.filter(models.Simulation.user_id == user_id)
            
            simulation = query.first()
            
            if not simulation:
                logger.warning(f"⚠️ Simulation {simulation_id} not found or not authorized")
                return False
            
            db.delete(simulation)
            db.commit()
            
            logger.info(f"✅ Deleted simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting simulation {simulation_id}: {e}")
            db.rollback()
            return False
    
    def get_simulation_statistics(self, db, user_id: str = None) -> Dict[str, Any]:
        """
        Get statistics about simulations in the database.
        
        Args:
            db: Database session
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary with simulation statistics
        """
        
        try:
            query = db.query(models.Simulation)
            
            if user_id:
                query = query.filter(models.Simulation.user_id == user_id)
            
            simulations = query.all()
            
            if not simulations:
                return {"total_simulations": 0}
            
            # Calculate statistics
            total_simulations = len(simulations)
            enhanced_simulations = len([s for s in simulations if s.results.get("wealthwise_enhanced")])
            successful_simulations = len([s for s in simulations if s.target_achieved])
            
            # Average values
            avg_target_value = sum(s.target_value for s in simulations) / total_simulations
            avg_timeframe = sum(s.timeframe for s in simulations) / total_simulations
            
            # Risk distribution
            risk_distribution = {}
            for simulation in simulations:
                risk_label = simulation.risk_label or "Unknown"
                risk_distribution[risk_label] = risk_distribution.get(risk_label, 0) + 1
            
            return {
                "total_simulations": total_simulations,
                "enhanced_simulations": enhanced_simulations,
                "successful_simulations": successful_simulations,
                "success_rate_percent": round((successful_simulations / total_simulations) * 100, 1),
                "average_target_value": round(avg_target_value, 2),
                "average_timeframe_years": round(avg_timeframe, 1),
                "risk_distribution": risk_distribution,
                "enhancement_rate_percent": round((enhanced_simulations / total_simulations) * 100, 1)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating simulation statistics: {e}")
            return {"error": str(e)}
    
    def export_simulation_data(self, db, simulation_id: int) -> Optional[Dict[str, Any]]:
        """
        Export simulation data in a format suitable for external analysis.
        
        Args:
            db: Database session
            simulation_id: ID of the simulation to export
            
        Returns:
            Exportable simulation data or None if not found
        """
        
        try:
            simulation = self.get_simulation_by_id(db, simulation_id)
            
            if not simulation:
                return None
            
            export_data = {
                "simulation_metadata": {
                    "id": simulation.id,
                    "name": simulation.name,
                    "goal": simulation.goal,
                    "created_at": simulation.created_at.isoformat() if simulation.created_at else None,
                    "methodology": simulation.results.get("methodology", "Unknown")
                },
                "parameters": {
                    "target_value": simulation.target_value,
                    "lump_sum": simulation.lump_sum,
                    "monthly_contribution": simulation.monthly,
                    "timeframe_years": simulation.timeframe,
                    "risk_score": simulation.risk_score,
                    "risk_label": simulation.risk_label
                },
                "results": {
                    "starting_value": simulation.results.get("starting_value"),
                    "end_value": simulation.results.get("end_value"),
                    "portfolio_return": simulation.results.get("return"),
                    "target_achieved": simulation.target_achieved,
                    "stocks_picked": simulation.results.get("stocks_picked", [])
                },
                "timeline_data": simulation.results.get("timeline", {}),
                "enhanced_features": {
                    "has_crash_analysis": bool(simulation.results.get("market_crash_analysis")),
                    "has_goal_analysis": bool(simulation.results.get("goal_analysis")),
                    "has_shap_explanations": bool(simulation.results.get("shap_explanation")),
                    "wealthwise_enhanced": simulation.results.get("wealthwise_enhanced", False)
                }
            }
            
            logger.info(f"✅ Exported data for simulation {simulation_id}")
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Error exporting simulation {simulation_id}: {e}")
            return None