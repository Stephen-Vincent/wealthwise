"""
Main Portfolio Simulator Service - Orchestrates the complete simulation workflow.

This module provides the main service class that coordinates all components
to deliver complete portfolio simulation functionality with AI recommendations,
SHAP explanations, and visualizations.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from .config import get_config, initialize_config
from .exceptions import PortfolioSimulatorError, ValidationError
from .validators import InputValidator
from .data_provider import MarketDataProvider, DataQualityAnalyzer
from .portfolio_simulator import PortfolioSimulator, PortfolioWeightCalculator
from .ai_recommendation_service import AIRecommendationService, SHAPDataProcessor
from .visualization_service import VisualizationService, ChartDataGenerator
from .database_service import DatabaseService, SimulationResultsFormatter

logger = logging.getLogger(__name__)


class PortfolioSimulatorService:
    """
    Main service orchestrating the complete portfolio simulation workflow.
    
    This service coordinates all components to provide:
    1. Input validation and sanitization
    2. AI-powered stock recommendations with SHAP explanations
    3. Market data retrieval and processing
    4. Portfolio simulation with multiple scenarios
    5. Visualization generation (static and interactive)
    6. Database storage and retrieval
    7. Comprehensive error handling and logging
    """
    
    def __init__(self):
        """Initialize the portfolio simulator service with all components."""
        # Initialize configuration
        self.config = get_config()
        
        # Initialize core components
        self.validator = InputValidator()
        self.data_provider = MarketDataProvider(self.validator)
        self.data_quality_analyzer = DataQualityAnalyzer()
        self.portfolio_simulator = PortfolioSimulator(self.validator)
        self.weight_calculator = PortfolioWeightCalculator(self.validator)
        self.ai_service = AIRecommendationService(self.validator)
        self.visualization_service = VisualizationService(self.validator)
        self.chart_data_generator = ChartDataGenerator()
        self.database_service = DatabaseService(self.validator)
        self.results_formatter = SimulationResultsFormatter()
        self.shap_processor = SHAPDataProcessor()
        
        logger.info("Portfolio Simulator Service initialized successfully")
    
    async def simulate_portfolio(self, simulation_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Execute complete portfolio simulation workflow.
        
        Args:
            simulation_input: Raw input data from user
            db: Database session
            
        Returns:
            Complete simulation results with all enhancements
            
        Raises:
            PortfolioSimulatorError: If simulation fails at any stage
        """
        try:
            logger.info("Starting enhanced portfolio simulation workflow")
            
            # Step 1: Validate and sanitize input
            validated_input = self.validator.validate_simulation_input(simulation_input)
            logger.info("Input validation completed successfully")
            
            # Step 2: Get AI-powered stock recommendations
            recommendations = await self._get_stock_recommendations(validated_input)
            logger.info(f"AI recommendations obtained: {len(recommendations['stocks'])} stocks")
            
            # Step 3: Download and validate market data
            market_data = await self._get_market_data(recommendations['stocks'], validated_input)
            logger.info(f"Market data retrieved: {market_data.shape}")
            
            # Step 4: Calculate optimal portfolio weights
            portfolio_weights = self._calculate_portfolio_weights(
                market_data, validated_input, recommendations
            )
            logger.info("Portfolio weights calculated")
            
            # Step 5: Run portfolio simulation
            simulation_results = await self._run_simulation(
                market_data, portfolio_weights, validated_input
            )
            logger.info("Portfolio simulation completed")
            
            # Step 6: Process SHAP explanations
            processed_shap = self._process_shap_explanations(recommendations)
            
            # Step 7: Generate visualizations
            visualization_paths = await self._generate_visualizations(
                simulation_results, recommendations, processed_shap, validated_input
            )
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            
            # Step 8: Generate AI summary
            ai_summary = await self._generate_ai_summary(
                simulation_results, recommendations, processed_shap, validated_input
            )
            
            # Step 9: Save to database
            saved_simulation = self._save_simulation(
                db, validated_input, simulation_results, recommendations,
                ai_summary, processed_shap, visualization_paths
            )
            
            # Step 10: Generate chart data for frontend
            chart_data = self._generate_chart_data(
                simulation_results, recommendations, processed_shap
            )
            
            # Step 11: Format final response
            response = self._format_final_response(
                saved_simulation, chart_data, visualization_paths
            )
            
            logger.info(f"Simulation workflow completed successfully (ID: {saved_simulation.id})")
            return response
            
        except Exception as e:
            logger.error(f"Portfolio simulation workflow failed: {str(e)}")
            if isinstance(e, PortfolioSimulatorError):
                raise
            else:
                raise PortfolioSimulatorError(
                    f"Simulation workflow failed: {str(e)}",
                    details={"stage": "workflow_orchestration"}
                )
    
    async def get_simulation_chart_data(self, simulation_id: int, db: Session) -> Dict[str, Any]:
        """
        Get chart data for an existing simulation.
        
        Args:
            simulation_id: ID of the simulation
            db: Database session
            
        Returns:
            Chart data suitable for frontend visualization
            
        Raises:
            PortfolioSimulatorError: If data retrieval fails
        """
        try:
            # Retrieve simulation from database
            simulation = self.database_service.get_simulation(db, simulation_id)
            
            if not simulation:
                raise PortfolioSimulatorError(
                    f"Simulation with ID {simulation_id} not found",
                    error_code="SIMULATION_NOT_FOUND"
                )
            
            # Extract data from simulation results
            results = simulation.results or {}
            stocks_data = results.get("stocks_picked", [])
            shap_explanation = results.get("shap_explanation")
            
            # Generate chart data
            chart_data = self.chart_data_generator.generate_chart_data(
                results, stocks_data, shap_explanation
            )
            
            return {
                "success": True,
                "simulation_id": simulation_id,
                "chart_data": chart_data,
                "metadata": {
                    "has_shap": bool(shap_explanation),
                    "visualization_count": results.get("visualization_count", 0),
                    "generated_at": results.get("metadata", {}).get("created_timestamp")
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get chart data for simulation {simulation_id}: {str(e)}")
            if isinstance(e, PortfolioSimulatorError):
                raise
            else:
                raise PortfolioSimulatorError(
                    f"Failed to retrieve chart data: {str(e)}",
                    error_code="CHART_DATA_RETRIEVAL_FAILED"
                )
    
    async def get_user_simulations(
        self, 
        user_id: int, 
        db: Session, 
        limit: int = 10, 
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get simulations for a specific user.
        
        Args:
            user_id: ID of the user
            db: Database session
            limit: Maximum number of simulations to return
            offset: Number of simulations to skip
            
        Returns:
            List of user simulations with summary data
        """
        try:
            simulations = self.database_service.get_user_simulations(
                db, user_id, limit, offset
            )
            
            formatted_simulations = self.results_formatter.format_simulation_list(simulations)
            
            # Get user statistics
            statistics = self.database_service.get_simulation_statistics(db, user_id)
            
            return {
                "simulations": formatted_simulations,
                "statistics": statistics,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(simulations)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get user simulations: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to retrieve user simulations: {str(e)}",
                error_code="USER_SIMULATIONS_RETRIEVAL_FAILED"
            )
    
    # Private methods for workflow steps
    
    async def _get_stock_recommendations(self, validated_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered stock recommendations."""
        try:
            recommendations = await self.ai_service.get_recommendations(
                target_value=validated_input["target_value"],
                timeframe_years=validated_input["timeframe"],
                risk_score=validated_input["risk_score"],
                risk_label=validated_input["risk_label"],
                current_investment=validated_input["lump_sum"],
                monthly_contribution=validated_input["monthly"]
            )
            
            if not recommendations.get("stocks"):
                raise PortfolioSimulatorError(
                    "No stock recommendations could be generated",
                    error_code="NO_RECOMMENDATIONS"
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Stock recommendation failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to get stock recommendations: {str(e)}",
                error_code="RECOMMENDATION_FAILED"
            )
    
    async def _get_market_data(self, stocks: List[str], validated_input: Dict[str, Any]) -> Any:
        """Download and validate market data."""
        try:
            # Download historical data
            market_data = self.data_provider.download_stock_data(
                stocks, validated_input["timeframe"]
            )
            
            # Analyze data quality
            quality_analysis = self.data_quality_analyzer.analyze_data_quality(market_data)
            
            if quality_analysis["overall_quality"] == "poor":
                logger.warning("Poor data quality detected")
                # Could implement fallback logic here
            
            return market_data
            
        except Exception as e:
            logger.error(f"Market data retrieval failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to retrieve market data: {str(e)}",
                error_code="DATA_RETRIEVAL_FAILED"
            )
    
    def _calculate_portfolio_weights(
        self, 
        market_data: Any, 
        validated_input: Dict[str, Any], 
        recommendations: Dict[str, Any]
    ) -> Any:
        """Calculate optimal portfolio weights."""
        try:
            weights = self.weight_calculator.calculate_weights(
                tickers=list(market_data.columns),
                risk_score=validated_input["risk_score"],
                risk_label=validated_input["risk_label"],
                data=market_data
            )
            
            return weights
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to calculate portfolio weights: {str(e)}",
                error_code="WEIGHT_CALCULATION_FAILED"
            )
    
    async def _run_simulation(
        self, 
        market_data: Any, 
        weights: Any, 
        validated_input: Dict[str, Any]
    ) -> Any:
        """Run the portfolio simulation."""
        try:
            simulation_results = self.portfolio_simulator.simulate_growth(
                data=market_data,
                weights=weights,
                lump_sum=validated_input["lump_sum"],
                monthly_contribution=validated_input["monthly"],
                timeframe_years=validated_input["timeframe"]
            )
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Portfolio simulation failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Portfolio simulation failed: {str(e)}",
                error_code="SIMULATION_FAILED"
            )
    
    def _process_shap_explanations(self, recommendations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process SHAP explanations if available."""
        try:
            shap_explanation = recommendations.get("shap_explanation")
            
            if shap_explanation:
                processed_shap = self.shap_processor.clean_shap_explanation(shap_explanation)
                return processed_shap
            
            return None
            
        except Exception as e:
            logger.warning(f"SHAP processing failed: {str(e)}")
            return None
    
    async def _generate_visualizations(
        self,
        simulation_results: Any,
        recommendations: Dict[str, Any],
        processed_shap: Optional[Dict[str, Any]],
        validated_input: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate visualization files."""
        try:
            # Prepare stocks data for visualization
            stocks_data = self._prepare_stocks_data_for_viz(
                recommendations["stocks"], simulation_results.asset_breakdown
            )
            
            visualization_paths = await self.visualization_service.create_simulation_visualizations(
                simulation_id=None,  # Will be updated after database save
                stocks_data=stocks_data,
                simulation_results=simulation_results.__dict__,
                shap_explanation=processed_shap,
                user_data=validated_input
            )
            
            return visualization_paths
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
            return {}
    
    async def _generate_ai_summary(
        self,
        simulation_results: Any,
        recommendations: Dict[str, Any],
        processed_shap: Optional[Dict[str, Any]],
        validated_input: Dict[str, Any]
    ) -> str:
        """Generate AI-powered summary of results."""
        try:
            # Create a comprehensive summary using available data
            summary_parts = []
            
            # Basic performance summary
            performance = simulation_results.portfolio_metrics
            summary_parts.append(
                f"Your {validated_input['risk_label'].lower()} risk portfolio "
                f"grew from £{performance.starting_value:,.2f} to £{performance.ending_value:,.2f} "
                f"over {validated_input['timeframe']} years, achieving a "
                f"{performance.total_return:.1f}% total return."
            )
            
            # Goal achievement
            target_achieved = performance.ending_value >= validated_input["target_value"]
            summary_parts.append(
                f"Your target of £{validated_input['target_value']:,.2f} was "
                f"{'achieved' if target_achieved else 'partially achieved'}."
            )
            
            # SHAP insights if available
            if processed_shap and processed_shap.get("human_readable_explanation"):
                explanations = processed_shap["human_readable_explanation"]
                top_explanation = next(iter(explanations.values()), "")
                if top_explanation:
                    summary_parts.append(f"Key insight: {top_explanation}")
            
            # Risk metrics
            if performance.volatility > 0:
                summary_parts.append(
                    f"Portfolio volatility was {performance.volatility:.1f}% annually "
                    f"with a Sharpe ratio of {performance.sharpe_ratio:.2f}."
                )
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"AI summary generation failed: {str(e)}")
            return "Portfolio simulation completed successfully with AI-enhanced recommendations."
    
    def _save_simulation(
        self,
        db: Session,
        validated_input: Dict[str, Any],
        simulation_results: Any,
        recommendations: Dict[str, Any],
        ai_summary: str,
        processed_shap: Optional[Dict[str, Any]],
        visualization_paths: Dict[str, str]
    ) -> Any:
        """Save simulation to database."""
        try:
            # Prepare stocks data
            stocks_data = self._prepare_stocks_data_for_db(
                recommendations["stocks"], simulation_results.asset_breakdown
            )
            
            # Convert simulation results to dictionary format
            results_dict = {
                "starting_value": simulation_results.portfolio_metrics.starting_value,
                "ending_value": simulation_results.portfolio_metrics.ending_value,
                "total_return": simulation_results.portfolio_metrics.total_return,
                "annualized_return": simulation_results.portfolio_metrics.annualized_return,
                "volatility": simulation_results.portfolio_metrics.volatility,
                "sharpe_ratio": simulation_results.portfolio_metrics.sharpe_ratio,
                "max_drawdown": simulation_results.portfolio_metrics.max_drawdown,
                "total_contributed": simulation_results.portfolio_metrics.total_contributed,
                "profit_loss": simulation_results.portfolio_metrics.profit_loss,
                "timeline_data": simulation_results.timeline_data,
                "contribution_data": simulation_results.contribution_data,
                "asset_breakdown": simulation_results.asset_breakdown
            }
            
            saved_simulation = self.database_service.save_simulation(
                db=db,
                simulation_input=validated_input,
                user_data=validated_input,
                simulation_results=results_dict,
                ai_summary=ai_summary,
                stocks_data=stocks_data,
                risk_score=validated_input["risk_score"],
                risk_label=validated_input["risk_label"],
                shap_explanation=processed_shap,
                visualization_paths=visualization_paths
            )
            
            return saved_simulation
            
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to save simulation: {str(e)}",
                error_code="DATABASE_SAVE_FAILED"
            )
    
    def _generate_chart_data(
        self,
        simulation_results: Any,
        recommendations: Dict[str, Any],
        processed_shap: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate chart data for frontend."""
        try:
            # Prepare stocks data
            stocks_data = self._prepare_stocks_data_for_viz(
                recommendations["stocks"], simulation_results.asset_breakdown
            )
            
            # Convert simulation results to dictionary format
            results_dict = {
                "timeline": {
                    "portfolio": simulation_results.timeline_data,
                    "contributions": simulation_results.contribution_data
                }
            }
            
            chart_data = self.chart_data_generator.generate_chart_data(
                results_dict, stocks_data, processed_shap
            )
            
            return chart_data
            
        except Exception as e:
            logger.warning(f"Chart data generation failed: {str(e)}")
            return {}
    
    def _format_final_response(
        self,
        saved_simulation: Any,
        chart_data: Dict[str, Any],
        visualization_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """Format the final response."""
        formatted_response = self.results_formatter.format_simulation_response(saved_simulation)
        
        # Add chart data and additional metadata
        formatted_response.update({
            "chart_data": chart_data,
            "success": True,
            "workflow_completed": True,
            "enhanced_features_enabled": {
                "ai_recommendations": True,
                "shap_explanations": bool(saved_simulation.results.get("shap_explanation")),
                "visualizations": bool(visualization_paths),
                "interactive_charts": bool(chart_data)
            }
        })
        
        return formatted_response
    
    def _prepare_stocks_data_for_viz(self, stocks: List[str], asset_breakdown: Dict[str, float]) -> List[Dict[str, Any]]:
        """Prepare stocks data for visualization."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        
        stocks_data = []
        for stock in stocks:
            stock_info = metadata.get(stock, {})
            allocation = asset_breakdown.get(stock, 0)
            
            stocks_data.append({
                "symbol": stock,
                "name": stock_info.get("name", stock),
                "allocation": allocation,
                "category": stock_info.get("category", "equity"),
                "risk_score": stock_info.get("risk_score", 15),
                "expected_return": stock_info.get("expected_return", 8)
            })
        
        return stocks_data
    
    def _prepare_stocks_data_for_db(self, stocks: List[str], asset_breakdown: Dict[str, float]) -> List[Dict[str, Any]]:
        """Prepare stocks data for database storage."""
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        
        stocks_data = []
        for stock in stocks:
            stock_info = metadata.get(stock, {})
            allocation = asset_breakdown.get(stock, 0)
            
            stocks_data.append({
                "symbol": stock,
                "name": stock_info.get("name", stock),
                "allocation": allocation,
                "explanation": f"Selected based on AI analysis for your risk profile and goals"
            })
        
        return stocks_data
    
    async def cleanup_old_files(self, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old visualization files.
        
        Args:
            max_age_days: Maximum age in days for files to keep
            
        Returns:
            Cleanup summary
        """
        try:
            deleted_count = await self.visualization_service.cleanup_old_files(max_age_days)
            
            return {
                "success": True,
                "files_deleted": deleted_count,
                "max_age_days": max_age_days
            }
            
        except Exception as e:
            logger.error(f"File cleanup failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "files_deleted": 0
            }


# Factory function for service initialization
def create_portfolio_simulator_service(config_override: Optional[Dict[str, Any]] = None) -> PortfolioSimulatorService:
    """
    Factory function to create and initialize the portfolio simulator service.
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Initialized PortfolioSimulatorService instance
        
    Raises:
        PortfolioSimulatorError: If initialization fails
    """
    try:
        # Initialize configuration
        config = initialize_config()
        
        # Apply any overrides
        if config_override:
            # This would require implementing config override logic
            pass
        
        # Create and return service
        service = PortfolioSimulatorService()
        
        logger.info("Portfolio Simulator Service factory initialization complete")
        return service
        
    except Exception as e:
        logger.error(f"Service initialization failed: {str(e)}")
        raise PortfolioSimulatorError(
            f"Failed to initialize portfolio simulator service: {str(e)}",
            error_code="SERVICE_INITIALIZATION_FAILED"
        )


# Convenience functions for common operations
async def simulate_portfolio_workflow(
    simulation_input: Dict[str, Any], 
    db: Session
) -> Dict[str, Any]:
    """
    Convenience function to run a complete portfolio simulation.
    
    Args:
        simulation_input: Raw simulation input
        db: Database session
        
    Returns:
        Complete simulation results
    """
    service = create_portfolio_simulator_service()
    return await service.simulate_portfolio(simulation_input, db)


async def get_simulation_charts(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Convenience function to get chart data for a simulation.
    
    Args:
        simulation_id: ID of the simulation
        db: Database session
        
    Returns:
        Chart data for the simulation
    """
    service = create_portfolio_simulator_service()
    return await service.get_simulation_chart_data(simulation_id, db)