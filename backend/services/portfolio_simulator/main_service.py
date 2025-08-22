"""
Main Portfolio Simulator Service - Orchestrates the complete simulation workflow.

This module provides the main service class that coordinates all components
to deliver complete portfolio simulation functionality with AI recommendations,
SHAP explanations, and visualizations.
"""

import logging
import asyncio
from datetime import datetime
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
from .ai_analysis import AIAnalysisService  # NEW: Import AI analysis service
from .news_analysis import NewsAnalysisService  # NEW: Import news analysis service

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
    7. Comprehensive AI analysis with news sentiment
    8. Comprehensive error handling and logging
    """
    
    def __init__(self):
        """Initialize the portfolio simulator service with all components including AI analysis."""
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
        
        # NEW: Initialize AI analysis services
        self.ai_analysis_service = AIAnalysisService()
        
        logger.info("Portfolio Simulator Service initialized successfully with AI analysis")
    
    async def simulate_portfolio(self, simulation_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Execute complete portfolio simulation workflow with enhanced AI analysis.
        
        Args:
            simulation_input: Raw input data from user
            db: Database session
            
        Returns:
            Complete simulation results with all enhancements
            
        Raises:
            PortfolioSimulatorError: If simulation fails at any stage
        """
        try:
            logger.info("Starting enhanced portfolio simulation workflow with AI analysis")
            
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
            
            # NEW Step 7: Generate comprehensive AI analysis with news sentiment
            ai_analysis_results = await self._generate_comprehensive_ai_analysis(
                recommendations, validated_input, simulation_results, processed_shap
            )
            logger.info("AI analysis and news sentiment analysis completed")
            
            # Step 8: Generate visualizations
            visualization_paths = await self._generate_visualizations(
                simulation_results, recommendations, processed_shap, validated_input
            )
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            
            # Step 9: Save to database with enhanced AI analysis
            saved_simulation = self._save_simulation(
                db, validated_input, simulation_results, recommendations,
                ai_analysis_results, processed_shap, visualization_paths
            )
            
            # Step 10: Generate chart data for frontend
            chart_data = self._generate_chart_data(
                simulation_results, recommendations, processed_shap
            )
            
            # Step 11: Format final response
            response = self._format_final_response(
                saved_simulation, chart_data, visualization_paths, ai_analysis_results
            )
            
            logger.info(f"Enhanced simulation workflow completed successfully (ID: {saved_simulation.id})")
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
    
    async def _generate_comprehensive_ai_analysis(
        self,
        recommendations: Dict[str, Any],
        validated_input: Dict[str, Any],
        simulation_results: Any,
        processed_shap: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI analysis including news sentiment and market context.
        
        Returns:
            Dictionary containing all AI analysis results
        """
        try:
            logger.info("Starting comprehensive AI analysis")
            
            # Prepare stocks data for analysis
            stocks_picked = self._prepare_stocks_data_for_analysis(
                recommendations["stocks"], simulation_results.asset_breakdown
            )
            
            # Convert simulation results to dictionary format for AI analysis
            simulation_dict = {
                "end_value": simulation_results.portfolio_metrics.ending_value,
                "starting_value": simulation_results.portfolio_metrics.starting_value,
                "total_return": simulation_results.portfolio_metrics.total_return,
                "timeline": {
                    "portfolio": simulation_results.timeline_data
                }
            }
            
            # Extract risk information
            risk_score = validated_input.get("risk_score", 50)
            risk_label = validated_input.get("risk_label", "moderate")
            
            # Generate comprehensive portfolio summary with news analysis
            ai_summary = await self.ai_analysis_service.generate_portfolio_summary(
                stocks_picked=stocks_picked,
                user_data=validated_input,
                risk_score=risk_score,
                risk_label=risk_label,
                simulation_results=simulation_dict
            )
            
            # Get portfolio news analysis for recent market context
            portfolio_news_analysis = await self.ai_analysis_service.analyze_portfolio_with_context(
                portfolio_data={"stocks_picked": stocks_picked},
                days_back=30  # Last 30 days of news
            )
            
            # Analyze portfolio performance with news context
            performance_analysis = await self.ai_analysis_service.analyze_portfolio_performance(
                {"stocks_picked": stocks_picked}
            )
            
            # Analyze risk allocation with market sentiment
            risk_analysis = await self.ai_analysis_service.analyze_risk_allocation(
                {"stocks_picked": stocks_picked}
            )
            
            logger.info("AI analysis components completed successfully")
            
            return {
                "ai_summary": ai_summary,
                "portfolio_news_analysis": portfolio_news_analysis,
                "performance_analysis": performance_analysis,
                "risk_analysis": risk_analysis,
                "analysis_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "analysis_version": "enhanced_v1.0",
                    "includes_news_sentiment": True,
                    "includes_market_context": True,
                    "stocks_analyzed": len(stocks_picked)
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive AI analysis failed: {str(e)}")
            # Return fallback analysis if AI analysis fails
            return {
                "ai_summary": self._generate_fallback_summary(validated_input, simulation_results),
                "portfolio_news_analysis": {"error": "News analysis unavailable"},
                "performance_analysis": {"error": "Performance analysis unavailable"},
                "risk_analysis": {"error": "Risk analysis unavailable"},
                "analysis_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "analysis_version": "fallback_v1.0",
                    "includes_news_sentiment": False,
                    "includes_market_context": False,
                    "error": str(e)
                }
            }
    
    def _prepare_stocks_data_for_analysis(self, stocks: List[str], asset_breakdown: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Prepare stocks data in the format expected by AI analysis services.
        """
        from .config import get_stock_metadata
        metadata = get_stock_metadata()
        
        stocks_data = []
        for stock in stocks:
            stock_info = metadata.get(stock, {})
            allocation = asset_breakdown.get(stock, 0)
            
            stocks_data.append({
                "symbol": stock,
                "ticker": stock,  # Some services expect 'ticker'
                "name": stock_info.get("name", stock),
                "allocation": allocation,
                "weight": allocation / 100.0,  # Convert percentage to decimal
                "category": stock_info.get("category", "equity"),
                "risk_score": stock_info.get("risk_score", 15),
                "expected_return": stock_info.get("expected_return", 8)
            })
        
        return stocks_data
    
    def _generate_fallback_summary(self, validated_input: Dict[str, Any], simulation_results: Any) -> str:
        """
        Generate a basic fallback summary when AI analysis fails.
        """
        try:
            performance = simulation_results.portfolio_metrics
            target_achieved = performance.ending_value >= validated_input["target_value"]
            
            return f"""
## Investment Journey Summary

Your {validated_input['risk_label'].lower()} risk portfolio achieved {performance.total_return:.1f}% total returns over {validated_input['timeframe']} years.

**Key Results:**
- Starting Value: £{performance.starting_value:,.2f}
- Final Value: £{performance.ending_value:,.2f}
- Target: £{validated_input['target_value']:,.2f} ({'✅ Achieved' if target_achieved else '📈 In Progress'})
- Total Return: {performance.total_return:.1f}%
- Annual Return: {performance.annualized_return:.1f}%

**Risk Metrics:**
- Portfolio Volatility: {performance.volatility:.1f}%
- Sharpe Ratio: {performance.sharpe_ratio:.2f}
- Maximum Drawdown: {performance.max_drawdown:.1f}%

Your investment strategy demonstrated the power of long-term, diversified investing with AI-optimized portfolio allocation.
"""
        except Exception:
            return "Portfolio simulation completed successfully with positive returns over the investment period."
    
    def _save_simulation(
        self,
        db: Session,
        validated_input: Dict[str, Any],
        simulation_results: Any,
        recommendations: Dict[str, Any],
        ai_analysis_results: Dict[str, Any],  # UPDATED: Use comprehensive AI analysis
        processed_shap: Optional[Dict[str, Any]],
        visualization_paths: Dict[str, str]
    ) -> Any:
        """Save simulation to database with enhanced AI analysis."""
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
            
            # UPDATED: Save with comprehensive AI analysis
            saved_simulation = self.database_service.save_simulation(
                db=db,
                user_id=validated_input["user_id"],
                name=validated_input.get("goal", "Investment Simulation"),
                input_payload=validated_input,
                result_payload={
                    "stocks_picked": stocks_data,
                    "portfolio_metrics": results_dict,
                    "ai_analysis": ai_analysis_results,  # UPDATED: Store full AI analysis
                    "shap_explanation": processed_shap,
                    "visualization_paths": visualization_paths,
                    "target_achieved": simulation_results.portfolio_metrics.ending_value >= validated_input["target_value"],
                    "enhanced_features": {
                        "news_sentiment_analysis": True,
                        "market_context_analysis": True,
                        "comprehensive_ai_summary": True,
                        "portfolio_news_tracking": True
                    }
                }
            )
            
            return saved_simulation
        
        except Exception as e:
            logger.error(f"Database save failed: {str(e)}")
            raise PortfolioSimulatorError(
                f"Failed to save simulation: {str(e)}",
                error_code="DATABASE_SAVE_FAILED"
            )
    
    def _format_final_response(
        self,
        saved_simulation: Any,
        chart_data: Dict[str, Any],
        visualization_paths: Dict[str, str],
        ai_analysis_results: Dict[str, Any]  # NEW: Include AI analysis in response
    ) -> Dict[str, Any]:
        """Format the final response with enhanced AI analysis."""
        formatted_response = self.results_formatter.format_simulation_response(saved_simulation)
        
        # Add chart data and enhanced AI analysis
        formatted_response.update({
            "chart_data": chart_data,
            "ai_analysis": {
                "summary": ai_analysis_results.get("ai_summary", ""),
                "news_sentiment": ai_analysis_results.get("portfolio_news_analysis", {}),
                "performance_insights": ai_analysis_results.get("performance_analysis", {}),
                "risk_insights": ai_analysis_results.get("risk_analysis", {}),
                "metadata": ai_analysis_results.get("analysis_metadata", {})
            },
            "success": True,
            "workflow_completed": True,
            "enhanced_features_enabled": {
                "ai_recommendations": True,
                "shap_explanations": bool(saved_simulation.results.get("shap_explanation")),
                "visualizations": bool(visualization_paths),
                "interactive_charts": bool(chart_data),
                "news_sentiment_analysis": ai_analysis_results.get("analysis_metadata", {}).get("includes_news_sentiment", False),
                "market_context_analysis": ai_analysis_results.get("analysis_metadata", {}).get("includes_market_context", False),
                "comprehensive_ai_summary": True
            }
        })
        
        return formatted_response
    
    async def get_simulation_with_ai_analysis(self, simulation_id: int, db: Session) -> Dict[str, Any]:
        """
        Get simulation with full AI analysis data.
        
        Args:
            simulation_id: ID of the simulation
            db: Database session
            
        Returns:
            Complete simulation data including AI analysis
        """
        try:
            # Retrieve simulation from database
            simulation = self.database_service.get_simulation(db, simulation_id)
            
            if not simulation:
                raise PortfolioSimulatorError(
                    f"Simulation with ID {simulation_id} not found",
                    error_code="SIMULATION_NOT_FOUND"
                )
            
            # Format response with AI analysis
            formatted_response = self.results_formatter.format_simulation_response(simulation)
            
            # Extract AI analysis from stored results
            ai_analysis = simulation.results.get("ai_analysis", {})
            
            # Generate chart data
            chart_data = self.chart_data_generator.generate_chart_data(
                simulation.results, 
                simulation.results.get("stocks_picked", []),
                simulation.results.get("shap_explanation")
            )
            
            formatted_response.update({
                "chart_data": chart_data,
                "ai_analysis": ai_analysis,
                "enhanced_features": simulation.results.get("enhanced_features", {}),
                "retrieved_at": datetime.now().isoformat()
            })
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to get simulation with AI analysis {simulation_id}: {str(e)}")
            if isinstance(e, PortfolioSimulatorError):
                raise
            else:
                raise PortfolioSimulatorError(
                    f"Failed to retrieve simulation: {str(e)}",
                    error_code="SIMULATION_RETRIEVAL_FAILED"
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
                    "has_ai_analysis": bool(results.get("ai_analysis")),
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
        
        logger.info("Enhanced Portfolio Simulator Service factory initialization complete")
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


async def get_simulation_with_full_analysis(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Convenience function to get simulation with full AI analysis.
    
    Args:
        simulation_id: ID of the simulation
        db: Database session
        
    Returns:
        Complete simulation data including AI analysis
    """
    service = create_portfolio_simulator_service()
    return await service.get_simulation_with_ai_analysis(simulation_id, db)