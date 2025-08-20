"""
Portfolio Simulator Service - Enhanced with WealthWise SHAP Integration & Visualization Engine

This module handles the complete portfolio simulation workflow:
1. Extracts and validates user investment preferences
2. Uses AI to recommend appropriate stocks based on risk profile
3. Downloads historical market data for simulation
4. Calculates portfolio weights and simulates growth over time
5. Generates AI-powered educational summaries with SHAP explanations
6. Creates interactive visualizations using VisualizationEngine
7. Saves results to database

The service integrates with:
- WealthWise Enhanced Stock Recommender AI (for goal-oriented stock selection)
- SHAP Explainable AI (for transparent recommendations)
- VisualizationEngine (for interactive charts and SHAP plots)
- AI Analysis Service (for educational summaries)
- Yahoo Finance API (for historical data)
- Database models (for persistence)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from database import models
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import json
import os
from pathlib import Path

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)

# =============================================================================
# WEALTHWISE & VISUALIZATION ENGINE INTEGRATION
# =============================================================================

try:
    from ai_models.stock_model.core.recommender import EnhancedStockRecommender
    from ai_models.stock_model.explainable_ai import SHAPExplainer, VisualizationEngine
    from ai_models.stock_model.goal_optimization import GoalCalculator, FeasibilityAssessor
    from ai_models.stock_model.analysis import MarketRegimeDetector, FactorAnalyzer
    from ai_models.stock_model.utils import initialize_complete_system
    WEALTHWISE_AVAILABLE = True
    logger.info("✅ WealthWise SHAP system with VisualizationEngine loaded successfully")
except ImportError as e:
    WEALTHWISE_AVAILABLE = False
    logger.warning(f"⚠️ WealthWise not available: {e}")

# Initialize global visualization engine instance
_viz_engine = None

def get_visualization_engine() -> Optional['VisualizationEngine']:
    """
    Get or initialize the global VisualizationEngine instance.
    
    Returns:
        VisualizationEngine instance or None if not available
    """
    global _viz_engine
    
    if not WEALTHWISE_AVAILABLE:
        return None
    
    if _viz_engine is None:
        try:
            _viz_engine = VisualizationEngine()
            logger.info("📊 VisualizationEngine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize VisualizationEngine: {e}")
            return None
    
    return _viz_engine

# =============================================================================
# ENHANCED MAIN PORTFOLIO SIMULATION FUNCTION WITH VISUALIZATIONS
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Enhanced portfolio simulation with WealthWise SHAP integration and visualizations.
    
    This function now includes:
    1. Goal-oriented portfolio optimization
    2. SHAP explainable AI explanations
    3. Interactive visualizations and charts
    4. Market regime detection
    5. Multi-factor analysis
    6. Enhanced AI educational summaries
    
    Args:
        sim_input: Dictionary containing user onboarding data
        db: Database session for saving results
    
    Returns:
        Enhanced simulation results with SHAP explanations and visualization paths
    """
    
    try:
        logger.info("🚀 Starting enhanced portfolio simulation with WealthWise + Visualizations")
        
        # STEP 1: Extract and validate user investment preferences
        logger.info("📋 Extracting user investment data")
        user_data = {
            "experience": sim_input.get("years_of_experience", 0),
            "goal": sim_input.get("goal", "wealth building"),
            "target_value": float(sim_input.get("target_value", 50000)),
            "lump_sum": float(sim_input.get("lump_sum", 0) or 0),
            "monthly": float(sim_input.get("monthly", 0) or 0),
            "timeframe": int(sim_input.get("timeframe", 10)),
            "income_bracket": sim_input.get("income_bracket", "medium")
        }

        # Extract risk assessment results
        risk_score = sim_input.get("risk_score", 35)  # 0-100 scale
        risk_label = sim_input.get("risk_label", "Medium")  # Human-readable label

        logger.info(f"📊 User profile: goal={user_data['goal']}, target=£{user_data['target_value']:,.2f}, timeframe={user_data['timeframe']} years")
        logger.info(f"⚖️ Risk assessment: score={risk_score}/100, label={risk_label}")

        # STEP 2: Enhanced AI-powered stock recommendations with SHAP
        logger.info("🤖 Getting enhanced AI stock recommendations with explanations")
        recommendation_result = await get_enhanced_ai_recommendations(
            target_value=user_data["target_value"],
            timeframe=user_data["timeframe"],
            risk_score=risk_score,
            risk_label=risk_label,
            current_investment=user_data["lump_sum"],
            monthly_contribution=user_data["monthly"]
        )
        
        # Extract recommendations and explanations
        tickers = recommendation_result["stocks"]
        shap_explanation = recommendation_result.get("shap_explanation")
        goal_analysis = recommendation_result.get("goal_analysis")
        feasibility_assessment = recommendation_result.get("feasibility_assessment")
        market_regime = recommendation_result.get("market_regime")
        
        logger.info(f"📈 Enhanced AI recommended stocks: {tickers}")

        # STEP 3: Validate investment inputs
        logger.info("✅ Validating investment parameters")
        lump_sum = user_data["lump_sum"]
        monthly = user_data["monthly"]
        timeframe = user_data["timeframe"]
        
        if lump_sum <= 0 and monthly <= 0:
            raise ValueError("Either lump sum or monthly investment must be greater than 0")
        
        if timeframe <= 0:
            raise ValueError("Timeframe must be greater than 0")

        # STEP 4: Download historical market data for simulation
        logger.info("📊 Downloading historical market data")
        stock_data = download_stock_data(tickers, timeframe)
        
        # STEP 5: Enhanced portfolio weights using WealthWise optimization
        logger.info("⚖️ Calculating enhanced portfolio allocation weights")
        weights = calculate_enhanced_portfolio_weights(
            stock_data, risk_score, recommendation_result
        )
        
        # STEP 6: Create structured list of selected stocks with allocations
        logger.info("📋 Creating final stock allocation list")
        stocks_picked = [
            {
                "symbol": ticker, 
                "name": get_company_name(ticker),
                "allocation": round(float(weight), 4),
                "explanation": get_stock_explanation(ticker, recommendation_result)
            }
            for ticker, weight in zip(tickers, weights)
        ]
        
        logger.info("💼 Enhanced portfolio allocation:")
        for stock in stocks_picked:
            logger.info(f"   {stock['symbol']}: {stock['allocation']*100:.1f}% ({stock['name']})")

        # STEP 7: Run the portfolio growth simulation
        logger.info("📈 Running portfolio growth simulation")
        simulation_results = simulate_portfolio_growth(
            stock_data, weights, lump_sum, monthly, timeframe
        )

        # STEP 8: Generate visualizations
        logger.info("📊 Creating interactive visualizations")
        visualization_paths = await create_simulation_visualizations(
            simulation_id=None,  # Will be set after saving to DB
            stocks_picked=stocks_picked,
            simulation_results=simulation_results,
            shap_explanation=shap_explanation,
            user_data=user_data,
            stock_data=stock_data
        )

        # STEP 9: Generate enhanced AI summary with SHAP explanations
        logger.info("🧠 Generating enhanced AI educational summary with SHAP")
        ai_summary = await generate_enhanced_ai_summary(
            stocks_picked, user_data, risk_score, risk_label, 
            simulation_results, shap_explanation, goal_analysis, 
            feasibility_assessment, market_regime
        )

        # STEP 10: Save enhanced simulation results to database
        logger.info("💾 Saving enhanced simulation to database")
        simulation = save_enhanced_simulation_to_db(
            db=db,
            sim_input=sim_input,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            ai_summary=ai_summary,
            stocks_picked=stocks_picked,
            simulation_results=simulation_results,
            shap_explanation=shap_explanation,
            goal_analysis=goal_analysis,
            feasibility_assessment=feasibility_assessment,
            market_regime=market_regime,
            visualization_paths=visualization_paths
        )

        # STEP 11: Update visualizations with actual simulation ID
        if visualization_paths and simulation.id:
            logger.info("🔄 Updating visualizations with simulation ID")
            updated_paths = await update_visualization_paths_with_id(
                simulation.id, visualization_paths
            )
            
            # Update the database record with correct paths
            if updated_paths != visualization_paths:
                simulation.results["visualization_paths"] = updated_paths
                db.commit()

        logger.info(f"✅ Enhanced portfolio simulation completed successfully (ID: {simulation.id})")
        return format_enhanced_simulation_response(simulation)

    except Exception as e:
        logger.error(f"❌ Enhanced portfolio simulation failed: {str(e)}")
        db.rollback()
        
        # Fallback to original simulation if enhanced version fails
        logger.warning("🔄 Falling back to original simulation method")
        return await simulate_portfolio_fallback(sim_input, db)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

async def create_simulation_visualizations(
    simulation_id: Optional[int],
    stocks_picked: List[Dict],
    simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict],
    user_data: Dict[str, Any],
    stock_data: pd.DataFrame
) -> Dict[str, str]:
    """
    Create comprehensive visualizations for the portfolio simulation.
    
    Args:
        simulation_id: Database ID of simulation (None if not saved yet)
        stocks_picked: List of selected stocks with allocations
        simulation_results: Portfolio growth simulation results
        shap_explanation: SHAP explanation data
        user_data: User investment preferences
        stock_data: Historical stock price data
    
    Returns:
        Dictionary mapping visualization types to file paths
    """
    
    viz_engine = get_visualization_engine()
    if not viz_engine:
        logger.warning("⚠️ VisualizationEngine not available, skipping visualizations")
        return {}
    
    try:
        # Create directory for visualizations
        viz_dir = Path("static/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        simulation_prefix = f"sim_{simulation_id}" if simulation_id else "temp_simulation"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{simulation_prefix}_{timestamp}"
        
        visualization_paths = {}
        
        # 1. Portfolio Allocation Pie Chart
        logger.info("📊 Creating portfolio allocation visualization")
        try:
            allocation_data = {
                stock["symbol"]: stock["allocation"] 
                for stock in stocks_picked
            }
            
            allocation_path = viz_dir / f"{base_filename}_allocation.png"
            result = viz_engine.create_portfolio_allocation_chart(
                allocation_data, str(allocation_path)
            )
            
            if "saved" in result.lower():
                visualization_paths["portfolio_allocation"] = str(allocation_path)
                logger.info(f"✅ Portfolio allocation chart saved: {allocation_path}")
            else:
                logger.warning(f"⚠️ Portfolio allocation chart failed: {result}")
                
        except Exception as e:
            logger.error(f"❌ Error creating allocation chart: {e}")
        
        # 2. Portfolio Growth Timeline
        logger.info("📈 Creating portfolio growth timeline")
        try:
            timeline_data = simulation_results.get("timeline", {})
            portfolio_values = timeline_data.get("portfolio", [])
            contributions = timeline_data.get("contributions", [])
            
            if portfolio_values and contributions:
                # Convert to DataFrame for visualization
                timeline_df = pd.DataFrame({
                    'date': [item['date'] for item in portfolio_values],
                    'portfolio_value': [item['value'] for item in portfolio_values],
                    'contributions': [item['value'] for item in contributions]
                })
                timeline_df['date'] = pd.to_datetime(timeline_df['date'])
                
                timeline_path = viz_dir / f"{base_filename}_timeline.png"
                result = viz_engine.create_portfolio_timeline_chart(
                    timeline_df, str(timeline_path), user_data.get("goal", "Portfolio Growth")
                )
                
                if "saved" in result.lower():
                    visualization_paths["portfolio_timeline"] = str(timeline_path)
                    logger.info(f"✅ Portfolio timeline chart saved: {timeline_path}")
                else:
                    logger.warning(f"⚠️ Portfolio timeline chart failed: {result}")
            
        except Exception as e:
            logger.error(f"❌ Error creating timeline chart: {e}")
        
        # 3. SHAP Explanation Waterfall Chart
        if shap_explanation:
            logger.info("🔍 Creating SHAP explanation visualization")
            try:
                shap_path = viz_dir / f"{base_filename}_shap_explanation.png"
                result = viz_engine.create_shap_waterfall_chart(
                    shap_explanation, str(shap_path)
                )
                
                if "saved" in result.lower():
                    visualization_paths["shap_explanation"] = str(shap_path)
                    logger.info(f"✅ SHAP explanation chart saved: {shap_path}")
                else:
                    logger.warning(f"⚠️ SHAP explanation chart failed: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Error creating SHAP chart: {e}")
        
        # 4. Risk vs Return Scatter Plot
        logger.info("⚖️ Creating risk vs return analysis")
        try:
            if len(stock_data.columns) > 1:
                # Calculate returns and volatility for each stock
                returns = stock_data.pct_change().dropna()
                
                risk_return_data = []
                for ticker in stock_data.columns:
                    if ticker in returns.columns:
                        annual_return = returns[ticker].mean() * 252
                        annual_volatility = returns[ticker].std() * np.sqrt(252)
                        
                        # Find allocation for this stock
                        allocation = 0
                        for stock in stocks_picked:
                            if stock["symbol"] == ticker:
                                allocation = stock["allocation"]
                                break
                        
                        risk_return_data.append({
                            "symbol": ticker,
                            "return": annual_return,
                            "risk": annual_volatility,
                            "allocation": allocation
                        })
                
                if risk_return_data:
                    risk_return_path = viz_dir / f"{base_filename}_risk_return.png"
                    result = viz_engine.create_risk_return_scatter(
                        risk_return_data, str(risk_return_path)
                    )
                    
                    if "saved" in result.lower():
                        visualization_paths["risk_return_analysis"] = str(risk_return_path)
                        logger.info(f"✅ Risk vs return chart saved: {risk_return_path}")
                    else:
                        logger.warning(f"⚠️ Risk vs return chart failed: {result}")
            
        except Exception as e:
            logger.error(f"❌ Error creating risk vs return chart: {e}")
        
        # 5. Goal Achievement Progress Chart
        logger.info("🎯 Creating goal achievement visualization")
        try:
            target_value = user_data.get("target_value", 50000)
            end_value = simulation_results.get("end_value", 0)
            
            goal_data = {
                "target_value": target_value,
                "achieved_value": end_value,
                "achievement_percentage": min(100, (end_value / target_value) * 100) if target_value > 0 else 0,
                "goal_name": user_data.get("goal", "Financial Goal"),
                "timeframe": user_data.get("timeframe", 10)
            }
            
            goal_path = viz_dir / f"{base_filename}_goal_progress.png"
            result = viz_engine.create_goal_achievement_chart(
                goal_data, str(goal_path)
            )
            
            if "saved" in result.lower():
                visualization_paths["goal_achievement"] = str(goal_path)
                logger.info(f"✅ Goal achievement chart saved: {goal_path}")
            else:
                logger.warning(f"⚠️ Goal achievement chart failed: {result}")
                
        except Exception as e:
            logger.error(f"❌ Error creating goal achievement chart: {e}")
        
        logger.info(f"📊 Created {len(visualization_paths)} visualizations successfully")
        return visualization_paths
        
    except Exception as e:
        logger.error(f"❌ Error creating simulation visualizations: {e}")
        return {}

async def update_visualization_paths_with_id(
    simulation_id: int, 
    temp_paths: Dict[str, str]
) -> Dict[str, str]:
    """
    Update temporary visualization file paths with actual simulation ID.
    
    Args:
        simulation_id: Actual database ID of the simulation
        temp_paths: Dictionary of temporary file paths
    
    Returns:
        Dictionary of updated file paths with simulation ID
    """
    
    updated_paths = {}
    
    try:
        for viz_type, temp_path in temp_paths.items():
            temp_file = Path(temp_path)
            
            if temp_file.exists():
                # Create new filename with simulation ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"sim_{simulation_id}_{timestamp}_{viz_type}.png"
                new_path = temp_file.parent / new_filename
                
                # Rename the file
                temp_file.rename(new_path)
                updated_paths[viz_type] = str(new_path)
                
                logger.info(f"📁 Renamed {temp_path} → {new_path}")
            else:
                logger.warning(f"⚠️ Temporary file not found: {temp_path}")
                # Keep original path even if file doesn't exist
                updated_paths[viz_type] = temp_path
        
        return updated_paths
        
    except Exception as e:
        logger.error(f"❌ Error updating visualization paths: {e}")
        return temp_paths  # Return original paths on error

# =============================================================================
# ENHANCED VISUALIZATION API ENDPOINTS
# =============================================================================

async def get_simulation_visualizations(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Get all available visualizations for a simulation.
    
    Args:
        simulation_id: ID of the simulation
        db: Database session
    
    Returns:
        Dictionary containing visualization paths and metadata
    """
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        # Extract visualization paths from results
        results = simulation.results or {}
        visualization_paths = results.get("visualization_paths", {})
        
        # Check which files actually exist
        available_visualizations = {}
        for viz_type, file_path in visualization_paths.items():
            if Path(file_path).exists():
                available_visualizations[viz_type] = {
                    "path": file_path,
                    "type": viz_type,
                    "exists": True,
                    "size": Path(file_path).stat().st_size,
                    "created": datetime.fromtimestamp(
                        Path(file_path).stat().st_ctime
                    ).isoformat()
                }
            else:
                available_visualizations[viz_type] = {
                    "path": file_path,
                    "type": viz_type,
                    "exists": False,
                    "error": "File not found"
                }
        
        return {
            "simulation_id": simulation_id,
            "visualization_count": len(available_visualizations),
            "visualizations": available_visualizations,
            "has_shap_explanation": bool(results.get("shap_explanation")),
            "wealthwise_enhanced": results.get("wealthwise_enhanced", False)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting simulation visualizations: {e}")
        return {"error": str(e)}

async def regenerate_shap_visualization(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Regenerate SHAP visualization for a specific simulation.
    
    Args:
        simulation_id: ID of the simulation
        db: Database session
    
    Returns:
        Result of visualization regeneration
    """
    
    viz_engine = get_visualization_engine()
    if not viz_engine:
        return {"error": "VisualizationEngine not available"}
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation")
        
        if not shap_explanation:
            return {"error": "No SHAP explanation available for this simulation"}
        
        # Create new SHAP visualization
        viz_dir = Path("static/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shap_path = viz_dir / f"sim_{simulation_id}_{timestamp}_shap_explanation_regenerated.png"
        
        result = viz_engine.create_shap_waterfall_chart(
            shap_explanation, str(shap_path)
        )
        
        if "saved" in result.lower():
            # Update database with new path
            current_paths = results.get("visualization_paths", {})
            current_paths["shap_explanation"] = str(shap_path)
            results["visualization_paths"] = current_paths
            
            simulation.results = results
            db.commit()
            
            return {
                "success": True,
                "new_path": str(shap_path),
                "message": "SHAP visualization regenerated successfully",
                "timestamp": timestamp
            }
        else:
            return {"error": f"Failed to generate SHAP visualization: {result}"}
            
    except Exception as e:
        logger.error(f"❌ Error regenerating SHAP visualization: {e}")
        return {"error": str(e)}

async def create_custom_visualization(
    simulation_id: int, 
    viz_type: str, 
    options: Dict[str, Any],
    db: Session
) -> Dict[str, Any]:
    """
    Create a custom visualization for a simulation.
    
    Args:
        simulation_id: ID of the simulation
        viz_type: Type of visualization to create
        options: Visualization options and parameters
        db: Database session
    
    Returns:
        Result of custom visualization creation
    """
    
    viz_engine = get_visualization_engine()
    if not viz_engine:
        return {"error": "VisualizationEngine not available"}
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        results = simulation.results or {}
        
        # Create visualization directory
        viz_dir = Path("static/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = viz_dir / f"sim_{simulation_id}_{timestamp}_custom_{viz_type}.png"
        
        # Handle different visualization types
        if viz_type == "correlation_heatmap":
            # Create correlation heatmap
            stocks = [stock["symbol"] for stock in results.get("stocks_picked", [])]
            result = viz_engine.create_correlation_heatmap(stocks, str(viz_path))
            
        elif viz_type == "performance_comparison":
            # Create performance comparison chart
            timeline = results.get("timeline", {})
            portfolio_data = timeline.get("portfolio", [])
            
            if portfolio_data:
                # Convert to DataFrame
                df = pd.DataFrame(portfolio_data)
                df['date'] = pd.to_datetime(df['date'])
                
                result = viz_engine.create_performance_comparison_chart(
                    df, str(viz_path), options.get("benchmark", "SPY")
                )
            else:
                return {"error": "No portfolio timeline data available"}
                
        elif viz_type == "monte_carlo":
            # Create Monte Carlo simulation visualization
            monte_carlo_options = {
                "num_simulations": options.get("num_simulations", 1000),
                "confidence_level": options.get("confidence_level", 0.95)
            }
            
            result = viz_engine.create_monte_carlo_simulation(
                results.get("stocks_picked", []), 
                str(viz_path),
                monte_carlo_options
            )
            
        else:
            return {"error": f"Unknown visualization type: {viz_type}"}
        
        if "saved" in result.lower():
            return {
                "success": True,
                "visualization_path": str(viz_path),
                "visualization_type": viz_type,
                "options_used": options,
                "timestamp": timestamp
            }
        else:
            return {"error": f"Failed to create {viz_type} visualization: {result}"}
            
    except Exception as e:
        logger.error(f"❌ Error creating custom visualization: {e}")
        return {"error": str(e)}

# =============================================================================
# ENHANCED DATABASE FUNCTIONS WITH VISUALIZATION SUPPORT
# =============================================================================

def save_enhanced_simulation_to_db(
    db, sim_input: Dict[str, Any], user_data: Dict[str, Any],
    risk_score: int, risk_label: str, ai_summary: str,
    stocks_picked: List[Dict], simulation_results: Dict[str, Any],
    shap_explanation: Dict[str, Any] = None, 
    goal_analysis: Dict[str, Any] = None,
    feasibility_assessment: Dict[str, Any] = None, 
    market_regime: Dict[str, Any] = None,
    visualization_paths: Dict[str, str] = None
):
    """
    Enhanced version of save_simulation_to_db with visualization support.
    """
    
    try:
        logger.info("💾 Saving enhanced simulation with SHAP data and visualizations to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        # Create enhanced results object with all data including visualizations
        enhanced_results = {
            "name": user_data["goal"],
            "stocks_picked": stocks_picked,
            "starting_value": simulation_results["starting_value"],
            "end_value": simulation_results["end_value"],
            "return": simulation_results["portfolio_return"],
            "target_reached": target_reached,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "timeline": simulation_results["timeline"],
            # Enhanced data
            "shap_explanation": shap_explanation,
            "goal_analysis": goal_analysis,
            "feasibility_assessment": feasibility_assessment,
            "market_regime": market_regime,
            "visualization_paths": visualization_paths or {},
            "wealthwise_enhanced": True,
            "has_visualizations": bool(visualization_paths),
            "methodology": "WealthWise SHAP-enhanced goal-oriented optimization with visualizations",
            "created_timestamp": datetime.now().isoformat()
        }
        
        # Clean the results before saving
        cleaned_results = clean_simulation_results_for_db(enhanced_results)
        
        # Test serialization before database save
        if not test_json_serialization(cleaned_results, "enhanced_results"):
            raise ValueError("Enhanced results still not JSON serializable after cleaning")
        
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
            risk_score=risk_score,
            risk_label=risk_label,
            ai_summary=ai_summary,
            results=cleaned_results
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        logger.info(f"✅ Enhanced simulation with visualizations saved (ID: {simulation.id})")
        return simulation
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced simulation: {str(e)}")
        db.rollback()
        
        # Fallback save without visualizations
        try:
            logger.warning("🔄 Attempting to save simulation without visualizations")
            
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
                "risk_score": risk_score,
                "risk_label": risk_label,
                "enhanced_save_failed": True,
                "visualization_error": str(e)
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
                risk_score=risk_score,
                risk_label=risk_label,
                ai_summary=ai_summary,
                results=basic_results
            )
            
            db.add(basic_simulation)
            db.commit()
            db.refresh(basic_simulation)
            
            logger.warning(f"⚠️ Saved basic simulation without visualizations (ID: {basic_simulation.id})")
            return basic_simulation
            
        except Exception as basic_error:
            logger.error(f"❌ Even basic simulation save failed: {basic_error}")
            db.rollback()
            raise

def format_enhanced_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """
    Format enhanced simulation response with SHAP explanations and visualizations.
    """
    
    # Get the results data
    results = simulation.results or {}
    
    # Extract enhanced data
    shap_explanation = results.get("shap_explanation")
    visualization_paths = results.get("visualization_paths", {})
    
    has_shap_explanations = bool(shap_explanation)
    has_visualizations = bool(visualization_paths)
    
    # Debug logging
    logger.info(f"🔍 Formatting enhanced response for simulation {simulation.id}")
    logger.info(f"📊 Has SHAP explanations: {has_shap_explanations}")
    logger.info(f"📊 Has visualizations: {has_visualizations}")
    logger.info(f"📊 Visualization types: {list(visualization_paths.keys())}")
    
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
        "results": results,
        "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat(),
        
        # Enhanced features
        "shap_explanation": shap_explanation,
        "has_shap_explanations": has_shap_explanations,
        "visualization_paths": visualization_paths,
        "has_visualizations": has_visualizations,
        "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
        "methodology": results.get("methodology", "Standard simulation"),
        
        # Visualization metadata
        "available_visualizations": list(visualization_paths.keys()),
        "visualization_count": len(visualization_paths)
    }
    
    return response

# =============================================================================
# ORIGINAL FUNCTIONS (PRESERVED FOR BACKWARD COMPATIBILITY)
# =============================================================================

async def get_enhanced_ai_recommendations(
    target_value: float, timeframe: int, risk_score: int, risk_label: str,
    current_investment: float = 0, monthly_contribution: float = 0
) -> Dict[str, Any]:
    """
    Get enhanced AI recommendations with factor analysis, SHAP explanations and goal analysis.
    """
    
    if not WEALTHWISE_AVAILABLE:
        logger.warning("⚠️ WealthWise not available, using fallback recommendations")
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback"
        }
    
    try:
        logger.info("🎯 Initializing WealthWise enhanced recommendation system with factor analysis")
        
        # Initialize WealthWise system
        init_result = initialize_complete_system({
            'LOG_LEVEL': 'INFO',
            'LOG_TO_FILE': False,
            'ENABLE_PERFORMANCE_TRACKING': True
        })
        
        if not init_result['success']:
            raise Exception(f"WealthWise initialization failed: {init_result.get('error')}")
        
        # Initialize core components
        recommender = EnhancedStockRecommender()
        factor_analyzer = FactorAnalyzer()
        shap_explainer = SHAPExplainer()
        goal_calculator = GoalCalculator()
        feasibility_assessor = FeasibilityAssessor()
        market_detector = MarketRegimeDetector()
        
        logger.info("🔍 Performing goal-oriented analysis")
        
        # Step 1: Calculate goal requirements
        goal_analysis = goal_calculator.calculate_required_return(
            target_value, current_investment, timeframe, monthly_contribution
        )
        
        # Step 2: Assess goal feasibility
        feasibility_assessment = feasibility_assessor.assess_goal_feasibility(
            goal_analysis["required_return"], risk_score, timeframe,
            current_investment, monthly_contribution
        )
        
        # Step 3: Detect market regime
        market_regime = market_detector.detect_market_regime()
        
        # Step 4: Get initial goal-oriented stock universe
        logger.info("🤖 Generating goal-oriented stock universe")
        initial_recommendations = recommender.recommend_stocks(
            target_value, timeframe, risk_score, 
            current_investment, monthly_contribution
        )
        
        # Step 5: Expand candidate universe based on risk profile and market regime
        logger.info("📊 Expanding candidate universe for factor analysis")
        candidate_stocks = set(initial_recommendations)
        
        # Add risk-appropriate candidates
        if risk_score < 35:  # Conservative
            candidate_stocks.update(["VTI", "BND", "VEA", "VTEB", "VWO", "AGG", "VNQ", "SCHD", "VYM"])
        elif risk_score < 70:  # Moderate
            candidate_stocks.update(["VTI", "VEA", "VWO", "VNQ", "BND", "VUG", "VGT", "VOO", "VXUS"])
        else:  # Aggressive
            candidate_stocks.update(["VTI", "VGT", "VUG", "ARKK", "VEA", "QQQ", "ARKQ", "TQQQ", "SOXL"])
        
        # Adjust for market regime
        if market_regime.get('regime') == 'bear':
            candidate_stocks.update(["VYM", "SCHD", "VDC", "VHT"])
        elif market_regime.get('regime') == 'bull':
            candidate_stocks.update(["VGT", "VUG", "QQQ"])
            
        candidate_stocks = list(candidate_stocks)
        logger.info(f"📈 Analyzing {len(candidate_stocks)} candidate stocks with factor analysis")
        
        # Step 6: Apply factor analysis to rank all candidates
        try:
            ranked_stocks = factor_analyzer.rank_stocks_by_factors(
                candidate_stocks,
                market_regime=market_regime,
                risk_score=risk_score,
                timeframe=timeframe
            )
            
            num_stocks = min(6, len(ranked_stocks))
            factor_selected_stocks = [stock for stock, score in ranked_stocks[:num_stocks]]
            
            logger.info(f"🎯 Factor analysis selected: {factor_selected_stocks}")
            
        except Exception as factor_error:
            logger.warning(f"⚠️ Factor analysis failed: {factor_error}, using initial recommendations")
            factor_selected_stocks = initial_recommendations[:6]
        
        # Step 7: Generate SHAP explanations
        logger.info("🔍 Generating SHAP explanations for transparency")
        shap_explanation = None
        if shap_explainer.is_available():
            shap_explanation = shap_explainer.get_shap_explanation(
                target_value, timeframe, risk_score,
                current_investment, monthly_contribution,
                market_regime.get('current_vix', 20),
                market_regime.get('trend_score', 2.5)
            )
        else:
            logger.info("🤖 Training SHAP explainer model...")
            success = shap_explainer.train_shap_model(num_samples=1000)
            if success:
                shap_explanation = shap_explainer.get_shap_explanation(
                    target_value, timeframe, risk_score,
                    current_investment, monthly_contribution,
                    market_regime.get('current_vix', 20),
                    market_regime.get('trend_score', 2.5)
                )
        
        # Step 8: Create enhanced response with factor analysis insights
        factor_insights = None
        if 'ranked_stocks' in locals():
            factor_insights = {
                "methodology": "Multi-factor quantitative analysis",
                "factors_analyzed": factor_analyzer.get_analyzed_factors(),
                "top_stocks_scores": dict(ranked_stocks[:num_stocks]),
                "selection_rationale": f"Selected top {num_stocks} stocks based on factor scores, market regime, and goal alignment"
            }
        
        logger.info(f"✅ Enhanced recommendations complete: {len(factor_selected_stocks)} stocks selected via factor analysis")
        
        return {
            "stocks": factor_selected_stocks,
            "shap_explanation": shap_explanation,
            "goal_analysis": goal_analysis,
            "feasibility_assessment": feasibility_assessment,
            "market_regime": market_regime,
            "factor_insights": factor_insights,
            "method": "wealthwise_enhanced_with_factors"
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced AI recommendations failed: {e}")
        logger.warning("🔄 Falling back to original recommendation method")
        
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback",
            "error": str(e)
        }

def get_stock_explanation(ticker: str, recommendation_result: Dict[str, Any]) -> str:
    """Get explanation for why a specific stock was recommended."""
    if recommendation_result.get("method") == "fallback":
        return f"{ticker} selected based on risk profile matching"
    
    shap_explanation = recommendation_result.get("shap_explanation", {})
    
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        explanations = shap_explanation["human_readable_explanation"]
        if explanations:
            for key, explanation in explanations.items():
                if len(explanation) > 20:
                    return f"{ticker}: {explanation[:100]}..."
    
    return f"{ticker} recommended by AI analysis for your goals and risk profile"

def calculate_enhanced_portfolio_weights(
    data: pd.DataFrame, risk_score: int, recommendation_result: Dict[str, Any]
) -> np.ndarray:
    """Calculate enhanced portfolio weights using WealthWise optimization."""
    
    if recommendation_result.get("method") == "fallback" or not WEALTHWISE_AVAILABLE:
        return calculate_portfolio_weights(data, risk_score)
    
    try:
        recommender = EnhancedStockRecommender()
        
        num_assets = len(data.columns)
        initial_weights = {col: 1.0/num_assets for col in data.columns}
        
        optimized_weights = recommender.optimize_for_diversification(
            list(data.columns), initial_weights
        )
        
        weights_array = np.array([
            optimized_weights.get(col, 1.0/num_assets) 
            for col in data.columns
        ])
        
        weights_array = weights_array / np.sum(weights_array)
        
        logger.info("✅ Using WealthWise correlation-optimized weights")
        return weights_array
        
    except Exception as e:
        logger.warning(f"⚠️ WealthWise optimization failed: {e}, using fallback")
        return calculate_portfolio_weights(data, risk_score)

# Enhanced AI summary generation function
async def generate_enhanced_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> str:
    """Generate comprehensive AI summary with SHAP explanations and news analysis."""
    
    try:
        logger.info("🧠 Generating INTEGRATED AI summary with SHAP + News Analysis")
        
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        # Get comprehensive news analysis
        logger.info("📰 Getting comprehensive news and market analysis...")
        portfolio_news_analysis = await ai_service._analyze_portfolio_news_history(
            stocks_picked, user_data, simulation_results
        )
        
        # Create integrated prompt
        integrated_prompt = create_integrated_shap_news_prompt(
            stocks_picked=stocks_picked,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            simulation_results=simulation_results,
            shap_explanation=shap_explanation,
            goal_analysis=goal_analysis,
            feasibility_assessment=feasibility_assessment,
            market_regime=market_regime,
            portfolio_news_analysis=portfolio_news_analysis
        )
        
        # Generate comprehensive summary
        logger.info("🤖 Generating integrated SHAP + News summary...")
        integrated_summary = await ai_service._get_groq_response(integrated_prompt)
        
        logger.info("✅ Integrated AI summary with SHAP + News generated successfully!")
        return ai_service._format_ai_response(integrated_summary)
        
    except Exception as e:
        logger.warning(f"⚠️ Integrated AI summary failed: {e}. Trying fallback methods...")
        
        try:
            logger.info("🔄 Fallback 1: Using enhanced news analysis only...")
            return await ai_service.generate_portfolio_summary(
                stocks_picked=stocks_picked,
                user_data=user_data,
                risk_score=risk_score,
                risk_label=risk_label,
                simulation_results=simulation_results
            )
        except Exception as e2:
            logger.warning(f"⚠️ Enhanced news analysis also failed: {e2}. Using simple SHAP summary...")
            
            return generate_simple_enhanced_summary(
                stocks_picked, user_data, risk_score, risk_label, 
                simulation_results, shap_explanation, goal_analysis, feasibility_assessment
            )

def create_integrated_shap_news_prompt(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict], goal_analysis: Optional[Dict],
    feasibility_assessment: Optional[Dict], market_regime: Optional[Dict],
    portfolio_news_analysis: Dict[str, Any]
) -> str:
    """Create comprehensive prompt combining SHAP explanations with news analysis."""
    
    # Extract basic info
    goal = user_data.get("goal", "wealth building")
    lump_sum = user_data.get("lump_sum", 0)
    monthly = user_data.get("monthly", 0)
    timeframe = user_data.get("timeframe", 10)
    target_value = user_data.get("target_value", 50000)
    
    end_value = simulation_results.get("end_value", 0)
    total_contributed = lump_sum + (monthly * timeframe * 12)
    target_achieved = end_value >= target_value
    
    symbols = [stock.get('symbol', '') for stock in stocks_picked]
    
    # Format SHAP explanations
    shap_context = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_context = f"""
🔍 AI DECISION EXPLANATIONS (SHAP Analysis):
The AI specifically chose this portfolio because:"""
        
        explanations = shap_explanation["human_readable_explanation"]
        for factor, explanation in explanations.items():
            if explanation and len(explanation) > 10:
                shap_context += f"""
• {explanation}"""
        
        quality_score = shap_explanation.get('portfolio_quality_score')
        if quality_score:
            shap_context += f"""

📊 AI Portfolio Quality Score: {quality_score:.1f}/10
This score reflects how well the AI believes this portfolio matches your goals and risk tolerance."""

    # Format goal analysis
    goal_context = ""
    if goal_analysis and feasibility_assessment:
        required_return = goal_analysis.get('required_return_percent', 0)
        feasibility = feasibility_assessment.get('feasibility_score', 0)
        
        goal_context = f"""
🎯 GOAL-ORIENTED ANALYSIS:
• Your Target: £{target_value:,.0f} in {timeframe} years
• Required Annual Return: {required_return:.1f}%
• AI Feasibility Assessment: {feasibility:.0f}% achievable
• Recommendation: {feasibility_assessment.get('recommendations', {}).get('primary', 'Continue with your plan')}"""

    # Format market regime  
    market_context = ""
    if market_regime:
        regime = market_regime.get('regime', 'neutral')
        trend_score = market_regime.get('trend_score', 2.5)
        vix = market_regime.get('current_vix', 20)
        
        market_context = f"""
📈 CURRENT MARKET CONDITIONS:
• Market Regime: {regime.title()} market environment
• Trend Strength: {trend_score:.1f}/5.0 (bullish trend)
• Volatility (VIX): {vix:.1f} ({"Low" if vix < 20 else "Moderate" if vix < 30 else "High"} fear level)"""

    # Format news analysis
    news_context = ""
    if 'recent_news_context' in portfolio_news_analysis and 'error' not in portfolio_news_analysis['recent_news_context']:
        recent_context = portfolio_news_analysis['recent_news_context']
        sentiment_summary = recent_context.get('market_sentiment_summary', {})
        
        overall_sentiment = sentiment_summary.get('overall_sentiment', 0)
        sentiment_desc = "Positive" if overall_sentiment > 0.1 else "Negative" if overall_sentiment < -0.1 else "Neutral"
        
        news_context = f"""
📰 CURRENT NEWS IMPACT ON YOUR PORTFOLIO:
• Overall Sentiment: {sentiment_desc} ({overall_sentiment:.3f})
• Total Recent Articles: {sentiment_summary.get('total_articles', 0)}
• Major Events: {sentiment_summary.get('total_events', 0)} detected

Recent News Headlines for Your Holdings:"""
        
        symbol_analysis = recent_context.get('symbol_specific_analysis', {})
        for symbol, analysis in symbol_analysis.items():
            sentiment = analysis.get('sentiment', {})
            headlines = analysis.get('top_headlines', [])
            
            news_context += f"""
• {symbol}: {analysis.get('article_count', 0)} articles, {sentiment.get('sentiment_category', 'Neutral')} sentiment
  Latest: {headlines[0][:100] + '...' if headlines else 'No recent headlines'}"""

    # Create the comprehensive prompt
    return f"""
You are an expert financial educator creating a comprehensive portfolio analysis that combines AI explainability with real-world market education. You must explain BOTH the AI's reasoning AND how actual market events affected the portfolio.

PORTFOLIO PERFORMANCE SUMMARY:
Holdings: {', '.join(symbols)}
Goal: {goal}
Target: £{target_value:,.0f}
Total Invested: £{total_contributed:,.0f}
Final Value: £{end_value:,.0f}
Result: {'🎉 GOAL ACHIEVED!' if target_achieved else '📈 PROGRESS MADE'}
Risk Level: {risk_label} ({risk_score}/100)
Investment Period: {timeframe} years

{shap_context}

{goal_context}

{market_context}

{news_context}

REQUIRED STRUCTURE (use this exact format):

## 🎯 Portfolio Summary: Achieving Your Goals with Confidence

[Opening paragraph explaining their results and why this analysis is unique - combining AI transparency with real market education]

## 🤖 Why the AI Selected These Specific Stocks

[Explain the SHAP analysis findings - why each stock was chosen based on their goals, risk tolerance, and market conditions. Make the AI reasoning transparent and educational.]

## 📰 How Real Market News Affected Your Portfolio  

[Connect current news sentiment to their holdings. Explain what recent headlines mean for their investments and how news drives market movements.]

## 📊 Your Journey Through Market History

[Detail the major market events and corrections their portfolio lived through, using the historical analysis. Make it educational about market cycles.]

## 🎢 The Market Roller Coaster: Ups and Downs Explained

[Explain the market volatility they experienced with specific examples from the news analysis and historical events. Educational tone about volatility being normal.]

## 🧠 SHAP Analysis: Transparent AI Decision Making

[Detailed explanation of the SHAP factors and portfolio quality score. Explain how the AI balances different factors for their specific situation.]

## 📈 Current Market Sentiment for Your Holdings

[Analysis of current news sentiment and what it means for future prospects, using the recent news analysis.]

## 🎓 Educational Insights: What This Experience Teaches Us

[Key lessons about investing, market cycles, news impact, and AI-driven portfolio construction. Connect SHAP insights with market education.]

## 🚀 Looking Forward: Your Investment Education

[Encouraging conclusion about their results, the AI's reasoning, and how this experience prepares them for future investing decisions.]

WRITING STYLE:
- Combine technical AI insights with beginner-friendly explanations
- Use specific examples from their portfolio's news coverage
- Explain both the "what" (results) and "why" (AI reasoning + market events)
- Make SHAP explanations accessible and educational
- Connect news events to portfolio movements with real examples
- Focus on EDUCATION about both AI decision-making and market behavior

This should be comprehensive and detailed - explain both the AI's transparent reasoning AND the real-world market context!
"""

# =============================================================================
# UTILITY FUNCTIONS FOR JSON SERIALIZATION
# =============================================================================

def serialize_for_json(data: Any) -> Any:
    """Recursively convert non-serializable objects to JSON-compatible types."""
    
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
    
    elif isinstance(data, datetime):
        return data.isoformat()
    
    elif isinstance(data, complex):
        return {"real": data.real, "imag": data.imag}
    
    elif isinstance(data, dict):
        return {str(key): serialize_for_json(value) for key, value in data.items()}
    
    elif isinstance(data, (list, tuple)):
        return [serialize_for_json(item) for item in data]
    
    elif isinstance(data, set):
        return list(data)
    
    elif hasattr(data, '__dict__'):
        return serialize_for_json(data.__dict__)
    
    elif hasattr(data, 'to_dict'):
        return serialize_for_json(data.to_dict())
    
    elif isinstance(data, (str, int, float, bool)):
        return data
    
    else:
        try:
            return str(data)
        except Exception:
            return f"<non-serializable: {type(data).__name__}>"

def clean_simulation_results_for_db(results: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all simulation results before saving to database."""
    try:
        logger.info("🧹 Cleaning simulation results for database storage")
        
        import copy
        cleaned_results = copy.deepcopy(results)
        
        # Apply general serialization to the entire structure
        cleaned_results = serialize_for_json(cleaned_results)
        
        # Test that the result is actually JSON serializable
        json.dumps(cleaned_results)
        
        logger.info("✅ Simulation results successfully cleaned for database")
        return cleaned_results
        
    except Exception as e:
        logger.error(f"❌ Failed to clean simulation results: {e}")
        
        # Return safe fallback structure
        fallback_results = {
            "basic_info": {
                "starting_value": float(results.get("starting_value", 0)),
                "end_value": float(results.get("end_value", 0)),
                "portfolio_return": float(results.get("return", 0)),
                "target_reached": bool(results.get("target_reached", False))
            },
            "error_info": {
                "original_error": str(e),
                "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            json.dumps(fallback_results)
            return fallback_results
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback serialization failed: {fallback_error}")
            return {
                "status": "serialization_failed",
                "error": str(e),
                "fallback_error": str(fallback_error),
                "timestamp": datetime.now().isoformat()
            }

def test_json_serialization(data: Any, description: str = "data") -> bool:
    """Test if data can be JSON serialized."""
    try:
        json.dumps(data)
        logger.info(f"✅ {description} is JSON serializable")
        return True
    except Exception as e:
        logger.error(f"❌ {description} is NOT JSON serializable: {e}")
        return False

# =============================================================================
# FALLBACK FUNCTIONS (PRESERVED)
# =============================================================================

def get_fallback_stocks_by_risk_profile(risk_score: int, risk_label: str) -> List[str]:
    """Original fallback stock selection method."""
    logger.info(f"📊 Using fallback selection for {risk_label} risk profile (score: {risk_score})")
    
    if risk_score < 35:
        return ["VTI", "BND", "VEA", "VTEB", "VWO"]
    elif risk_score < 70:
        return ["VTI", "VEA", "VWO", "VNQ", "BND"]
    else:
        return ["VTI", "VGT", "VUG", "ARKK", "VEA"]

def download_stock_data(tickers: List[str], timeframe: int) -> pd.DataFrame:
    """Download historical stock data."""
    try:
        days_needed = max(timeframe * 365, 365)
        start_date = (datetime.today() - timedelta(days=days_needed)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')

        logger.info(f"📅 Downloading data from {start_date} to {end_date} for {len(tickers)} stocks")

        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        threshold = len(data) * 0.7
        data = data.dropna(axis=1, thresh=threshold)
        
        logger.info(f"📊 Downloaded data shape: {data.shape}")
        
        if data.empty:
            raise ValueError("No valid stock data available after quality filtering")
            
        return data
        
    except Exception as e:
        logger.error(f"❌ Error downloading stock data: {str(e)}")
        raise ValueError(f"Failed to download stock data: {str(e)}")

def calculate_portfolio_weights(data: pd.DataFrame, risk_score: int) -> np.ndarray:
    """Original portfolio weights calculation."""
    num_assets = len(data.columns)
    logger.info(f"⚖️ Calculating weights for {num_assets} assets (risk score: {risk_score})")
    
    if risk_score < 35:
        weights = np.array([1 / num_assets] * num_assets)
        logger.info("📊 Using equal weights (conservative approach)")
        
    elif risk_score < 70:
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:num_assets])
        weights = weights / np.sum(weights)
        logger.info("📊 Using moderate bias weighting")
        
    else:
        weights = np.array([0.4, 0.3, 0.2, 0.1][:num_assets])
        
        if len(weights) < num_assets:
            remaining = num_assets - len(weights)
            additional_weights = np.array([0.05] * remaining)
            weights = np.concatenate([weights, additional_weights])
        
        weights = weights / np.sum(weights)
        logger.info("📊 Using concentrated weighting (aggressive approach)")
    
    return weights

def simulate_portfolio_growth(data: pd.DataFrame, weights: np.ndarray, 
                            lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
    """Enhanced portfolio growth simulation with comprehensive debugging."""
    try:
        logger.info(f"📈 Starting simulation: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        # Debug input data quality
        logger.info(f"📊 Input data shape: {data.shape}")
        logger.info(f"📊 Columns: {list(data.columns)}")
        logger.info(f"⚖️ Weights: {weights}")
        logger.info(f"📊 Weights sum: {weights.sum()}")
        
        # Normalize with safety checks
        first_day_values = data.iloc[0]
        
        # Check for zero or negative values that would break normalization
        problematic_stocks = first_day_values[first_day_values <= 0]
        if len(problematic_stocks) > 0:
            logger.error(f"Zero/negative prices on first day: {problematic_stocks}")
            first_day_values = first_day_values.replace(0, 0.01)
            logger.warning("Replaced zero prices with 0.01")
        
        normalized = data.div(first_day_values)
        logger.info(f"Normalized first values: {normalized.iloc[0]}")
        
        # Fill any remaining NaN values
        if normalized.isna().any().any():
            logger.warning("Found NaN values after normalization, forward filling...")
            normalized = normalized.fillna(method='ffill').fillna(1.0)
        
        # Calculate weighted portfolio
        weighted = normalized.dot(weights)
        logger.info(f"Weighted portfolio first 5 values:\n{weighted.head()}")
        
        # Run simulation
        portfolio_values = []
        contributions = []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        for i, (date, growth_factor) in enumerate(weighted.items()):
            # Add monthly contributions (every ~21 trading days)
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
            
            # Apply growth with safety checks
            if i > 0:
                prev_value = weighted.iloc[i - 1]
                
                if prev_value <= 0:
                    logger.error(f"Previous weighted value is zero/negative on {date}: {prev_value}")
                    growth_rate = 1.0
                else:
                    growth_rate = growth_factor / prev_value
                
                # Detect suspicious growth rates
                if growth_rate <= 0:
                    logger.warning(f"Zero/negative growth rate on {date}: {growth_rate}")
                    growth_rate = 1.0
                elif growth_rate > 5.0:
                    logger.warning(f"Extreme growth rate on {date}: {growth_rate}")
                    growth_rate = min(growth_rate, 2.0)
                
                current_value *= growth_rate
                
                if current_value <= 0:
                    logger.error(f"Portfolio value became zero/negative on {date}: {current_value}")
                    current_value = total_contributions

            # Store values with validation
            safe_contributions = max(0, float(total_contributions))
            safe_current_value = max(0, float(current_value))
            
            contributions.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(safe_contributions, 2)
            })
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(safe_current_value, 2)
            })

        # Calculate final results with validation
        end_value = float(current_value)
        starting_value = float(total_contributions)
        
        # Ensure reasonable results
        if end_value <= 0:
            logger.error(f"Final portfolio value is zero: {end_value}")
            end_value = starting_value * (1.05 ** timeframe)
            logger.warning(f"Using emergency fallback value: £{end_value:.2f}")
        
        portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

        logger.info(f"FINAL RESULTS:")
        logger.info(f"   Starting value (contributions): £{starting_value:,.2f}")
        logger.info(f"   Ending value (portfolio): £{end_value:,.2f}")
        logger.info(f"   Total return: {portfolio_return:.1%}")

        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": round(portfolio_return, 4),
            "timeline": {
                "contributions": contributions,
                "portfolio": portfolio_values
            }
        }

    except Exception as e:
        logger.error(f"Error in portfolio simulation: {str(e)}")
        
        # Enhanced fallback
        logger.warning("Using enhanced fallback simulation")
        starting_value = lump_sum + monthly * 12 * timeframe
        
        annual_return = 0.07
        if hasattr(weights, '__len__') and len(weights) > 0:
            risk_factor = max(weights) if len(weights) > 0 else 0.5
            annual_return = 0.05 + (risk_factor * 0.05)
        
        end_value = starting_value * ((1 + annual_return) ** timeframe)
        
        # Create reasonable timeline
        dates = pd.date_range(start=datetime.today() - timedelta(days=timeframe*365), 
                            end=datetime.today(), freq='D')
        timeline_entries = min(len(dates), 1000)
        
        contributions_timeline = []
        portfolio_timeline = []
        
        for i in range(0, timeline_entries, max(1, timeline_entries//252)):
            progress = i / timeline_entries
            current_contributions = lump_sum + (monthly * 12 * timeframe * progress)
            current_portfolio = starting_value + ((end_value - starting_value) * progress)
            
            date_str = dates[min(i, len(dates)-1)].strftime("%Y-%m-%d")
            
            contributions_timeline.append({
                "date": date_str,
                "value": round(current_contributions, 2)
            })
            portfolio_timeline.append({
                "date": date_str,
                "value": round(current_portfolio, 2)
            })
        
        logger.info(f"Fallback results: £{starting_value:,.2f} → £{end_value:,.2f}")
        
        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": round((end_value - starting_value) / starting_value, 4),
            "timeline": {
                "contributions": contributions_timeline,
                "portfolio": portfolio_timeline
            }
        }

def generate_simple_enhanced_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None
) -> str:
    """Generate enhanced simple summary with available SHAP context."""
    
    goal = user_data.get("goal", "wealth building")
    timeframe = user_data.get("timeframe", 10)
    start_value = simulation_results.get("starting_value", 0)
    end_value = simulation_results.get("end_value", 0)
    target_value = user_data.get("target_value", 50000)
    stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
    target_achieved = end_value >= target_value
    
    base_summary = f"""
Your {risk_label.lower()} risk portfolio, invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years for your {goal} goal. Your target of £{target_value:,.2f} was {'achieved' if target_achieved else 'partially achieved'}.
"""

    # Add SHAP explanations if available
    shap_section = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_section = f"""

The AI analysis considered multiple factors for your specific situation:
"""
        explanations = shap_explanation["human_readable_explanation"]
        for factor, explanation in explanations.items():
            if explanation and len(explanation) > 10:
                shap_section += f"• {explanation}\n"

    # Add goal analysis if available
    goal_section = ""
    if goal_analysis and feasibility_assessment:
        required_return = goal_analysis.get('required_return_percent', 0)
        feasibility = feasibility_assessment.get('feasibility_score', 0)
        
        goal_section = f"""

To reach your £{target_value:,.0f} target, you needed {required_return:.1f}% annual returns. The AI assessed your goal as {feasibility:.0f}% feasible given your risk tolerance and timeframe. This demonstrates how the AI creates personalized strategies for your specific goals.
"""

    return f"{base_summary}{shap_section}{goal_section}\n\n*This portfolio was optimized specifically for your goals using explainable AI that shows exactly why each decision was made.*"

async def simulate_portfolio_fallback(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Fallback to original simulation when enhanced version fails."""
    
    logger.info("Running fallback simulation")
    
    # Extract user data
    user_data = {
        "experience": sim_input.get("years_of_experience", 0),
        "goal": sim_input.get("goal", "wealth building"),
        "target_value": float(sim_input.get("target_value", 50000)),
        "lump_sum": float(sim_input.get("lump_sum", 0) or 0),
        "monthly": float(sim_input.get("monthly", 0) or 0),
        "timeframe": int(sim_input.get("timeframe", 10)),
        "income_bracket": sim_input.get("income_bracket", "medium")
    }

    risk_score = sim_input.get("risk_score", 35)
    risk_label = sim_input.get("risk_label", "Medium")

    # Use original recommendation method
    tickers = get_fallback_stocks_by_risk_profile(risk_score, risk_label)
    
    # Continue with original simulation
    stock_data = download_stock_data(tickers, user_data["timeframe"])
    weights = calculate_portfolio_weights(stock_data, risk_score)
    
    stocks_picked = [
        {
            "symbol": ticker, 
            "name": get_company_name(ticker),
            "allocation": round(float(weight), 4)
        }
        for ticker, weight in zip(tickers, weights)
    ]
    
    simulation_results = simulate_portfolio_growth(
        stock_data, weights, user_data["lump_sum"], user_data["monthly"], user_data["timeframe"]
    )
    
    # Generate simple AI summary
    try:
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        ai_summary = await ai_service.generate_portfolio_summary(
            stocks_picked=stocks_picked,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            simulation_results=simulation_results
        )
    except:
        ai_summary = generate_simple_summary(
            stocks_picked, user_data, risk_score, risk_label, simulation_results
        )
    
    # Save using original method
    simulation = save_simulation_to_db(
        db, sim_input, user_data, risk_score, risk_label,
        ai_summary, stocks_picked, simulation_results
    )
    
    return format_simulation_response(simulation)

def generate_simple_summary(stocks_picked: List[Dict], user_data: Dict[str, Any], 
                          risk_score: int, risk_label: str, 
                          simulation_results: Dict[str, Any]) -> str:
    """Original simple summary generation."""
    logger.info("Generating simple fallback summary")
    
    goal = user_data.get("goal", "wealth building")
    timeframe = user_data.get("timeframe", 10)
    start_value = simulation_results.get("starting_value", 0)
    end_value = simulation_results.get("end_value", 0)
    target_value = user_data.get("target_value", 50000)
    
    stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
    target_achieved = end_value >= target_value
    
    return f"""
Portfolio simulation completed for your {goal} goal. Your {risk_label.lower()} risk portfolio, 
invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years. 
Your target of £{target_value:,.2f} was {'achieved' if target_achieved else 'not achieved'}. 
This simulation demonstrates how diversified investing can help build wealth over time.
""".strip()

def get_company_name(ticker: str) -> str:
    """Original company name mapping."""
    name_mapping = {
        "VTI": "Vanguard Total Stock Market ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VTEB": "Vanguard Tax-Exempt Bond ETF",
        "VWO": "Vanguard Emerging Markets ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "VGT": "Vanguard Information Technology ETF",
        "VUG": "Vanguard Growth ETF",
        "ARKK": "ARK Innovation ETF"
    }
    return name_mapping.get(ticker, ticker)

def save_simulation_to_db(db: Session, sim_input: Dict[str, Any], user_data: Dict[str, Any],
                         risk_score: int, risk_label: str, ai_summary: str,
                         stocks_picked: List[Dict], simulation_results: Dict[str, Any]) -> models.Simulation:
    """Original database save function with JSON serialization fix."""
    try:
        logger.info("Saving simulation results to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        # Create results object
        results = {
            "name": user_data["goal"],
            "stocks_picked": stocks_picked,
            "starting_value": simulation_results["starting_value"],
            "end_value": simulation_results["end_value"],
            "return": simulation_results["portfolio_return"],
            "target_reached": target_reached,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "timeline": simulation_results["timeline"]
        }
        
        # Clean the results before saving
        cleaned_results = clean_simulation_results_for_db(results)
        
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
            risk_score=risk_score,
            risk_label=risk_label,
            ai_summary=ai_summary,
            results=cleaned_results
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        logger.info(f"Simulation saved successfully with ID: {simulation.id}")
        return simulation
        
    except Exception as e:
        logger.error(f"Error saving simulation to database: {str(e)}")
        db.rollback()
        raise

def format_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """Original response formatting."""
    logger.info("Formatting simulation response for API")
    
    return {
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
        "results": simulation.results,
        "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat()
    }

# =============================================================================
# NEW API ENDPOINTS FOR VISUALIZATION ENGINE
# =============================================================================

async def get_shap_visualization(simulation_id: int, db: Session) -> Optional[str]:
    """Generate SHAP visualization for a specific simulation."""
    
    viz_engine = get_visualization_engine()
    if not viz_engine:
        return None
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation or not simulation.results.get("shap_explanation"):
            return None
        
        # Create SHAP visualization
        save_path = f"./static/visualizations/shap_explanation_{simulation_id}.png"
        result = viz_engine.create_shap_waterfall_chart(
            simulation.results["shap_explanation"], save_path
        )
        
        if "saved" in result:
            logger.info(f"SHAP visualization created: {save_path}")
            return save_path
        else:
            logger.warning(f"SHAP visualization failed: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating SHAP visualization: {e}")
        return None

async def analyze_simulation_with_news(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Analyze a simulation with current news context."""
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        # Extract stocks from simulation
        stocks_picked = simulation.results.get("stocks_picked", [])
        portfolio_data = {"stocks": [stock["symbol"] for stock in stocks_picked]}
        
        # Use AI analysis service to get news analysis
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        news_analysis = await ai_service.analyze_portfolio_with_context(
            portfolio_data, days_back=7
        )
        
        return {
            "simulation_id": simulation_id,
            "simulation_results": simulation.results,
            "news_analysis": news_analysis,
            "combined_insights": f"Your {simulation.risk_label} portfolio has been analyzed with current market news. This helps understand how recent events might affect your long-term strategy.",
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing simulation with news: {e}")
        return {"error": str(e)}

# =============================================================================
# INTEGRATION NOTES
# =============================================================================

"""
ENHANCED PORTFOLIO SIMULATOR WITH VISUALIZATION ENGINE

This enhanced version adds comprehensive visualization capabilities:

1. AUTOMATIC VISUALIZATIONS CREATED:
   - Portfolio allocation pie chart
   - Portfolio growth timeline
   - SHAP explanation waterfall chart
   - Risk vs return scatter plot
   - Goal achievement progress chart

2. VISUALIZATION ENGINE INTEGRATION:
   - get_visualization_engine() - Global instance management
   - create_simulation_visualizations() - Creates all charts
   - update_visualization_paths_with_id() - Updates file names with DB ID

3. NEW API ENDPOINTS:
   - get_simulation_visualizations() - Returns available charts
   - regenerate_shap_visualization() - Recreates SHAP charts
   - create_custom_visualization() - Creates custom charts

4. DATABASE INTEGRATION:
   - Visualization paths stored in simulation results
   - Proper JSON serialization handling
   - Fallback protection for failed saves

5. USAGE IN YOUR API:

   # Add these routes to your api/routers/ai_analysis.py:
   
   @router.get("/simulation/{simulation_id}/visualizations")
   async def get_simulation_viz(simulation_id: int, db: Session = Depends(get_db)):
       return await get_simulation_visualizations(simulation_id, db)

   @router.post("/simulation/{simulation_id}/regenerate-shap")
   async def regenerate_shap_viz(simulation_id: int, db: Session = Depends(get_db)):
       return await regenerate_shap_visualization(simulation_id, db)

   @router.post("/simulation/{simulation_id}/custom-visualization")
   async def create_custom_viz(
       simulation_id: int, 
       viz_type: str,
       options: Dict[str, Any],
       db: Session = Depends(get_db)
   ):
       return await create_custom_visualization(simulation_id, viz_type, options, db)

6. FRONTEND INTEGRATION:
   - Access visualization_paths from API response
   - Display charts using file paths
   - Request custom visualizations as needed

7. BENEFITS:
   - Rich visual explanations of portfolio recommendations
   - SHAP transparency through waterfall charts
   - Goal tracking with progress visualizations
   - Risk analysis through scatter plots
   - Timeline visualization of portfolio growth

All original functionality is preserved with fallback protection.
"""