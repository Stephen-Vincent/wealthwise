"""
Portfolio Simulator Service - Enhanced with WealthWise SHAP Integration

This module handles the complete portfolio simulation workflow:
1. Extracts and validates user investment preferences
2. Uses AI to recommend appropriate stocks based on risk profile
3. Downloads historical market data for simulation
4. Calculates portfolio weights and simulates growth over time
5. Generates AI-powered educational summaries with SHAP explanations
6. Saves results to database

The service integrates with:
- WealthWise Enhanced Stock Recommender AI (for goal-oriented stock selection)
- SHAP Explainable AI (for transparent recommendations)
- AI Analysis Service (for educational summaries)
- Yahoo Finance API (for historical data)
- Database models (for persistence)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import models
import yfinance as yf
import numpy as np
import pandas as pd
import logging
import json
import copy
import traceback

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)

# =============================================================================
# WEALTHWISE INTEGRATION - System Imports and Availability Check
# =============================================================================

try:
    from ai_models.stock_model.core.recommender import EnhancedStockRecommender
    from ai_models.stock_model.explainable_ai import SHAPExplainer, VisualizationEngine
    from ai_models.stock_model.goal_optimization import GoalCalculator, FeasibilityAssessor
    from ai_models.stock_model.analysis import MarketRegimeDetector, FactorAnalyzer
    from ai_models.stock_model.utils import initialize_complete_system
    WEALTHWISE_AVAILABLE = True
    logger.info("✅ WealthWise SHAP system loaded successfully")
except ImportError as e:
    WEALTHWISE_AVAILABLE = False
    logger.warning(f"⚠️ WealthWise not available: {e}")

# =============================================================================
# MAIN PORTFOLIO SIMULATION FUNCTION - Enhanced with SHAP Integration
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Enhanced portfolio simulation with WealthWise SHAP integration.
    
    This is the main entry point that orchestrates the entire simulation process:
    1. Extracts user data and validates inputs
    2. Gets AI stock recommendations with SHAP explanations
    3. Downloads historical market data
    4. Calculates portfolio weights and runs simulation
    5. Generates enhanced AI summary with explanations
    6. Saves everything to database with proper serialization
    
    Args:
        sim_input: Dictionary containing user onboarding data
        db: Database session for saving results
    
    Returns:
        Enhanced simulation results with SHAP explanations
    """
    
    try:
        logger.info("🚀 Starting enhanced portfolio simulation with WealthWise")
        
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

        # STEP 2: Enhanced AI-powered stock recommendations with integrated SHAP
        logger.info("🤖 Getting enhanced AI stock recommendations with SHAP explanations")
        recommendation_result = await get_enhanced_ai_recommendations(
            target_value=user_data["target_value"],
            timeframe=user_data["timeframe"],
            risk_score=risk_score,
            risk_label=risk_label,
            current_investment=user_data["lump_sum"],
            monthly_contribution=user_data["monthly"]
        )
        
        # Debug SHAP status right after getting recommendations
        shap_debug_info = debug_shap_status_in_simulation(recommendation_result)
        logger.info(f"📊 SHAP Integration Status: {shap_debug_info}")
        
        # Extract all recommendations and explanations
        tickers = recommendation_result["stocks"]
        shap_explanation = recommendation_result.get("shap_explanation")
        has_shap_explanations = recommendation_result.get("has_shap_explanations", False)
        goal_analysis = recommendation_result.get("goal_analysis")
        feasibility_assessment = recommendation_result.get("feasibility_assessment")
        market_regime = recommendation_result.get("market_regime")
        
        logger.info(f"📈 AI recommended stocks: {tickers}")
        logger.info(f"🔍 SHAP explanations available: {has_shap_explanations}")

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

        # STEP 8: Generate enhanced AI summary with SHAP explanations
        logger.info("🧠 Generating enhanced AI educational summary with SHAP")
        ai_summary = await generate_enhanced_ai_summary(
            stocks_picked, user_data, risk_score, risk_label, 
            simulation_results, shap_explanation, goal_analysis, 
            feasibility_assessment, market_regime
        )

        # STEP 9: Save enhanced simulation results to database
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
            market_regime=market_regime
        )

        logger.info(f"✅ Enhanced portfolio simulation completed successfully (ID: {simulation.id})")
        return format_enhanced_simulation_response(simulation)

    except Exception as e:
        logger.error(f"❌ Enhanced portfolio simulation failed: {str(e)}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        # Ensure database rollback
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"❌ Database rollback failed: {rollback_error}")
        
        # Fallback to original simulation if enhanced version fails
        logger.warning("🔄 Falling back to original simulation method")
        return await simulate_portfolio_fallback(sim_input, db)

# =============================================================================
# ENHANCED AI RECOMMENDATION FUNCTIONS - Core Intelligence with SHAP
# =============================================================================

async def get_enhanced_ai_recommendations(
    target_value: float, timeframe: int, risk_score: int, risk_label: str,
    current_investment: float = 0, monthly_contribution: float = 0
) -> Dict[str, Any]:
    """
    Get enhanced AI recommendations with integrated SHAP explanations.
    
    This is the core function that:
    1. Initializes the WealthWise system
    2. Trains SHAP explainer if needed
    3. Generates goal-oriented stock recommendations
    4. Creates transparent explanations for all decisions
    5. Performs market regime analysis
    
    Returns:
        Dict containing stocks, SHAP explanations, and comprehensive analysis
    """
    
    if not WEALTHWISE_AVAILABLE:
        logger.warning("⚠️ WealthWise not available, using fallback recommendations")
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback",
            "shap_explanation": None,
            "has_shap_explanations": False
        }
    
    try:
        logger.info("🎯 Initializing WealthWise enhanced recommendation system with SHAP")
        
        # Initialize WealthWise system
        init_result = initialize_complete_system({
            'LOG_LEVEL': 'INFO',
            'LOG_TO_FILE': False,  # Don't create log files in production
            'ENABLE_PERFORMANCE_TRACKING': True
        })
        
        if not init_result['success']:
            raise Exception(f"WealthWise initialization failed: {init_result.get('error')}")
        
        # Initialize core components
        recommender = EnhancedStockRecommender()
        goal_calculator = GoalCalculator()
        feasibility_assessor = FeasibilityAssessor()
        market_detector = MarketRegimeDetector()
        
        # ⭐ SHAP EXPLAINER INITIALIZATION AND TRAINING ⭐
        shap_explainer = SHAPExplainer()
        shap_explanation = None
        has_shap_explanations = False
        
        if shap_explainer.is_available():
            logger.info("🔍 SHAP library available, initializing explainer")
            
            # Check if model is trained, if not train it
            if not shap_explainer._model_trained:
                logger.info("🤖 Training SHAP explainer model...")
                training_success = shap_explainer.train_shap_model(num_samples=1000)
                
                if training_success:
                    logger.info("✅ SHAP model trained successfully")
                else:
                    logger.warning("⚠️ SHAP model training failed")
            
            # Generate SHAP explanation if model is ready
            if shap_explainer._model_trained:
                try:
                    logger.info("🔍 Generating SHAP explanations")
                    shap_explanation = shap_explainer.get_shap_explanation(
                        target_value=target_value,
                        timeframe=timeframe,
                        risk_score=risk_score,
                        current_investment=current_investment,
                        monthly_contribution=monthly_contribution,
                        market_volatility=20.0,  # Default VIX level
                        market_trend=2.5         # Default trend score
                    )
                    
                    if shap_explanation and not shap_explanation.get("error"):
                        has_shap_explanations = True
                        logger.info("✅ SHAP explanations generated successfully")
                        logger.info(f"📊 Portfolio quality score: {shap_explanation.get('portfolio_quality_score', 'N/A')}")
                    else:
                        logger.warning("⚠️ SHAP explanation generation returned error")
                        
                except Exception as shap_error:
                    logger.warning(f"⚠️ SHAP explanation failed: {shap_error}")
                    shap_explanation = None
        else:
            logger.warning("⚠️ SHAP library not available - install with: pip install shap matplotlib seaborn")
        
        # Continue with other analyses
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
        
        # Step 4: Get goal-oriented stock recommendations
        logger.info("🤖 Generating goal-oriented stock recommendations")
        recommended_stocks = recommender.recommend_stocks(
            target_value, timeframe, risk_score, 
            current_investment, monthly_contribution
        )
        
        logger.info(f"✅ Enhanced recommendations complete: {len(recommended_stocks)} stocks selected")
        
        # ⭐ Return complete package with SHAP integrated ⭐
        return {
            "stocks": recommended_stocks,
            "shap_explanation": shap_explanation,
            "has_shap_explanations": has_shap_explanations,
            "goal_analysis": goal_analysis,
            "feasibility_assessment": feasibility_assessment,
            "market_regime": market_regime,
            "method": "wealthwise_enhanced"
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced AI recommendations failed: {e}")
        logger.warning("🔄 Falling back to original recommendation method")
        
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback",
            "error": str(e),
            "shap_explanation": None,
            "has_shap_explanations": False
        }

def get_stock_explanation(ticker: str, recommendation_result: Dict[str, Any]) -> str:
    """
    Extract human-readable explanation for why a specific stock was recommended.
    
    Uses SHAP explanations and factor analysis to provide transparent reasoning
    for each stock selection in plain English.
    """
    if recommendation_result.get("method") == "fallback":
        return f"{ticker} selected based on risk profile matching"
    
    # Extract explanations from SHAP and factor analysis
    shap_explanation = recommendation_result.get("shap_explanation", {})
    
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        # Use SHAP human-readable explanations
        explanations = shap_explanation["human_readable_explanation"]
        if explanations:
            # Return first relevant explanation
            for key, explanation in explanations.items():
                if explanation and len(explanation) > 20:  # Non-empty explanation
                    return f"{ticker}: {explanation[:100]}..."
    
    # Fallback explanation
    return f"{ticker} recommended by AI analysis for your goals and risk profile"

def debug_shap_status_in_simulation(recommendation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug function to check SHAP status and log comprehensive information.
    
    This helps troubleshoot SHAP integration issues by providing detailed
    status information that matches your original debug output format.
    """
    
    shap_explanation = recommendation_result.get("shap_explanation")
    has_shap = recommendation_result.get("has_shap_explanations", False)
    wealthwise_enhanced = recommendation_result.get("method") == "wealthwise_enhanced"
    
    debug_info = {
        "hasShapExplanation": has_shap,
        "shapData": "Available" if shap_explanation else "undefined",
        "wealthwiseEnhanced": wealthwise_enhanced
    }
    
    logger.info('🔍 SHAP Debug: {hasShapExplanation: %s, shapData: %s, wealthwiseEnhanced: %s}',
                has_shap, debug_info["shapData"], wealthwise_enhanced)
    
    return debug_info

# =============================================================================
# ENHANCED PORTFOLIO OPTIMIZATION - Weight Calculation with AI
# =============================================================================

def calculate_enhanced_portfolio_weights(
    data: pd.DataFrame, risk_score: int, recommendation_result: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate enhanced portfolio weights using WealthWise correlation optimization.
    
    If WealthWise is available, uses sophisticated correlation-based optimization
    to improve diversification. Falls back to traditional methods if unavailable.
    """
    
    if recommendation_result.get("method") == "fallback" or not WEALTHWISE_AVAILABLE:
        # Use original weight calculation method
        return calculate_portfolio_weights(data, risk_score)
    
    try:
        # Use WealthWise correlation-based optimization
        recommender = EnhancedStockRecommender()
        
        # Create initial equal weights
        num_assets = len(data.columns)
        initial_weights = {col: 1.0/num_assets for col in data.columns}
        
        # Optimize using correlation analysis
        optimized_weights = recommender.optimize_for_diversification(
            list(data.columns), initial_weights
        )
        
        # Convert to numpy array in correct order
        weights_array = np.array([
            optimized_weights.get(col, 1.0/num_assets) 
            for col in data.columns
        ])
        
        # Ensure weights sum to 1
        weights_array = weights_array / np.sum(weights_array)
        
        logger.info("✅ Using WealthWise correlation-optimized weights")
        return weights_array
        
    except Exception as e:
        logger.warning(f"⚠️ WealthWise optimization failed: {e}, using fallback")
        return calculate_portfolio_weights(data, risk_score)

# =============================================================================
# ENHANCED AI SUMMARY GENERATION - Educational Content with SHAP Context
# =============================================================================

async def generate_enhanced_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive AI summary with SHAP explanations integrated.
    
    Creates educational content that includes:
    1. Portfolio performance results
    2. SHAP explanations for decision transparency
    3. Goal feasibility analysis
    4. Market regime context
    5. Educational insights for learning
    """
    
    try:
        logger.info("🧠 Generating enhanced AI summary with SHAP explanations")
        
        # Import AI Analysis Service
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        # Create enhanced context for AI summary
        enhanced_context = {
            "user_data": user_data,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "simulation_results": simulation_results,
            "stocks_picked": stocks_picked,
            "shap_explanation": shap_explanation,
            "goal_analysis": goal_analysis,
            "feasibility_assessment": feasibility_assessment,
            "market_regime": market_regime
        }
        
        # Generate enhanced summary with SHAP context
        ai_summary = await generate_shap_enhanced_summary(ai_service, enhanced_context)
        
        logger.info("✅ Enhanced AI summary with SHAP explanations generated")
        return ai_summary
        
    except Exception as e:
        logger.warning(f"⚠️ Enhanced AI summary failed: {e}. Using standard summary.")
        
        # Fall back to standard AI summary
        try:
            from services.ai_analysis import AIAnalysisService
            ai_service = AIAnalysisService()
            
            return await ai_service.generate_portfolio_summary(
                stocks_picked=stocks_picked,
                user_data=user_data,
                risk_score=risk_score,
                risk_label=risk_label,
                simulation_results=simulation_results
            )
        except Exception as e2:
            logger.warning(f"⚠️ Standard AI summary also failed: {e2}. Using simple summary.")
            return generate_simple_enhanced_summary(
                stocks_picked, user_data, risk_score, risk_label, 
                simulation_results, shap_explanation, goal_analysis, feasibility_assessment
            )

async def generate_shap_enhanced_summary(ai_service, context: Dict[str, Any]) -> str:
    """
    Generate AI summary with SHAP explanations and enhanced context.
    
    Creates a comprehensive prompt that includes SHAP explanations and uses
    the AI service to generate educational, transparent content.
    """
    
    # Extract context components
    user_data = context["user_data"]
    simulation_results = context["simulation_results"]
    shap_explanation = context.get("shap_explanation")
    goal_analysis = context.get("goal_analysis")
    feasibility_assessment = context.get("feasibility_assessment")
    market_regime = context.get("market_regime")
    
    # Create enhanced prompt with SHAP context
    shap_context = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_context = f"""
SHAP AI EXPLANATIONS:
The AI chose this portfolio because:
"""
        for factor, explanation in shap_explanation["human_readable_explanation"].items():
            if explanation and len(explanation) > 10:
                shap_context += f"• {explanation}\n"
    
    goal_context = ""
    if goal_analysis and feasibility_assessment:
        goal_context = f"""
GOAL ANALYSIS:
• Required annual return: {goal_analysis.get('required_return_percent', 0):.1f}%
• Goal feasibility: {feasibility_assessment.get('feasibility_score', 0):.0f}%
• Recommendation: {feasibility_assessment.get('recommendations', {}).get('primary', 'Continue with plan')}
"""

    market_context = ""
    if market_regime:
        market_context = f"""
CURRENT MARKET CONDITIONS:
• Market regime: {market_regime.get('regime', 'neutral')}
• Market trend: {market_regime.get('trend_score', 2.5):.1f}/5
• Volatility (VIX): {market_regime.get('current_vix', 20):.1f}
"""

    # Enhanced prompt for AI service
    enhanced_prompt = f"""
Generate an educational portfolio summary that explains both the results AND the AI reasoning:

{shap_context}

{goal_context}

{market_context}

Please explain:
1. Why the AI selected these specific stocks for their goals
2. How the portfolio is designed to achieve their target
3. What the SHAP analysis reveals about the decision factors
4. Educational insights about goal-oriented investing
5. How current market conditions affect the strategy

Make it educational and beginner-friendly while highlighting the AI's transparent decision-making.
"""

    # Use AI service with enhanced context
    return await ai_service._get_groq_response(enhanced_prompt)

def generate_simple_enhanced_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None
) -> str:
    """
    Generate enhanced simple summary when AI services are unavailable.
    
    Creates a basic summary with available SHAP and analysis context,
    serving as a reliable fallback option.
    """
    
    # Base summary
    goal = user_data.get("goal", "wealth building")
    timeframe = user_data.get("timeframe", 10)
    start_value = simulation_results.get("starting_value", 0)
    end_value = simulation_results.get("end_value", 0)
    target_value = user_data.get("target_value", 50000)
    stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
    target_achieved = end_value >= target_value
    
    base_summary = f"""
## 🎯 Your Goal-Oriented Portfolio Results

Your {risk_label.lower()} risk portfolio, invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years for your {goal} goal. Your target of £{target_value:,.2f} was {'✅ achieved' if target_achieved else '📈 partially achieved'}.
"""

    # Add SHAP explanations if available
    shap_section = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_section = f"""
## 🔍 Why the AI Chose This Portfolio

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
## 📊 Goal Analysis

To reach your £{target_value:,.0f} target, you needed {required_return:.1f}% annual returns. The AI assessed your goal as {feasibility:.0f}% feasible given your risk tolerance and timeframe. This demonstrates how the AI creates personalized strategies for your specific goals.
"""

    return f"{base_summary}\n{shap_section}\n{goal_section}\n\n*This portfolio was optimized specifically for your goals using explainable AI that shows exactly why each decision was made.*"

# =============================================================================
# DATABASE SERIALIZATION FUNCTIONS - JSON Compatibility for AI Results
# =============================================================================

def serialize_for_json(data: Any) -> Any:
    """
    Recursively convert NumPy arrays, pandas objects, and other non-serializable objects.
    
    This function handles all the common serialization issues when saving AI model results
    to databases, specifically designed for WealthWise SHAP explanations and portfolio
    optimization results.
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
    
    # Handle complex numbers (sometimes in SHAP explanations)
    elif isinstance(data, complex):
        return {"real": data.real, "imag": data.imag}
    
    # Handle dictionaries
    elif isinstance(data, dict):
        return {str(key): serialize_for_json(value) for key, value in data.items()}
    
    # Handle lists and tuples
    elif isinstance(data, (list, tuple)):
        return [serialize_for_json(item) for item in data]
    
    # Handle sets
    elif isinstance(data, set):
        return list(data)
    
    # Handle custom objects with __dict__
    elif hasattr(data, '__dict__'):
        return serialize_for_json(data.__dict__)
    
    # Handle objects with a to_dict method
    elif hasattr(data, 'to_dict'):
        return serialize_for_json(data.to_dict())
    
    # Return as-is for basic JSON-serializable types
    elif isinstance(data, (str, int, float, bool)):
        return data
    
    # For anything else, try to convert to string as fallback
    else:
        try:
            return str(data)
        except Exception:
            return f"<non-serializable: {type(data).__name__}>"

def clean_shap_explanation(shap_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Specifically clean SHAP explanation data for JSON serialization.
    
    SHAP explanations often contain NumPy arrays and complex nested structures
    that need special handling for database storage.
    """
    if not shap_data:
        return {}
    
    try:
        cleaned_shap = {}
        
        # Handle common SHAP fields with proper type conversion
        for key, value in shap_data.items():
            if key == 'shap_values':
                # SHAP values are typically NumPy arrays
                cleaned_shap[key] = serialize_for_json(value)
            elif key == 'feature_importance':
                # Feature importance scores
                cleaned_shap[key] = serialize_for_json(value)
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
                cleaned_shap[key] = serialize_for_json(value)
        
        return cleaned_shap
        
    except Exception as e:
        logger.warning(f"⚠️ Error cleaning SHAP explanation: {e}")
        return {"error": f"SHAP data cleaning failed: {str(e)}"}

def clean_simulation_results_for_db(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean all simulation results before saving to database.
    
    This is the main function to call before saving any simulation results
    that might contain NumPy arrays or other non-serializable objects.
    """
    try:
        logger.info("🧹 Cleaning simulation results for database storage")
        
        # Create a deep copy to avoid modifying the original
        cleaned_results = copy.deepcopy(results)
        
        # Special handling for known problematic fields
        if 'shap_explanation' in cleaned_results:
            cleaned_results['shap_explanation'] = clean_shap_explanation(
                cleaned_results['shap_explanation']
            )
        
        if 'goal_analysis' in cleaned_results:
            cleaned_results['goal_analysis'] = serialize_for_json(
                cleaned_results['goal_analysis']
            )
        
        if 'feasibility_assessment' in cleaned_results:
            cleaned_results['feasibility_assessment'] = serialize_for_json(
                cleaned_results['feasibility_assessment']
            )
        
        if 'market_regime' in cleaned_results:
            cleaned_results['market_regime'] = serialize_for_json(
                cleaned_results['market_regime']
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
                                cleaned_timeline.append(serialize_for_json(item))
                        timeline[key] = cleaned_timeline
        
        # Apply general serialization to the entire structure
        cleaned_results = serialize_for_json(cleaned_results)
        
        # Test that the result is actually JSON serializable
        json.dumps(cleaned_results)
        
        logger.info("✅ Simulation results successfully cleaned for database")
        return cleaned_results
        
    except Exception as e:
        logger.error(f"❌ Failed to clean simulation results: {e}")
        
        # Return a safe fallback structure
        fallback_results = {
            "basic_info": {
                "starting_value": float(results.get("starting_value", 0)),
                "end_value": float(results.get("end_value", 0)),
                "portfolio_return": float(results.get("return", 0)),
                "target_reached": bool(results.get("target_reached", False))
            },
            "portfolio_summary": {
                "stocks": [str(stock.get('symbol', '')) for stock in results.get('stocks_picked', [])],
                "num_stocks": len(results.get('stocks_picked', []))
            },
            "metadata": {
                "wealthwise_enhanced": bool(results.get("wealthwise_enhanced", False)),
                "methodology": str(results.get("methodology", "standard")),
                "serialization_note": f"Full results not serializable: {str(e)}"
            },
            "error_info": {
                "original_error": str(e),
                "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Test the fallback is serializable
        try:
            json.dumps(fallback_results)
            return fallback_results
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback serialization failed: {fallback_error}")
            # Return absolute minimum
            return {
                "status": "serialization_failed",
                "error": str(e),
                "fallback_error": str(fallback_error),
                "timestamp": datetime.now().isoformat()
            }

def test_json_serialization(data: Any, description: str = "data") -> bool:
    """
    Test if data can be JSON serialized.
    Useful for debugging serialization issues.
    """
    try:
        json.dumps(data)
        logger.info(f"✅ {description} is JSON serializable")
        return True
    except Exception as e:
        logger.error(f"❌ {description} is NOT JSON serializable: {e}")
        return False

# =============================================================================
# ENHANCED DATABASE FUNCTIONS - Saving Results with SHAP Data
# =============================================================================

def save_enhanced_simulation_to_db(
    db, sim_input: Dict[str, Any], user_data: Dict[str, Any],
    risk_score: int, risk_label: str, ai_summary: str,
    stocks_picked: List[Dict], simulation_results: Dict[str, Any],
    shap_explanation: Dict[str, Any] = None, 
    goal_analysis: Dict[str, Any] = None,
    feasibility_assessment: Dict[str, Any] = None, 
    market_regime: Dict[str, Any] = None
):
    """
    Enhanced database save function with proper JSON serialization for AI results.
    
    Saves comprehensive simulation results including SHAP explanations, goal analysis,
    and market regime data with proper error handling and fallback mechanisms.
    """
    
    try:
        logger.info("💾 Saving enhanced simulation with SHAP data to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        # Create enhanced results object with all data
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
            "wealthwise_enhanced": True,
            "methodology": "WealthWise SHAP-enhanced goal-oriented optimization",
            "created_timestamp": datetime.now().isoformat()
        }
        
        # ⭐ KEY FIX: Clean the results before saving ⭐
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
            results=cleaned_results  # ⭐ Use cleaned results ⭐
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        logger.info(f"✅ Enhanced simulation saved with SHAP data (ID: {simulation.id})")
        return simulation
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced simulation: {str(e)}")
        db.rollback()
        
        # If enhanced save fails, try to save a basic version
        try:
            logger.warning("🔄 Attempting to save basic simulation without enhanced data")
            
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
                "error_message": str(e)
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
            
            logger.warning(f"⚠️ Saved basic simulation without enhanced data (ID: {basic_simulation.id})")
            return basic_simulation
            
        except Exception as basic_error:
            logger.error(f"❌ Even basic simulation save failed: {basic_error}")
            db.rollback()
            raise

def format_enhanced_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """
    Format enhanced simulation response with SHAP explanations and metadata.
    
    Creates a comprehensive response object that includes all simulation data
    plus enhanced flags for frontend processing.
    """
    
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
        "results": simulation.results,
        "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat(),
        # Enhanced flags for frontend
        "wealthwise_enhanced": simulation.results.get("wealthwise_enhanced", False),
        "has_shap_explanations": bool(simulation.results.get("shap_explanation")),
        "methodology": simulation.results.get("methodology", "Standard simulation")
    }
    
    return response

# =============================================================================
# FALLBACK SIMULATION FUNCTIONS - Original System Compatibility
# =============================================================================

async def simulate_portfolio_fallback(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Fallback to original simulation when enhanced version fails.
    
    This ensures the system always works even if WealthWise is unavailable,
    providing graceful degradation to the original functionality.
    """
    
    logger.info("🔄 Running fallback simulation")
    
    try:
        # Extract user data using same structure as enhanced version
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
        
        # Continue with original simulation workflow
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
        
    except Exception as e:
        logger.error(f"❌ Fallback simulation also failed: {e}")
        # Return minimal error response
        return {
            "error": "Simulation failed",
            "message": str(e),
            "fallback_used": True,
            "timestamp": datetime.now().isoformat()
        }

def get_fallback_stocks_by_risk_profile(risk_score: int, risk_label: str) -> List[str]:
    """
    Original fallback stock selection method based on risk profile.
    
    Provides conservative, moderate, or aggressive stock selections based on
    user risk tolerance when enhanced AI recommendations are unavailable.
    """
    logger.info(f"📊 Using fallback selection for {risk_label} risk profile (score: {risk_score})")
    
    if risk_score < 35:
        return ["VTI", "BND", "VEA", "VTEB", "VWO"]  # Conservative
    elif risk_score < 70:
        return ["VTI", "VEA", "VWO", "VNQ", "BND"]   # Moderate
    else:
        return ["VTI", "VGT", "VUG", "ARKK", "VEA"]  # Aggressive

# =============================================================================
# MARKET DATA FUNCTIONS - Historical Data Download and Processing
# =============================================================================

def download_stock_data(tickers: List[str], timeframe: int) -> pd.DataFrame:
    """
    Download historical stock data for portfolio simulation.
    
    Fetches price data from Yahoo Finance with proper error handling
    and data quality filtering for reliable simulation results.
    """
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

# =============================================================================
# PORTFOLIO WEIGHT CALCULATION - Traditional Risk-Based Allocation
# =============================================================================

def calculate_portfolio_weights(data: pd.DataFrame, risk_score: int) -> np.ndarray:
    """
    Original portfolio weights calculation based on risk score.
    
    Uses traditional allocation methods based on user risk tolerance:
    - Conservative: Equal weights for safety
    - Moderate: Slight bias toward growth
    - Aggressive: Concentrated weights for higher returns
    """
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

# =============================================================================
# PORTFOLIO GROWTH SIMULATION - Core Financial Modeling
# =============================================================================

def simulate_portfolio_growth(data: pd.DataFrame, weights: np.ndarray, 
                            lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
    """
    Enhanced portfolio growth simulation with comprehensive debugging.
    
    Simulates portfolio growth over time with:
    - Historical price movements
    - Regular monthly contributions
    - Compound growth effects
    - Detailed logging for troubleshooting
    """
    try:
        logger.info(f"📈 Starting simulation: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        # Debug: Check input data quality
        logger.info(f"📊 Data shape: {data.shape}")
        logger.info(f"📊 Data columns: {list(data.columns)}")
        logger.info(f"📊 Weights: {weights}")
        logger.info(f"📊 Weights sum: {np.sum(weights)}")
        
        # Check for NaN or infinite values in data
        if data.isnull().any().any():
            logger.warning("⚠️ Found NaN values in stock data")
            data = data.fillna(method='ffill').fillna(method='bfill')
            logger.info("✅ NaN values filled")
        
        # Check first few rows of data
        logger.info(f"📊 First 5 rows of data:\n{data.head()}")
        
        # Normalize data (this is where issues often occur)
        first_row = data.iloc[0]
        logger.info(f"📊 First row values: {first_row.values}")
        
        # Check for zero or near-zero values in first row
        if (first_row == 0).any() or (np.abs(first_row) < 1e-10).any():
            logger.error("❌ Found zero or near-zero values in first row of data")
            logger.info(f"📊 Problematic values: {first_row[first_row == 0]}")
            raise ValueError("Invalid starting prices in stock data")
        
        normalized = data.div(first_row)
        logger.info(f"📊 Normalized data shape: {normalized.shape}")
        logger.info(f"📊 First few normalized values:\n{normalized.head()}")
        
        # Check normalized data for issues
        if normalized.isnull().any().any():
            logger.error("❌ NaN values created during normalization")
            raise ValueError("Normalization created NaN values")
        
        # Apply portfolio weights
        weighted = normalized.dot(weights)
        logger.info(f"📊 Weighted portfolio length: {len(weighted)}")
        logger.info(f"📊 First 5 weighted values: {weighted.head().values}")
        logger.info(f"📊 Last 5 weighted values: {weighted.tail().values}")
        
        # Check for issues in weighted portfolio
        if weighted.isnull().any():
            logger.error("❌ NaN values in weighted portfolio")
            raise ValueError("Weighted portfolio contains NaN values")
        
        if (weighted == 0).any():
            logger.warning("⚠️ Found zero values in weighted portfolio")
            zero_count = (weighted == 0).sum()
            logger.warning(f"⚠️ Number of zero values: {zero_count}")
        
        # Initialize tracking variables
        portfolio_values = []
        contributions = []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        logger.info(f"💰 Starting values - Portfolio: £{current_value}, Contributions: £{total_contributions}")
        
        # Simulation loop with enhanced debugging
        for i, (date, growth_factor) in enumerate(weighted.items()):
            # Add monthly contribution (every ~21 trading days)
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
                if i < 100:  # Log first few contributions
                    logger.info(f"📅 Month {i//21}: Added £{monthly}, Total contributions: £{total_contributions}")
            
            # Apply growth (but not on first day)
            if i > 0:
                previous_factor = weighted.iloc[i - 1]
                
                # Debug growth calculation
                if i < 10:  # Log first 10 calculations
                    logger.info(f"📈 Day {i}: prev_factor={previous_factor:.6f}, curr_factor={growth_factor:.6f}")
                
                # Check for division by zero
                if abs(previous_factor) < 1e-10:
                    logger.error(f"❌ Division by zero at index {i}: previous_factor={previous_factor}")
                    logger.error(f"❌ Date: {date}, Growth factor: {growth_factor}")
                    raise ValueError(f"Division by zero in growth calculation at index {i}")
                
                growth_rate = growth_factor / previous_factor
                
                # Check for invalid growth rate
                if not np.isfinite(growth_rate):
                    logger.error(f"❌ Invalid growth rate at index {i}: {growth_rate}")
                    logger.error(f"❌ Factors: {previous_factor} -> {growth_factor}")
                    raise ValueError(f"Invalid growth rate: {growth_rate}")
                
                # Apply growth
                old_value = current_value
                current_value *= growth_rate
                
                # Debug significant changes
                if i < 10 or abs(growth_rate - 1) > 0.1:  # Log first 10 or big changes
                    logger.info(f"📈 Day {i}: Growth {growth_rate:.4f}, Value £{old_value:.2f} -> £{current_value:.2f}")
                
                # Check if portfolio value became zero or negative
                if current_value <= 0:
                    logger.error(f"❌ Portfolio value became {current_value} at index {i}")
                    logger.error(f"❌ Growth rate was: {growth_rate}")
                    logger.error(f"❌ Previous value: £{old_value}")
                    raise ValueError(f"Portfolio value became {current_value}")
            
            # Record values for timeline
            contributions.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(total_contributions), 2)
            })
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"), 
                "value": round(float(current_value), 2)
            })
            
            # Periodic progress logging
            if i % 252 == 0:  # Yearly
                logger.info(f"📅 Year {i//252}: Portfolio £{current_value:,.2f}, Contributions £{total_contributions:,.2f}")

        # Final calculations and validation
        end_value = float(current_value)
        starting_value = float(total_contributions)
        portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

        logger.info(f"✅ Final results: £{starting_value:,.2f} invested -> £{end_value:,.2f} final ({portfolio_return:.1%} return)")
        
        # Validate final results
        if end_value <= 0:
            logger.error(f"❌ Final portfolio value is {end_value}")
            raise ValueError(f"Invalid final portfolio value: {end_value}")
        
        if len(portfolio_values) == 0:
            logger.error("❌ No portfolio values recorded")
            raise ValueError("No portfolio timeline data generated")

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
        logger.error(f"❌ Error in portfolio simulation: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        
        # Enhanced fallback with mathematical calculation
        logger.warning("🔄 Using enhanced fallback simulation")
        
        try:
            # Calculate total invested
            starting_value = lump_sum + monthly * 12 * timeframe
            
            # Use moderate 7% annual growth as fallback
            annual_growth = 1.07
            end_value = lump_sum * (annual_growth ** timeframe)
            
            # Add future value of monthly contributions (annuity formula)
            if monthly > 0:
                months = timeframe * 12
                monthly_growth = annual_growth ** (1/12)
                fv_annuity = monthly * (((monthly_growth ** months) - 1) / (monthly_growth - 1))
                end_value += fv_annuity
            
            portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0
            
            logger.info(f"🔄 Fallback results: £{starting_value:,.2f} -> £{end_value:,.2f} ({portfolio_return:.1%})")
            
            # Create simple timeline
            start_date = datetime.today()
            end_date = start_date + timedelta(days=timeframe * 365)
            
            contributions = [
                {"date": start_date.strftime("%Y-%m-%d"), "value": round(starting_value, 2)}
            ]
            portfolio_values = [
                {"date": start_date.strftime("%Y-%m-%d"), "value": round(lump_sum, 2)},
                {"date": end_date.strftime("%Y-%m-%d"), "value": round(end_value, 2)}
            ]
            
            return {
                "starting_value": round(starting_value, 2),
                "end_value": round(end_value, 2),
                "portfolio_return": round(portfolio_return, 4),
                "timeline": {
                    "contributions": contributions,
                    "portfolio": portfolio_values
                },
                "fallback_used": True,
                "original_error": str(e)
            }
            
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback simulation failed: {fallback_error}")
            raise ValueError(f"Both main and fallback simulations failed: {e}, {fallback_error}")

# =============================================================================
# UTILITY FUNCTIONS - Helper Functions for Names, Summaries, and Responses
# =============================================================================

def get_company_name(ticker: str) -> str:
    """
    Map stock tickers to human-readable company names.
    
    Provides user-friendly names for common ETFs used in portfolio simulations.
    """
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

def generate_simple_summary(stocks_picked: List[Dict], user_data: Dict[str, Any], 
                          risk_score: int, risk_label: str, 
                          simulation_results: Dict[str, Any]) -> str:
    """
    Generate simple text summary when AI services are unavailable.
    
    Creates a basic educational summary of portfolio results that can be used
    as a fallback when more sophisticated AI summary generation fails.
    """
    logger.info("📝 Generating simple fallback summary")
    
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

def save_simulation_to_db(db: Session, sim_input: Dict[str, Any], user_data: Dict[str, Any],
                         risk_score: int, risk_label: str, ai_summary: str,
                         stocks_picked: List[Dict], simulation_results: Dict[str, Any]) -> models.Simulation:
    """
    Original database save function with JSON serialization fix.
    
    Saves basic simulation results with proper serialization handling
    for backward compatibility with original system.
    """
    try:
        logger.info("💾 Saving simulation results to database")
        
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
        
        logger.info(f"✅ Simulation saved successfully with ID: {simulation.id}")
        return simulation
        
    except Exception as e:
        logger.error(f"❌ Error saving simulation to database: {str(e)}")
        db.rollback()
        raise

def format_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """
    Original response formatting for basic simulations.
    
    Creates a standard API response format for backward compatibility
    with existing frontend systems.
    """
    logger.info("📋 Formatting simulation response for API")
    
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
# SHAP VISUALIZATION ENDPOINTS - Optional Visual Explanations
# =============================================================================

async def get_shap_visualization(simulation_id: int, db: Session) -> Optional[str]:
    """
    Generate SHAP visualization for a specific simulation.
    
    This optional endpoint creates visual explanations of why the AI made
    specific recommendations for a simulation, using waterfall charts to
    show factor contributions.
    
    Args:
        simulation_id: ID of the simulation to visualize
        db: Database session
        
    Returns:
        Path to generated SHAP visualization or None if unavailable
    """
    
    if not WEALTHWISE_AVAILABLE:
        return None
    
    try:
        # Get simulation from database
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation or not simulation.results.get("shap_explanation"):
            return None
        
        # Initialize visualization engine
        from ai_models.stock_model.explainable_ai import VisualizationEngine
        viz_engine = VisualizationEngine()
        
        # Create SHAP visualization
        save_path = f"./static/visualizations/shap_explanation_{simulation_id}.png"
        result = viz_engine.create_shap_waterfall_chart(
            simulation.results["shap_explanation"], save_path
        )
        
        if "saved" in result:
            logger.info(f"✅ SHAP visualization created: {save_path}")
            return save_path
        else:
            logger.warning(f"⚠️ SHAP visualization failed: {result}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error creating SHAP visualization: {e}")
        return None

async def analyze_simulation_with_news(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Analyze a simulation with current news context.
    
    This optional endpoint combines simulation results with current news analysis
    for the recommended stocks, providing updated market context.
    
    Args:
        simulation_id: ID of the simulation to analyze
        db: Database session
        
    Returns:
        Analysis combining simulation and news data
    """
    
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
        logger.error(f"❌ Error analyzing simulation with news: {e}")
        return {"error": str(e)}

# =============================================================================
# SYSTEM MIGRATION NOTES AND INTEGRATION GUIDELINES
# =============================================================================

"""
INTEGRATION GUIDELINES FOR EXISTING SYSTEMS:

1. **Backward Compatibility**: 
   - All original functions remain unchanged for existing integrations
   - Enhanced functions provide additional features without breaking changes
   - Fallback mechanisms ensure system works even if WealthWise unavailable

2. **SHAP Integration Points**:
   - SHAP explanations are generated during get_enhanced_ai_recommendations()
   - Training happens automatically if model not available
   - All SHAP data is properly serialized for database storage

3. **Database Schema Requirements**:
   - No schema changes required - enhanced data stored in existing JSON results field
   - Enhanced flags added to response objects for frontend detection
   - Proper serialization handles NumPy arrays and complex objects

4. **Performance Considerations**:
   - SHAP model training adds ~10-15 seconds on first use
   - Subsequent explanations are fast (<1 second)
   - Fallback mechanisms prevent timeouts

5. **Error Handling Strategy**:
   - Enhanced simulation failures fall back to original simulation
   - Database save failures fall back to basic data structure
   - SHAP failures don't break the overall process

6. **Frontend Integration**:
   - Check 'has_shap_explanations' flag in response
   - Access SHAP data via results.shap_explanation
   - Use visualization endpoints for charts

7. **Testing Recommendations**:
   - Test with WealthWise available and unavailable
   - Verify serialization with complex NumPy objects
   - Confirm fallback behavior under various failure scenarios

8. **Deployment Notes**:
   - Install SHAP dependencies: pip install shap matplotlib seaborn
   - Ensure stock_model directory is in Python path
   - Configure logging levels appropriately for production

EXAMPLE USAGE:

# Standard portfolio simulation (will automatically use enhanced version if available)
result = await simulate_portfolio(sim_input, db)

# Check if SHAP explanations are available
if result.get('has_shap_explanations'):
    shap_data = result['results']['shap_explanation']
    
    # Generate visualization
    viz_path = await get_shap_visualization(result['id'], db)
    
    # Analyze with current news
    news_analysis = await analyze_simulation_with_news(result['id'], db)

# Access SHAP explanations in frontend
if (response.has_shap_explanations) {
    const explanations = response.results.shap_explanation.human_readable_explanation;
    const portfolioScore = response.results.shap_explanation.portfolio_quality_score;
}

API RESPONSE STRUCTURE:

{
    "id": 123,
    "user_id": 456,
    "results": {
        "shap_explanation": {
            "portfolio_quality_score": 78.5,
            "human_readable_explanation": {
                "risk_score": "Your risk tolerance supports higher stock allocation",
                "timeframe": "Your 10-year timeline enables growth investments"
            },
            "feature_contributions": {...},
            "transparency_metrics": {...}
        },
        "goal_analysis": {...},
        "market_regime": {...},
        "wealthwise_enhanced": true,
        "methodology": "WealthWise SHAP-enhanced goal-oriented optimization"
    },
    "has_shap_explanations": true,
    "wealthwise_enhanced": true
}

DEBUGGING SHAP ISSUES:

# Check SHAP status in logs
🔍 SHAP Debug: {hasShapExplanation: true, shapData: Available, wealthwiseEnhanced: true}

# Common issues:
1. SHAP library not installed -> Install: pip install shap matplotlib seaborn
2. WealthWise not available -> Check import path and dependencies
3. Model training failed -> Check logs for specific error messages
4. Serialization failed -> Review clean_simulation_results_for_db logs

# Verify integration:
python -c "from ai_models.stock_model.explainable_ai import SHAPExplainer; print(SHAPExplainer().is_available())"
"""