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
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from database import models
import yfinance as yf
import numpy as np
import pandas as pd
import logging

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)

# =============================================================================
# WEALTHWISE INTEGRATION - Import the new system
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
# ENHANCED MAIN PORTFOLIO SIMULATION FUNCTION
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Enhanced portfolio simulation with WealthWise SHAP integration.
    
    This function now includes:
    1. Goal-oriented portfolio optimization
    2. SHAP explainable AI explanations
    3. Market regime detection
    4. Multi-factor analysis
    5. Enhanced AI educational summaries
    
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
        db.rollback()
        
        # Fallback to original simulation if enhanced version fails
        logger.warning("🔄 Falling back to original simulation method")
        return await simulate_portfolio_fallback(sim_input, db)

# =============================================================================
# ENHANCED AI RECOMMENDATION FUNCTIONS
# =============================================================================

async def get_enhanced_ai_recommendations(
    target_value: float, timeframe: int, risk_score: int, risk_label: str,
    current_investment: float = 0, monthly_contribution: float = 0
) -> Dict[str, Any]:
    """
    Get enhanced AI recommendations with SHAP explanations and goal analysis.
    
    This function provides:
    1. Goal-oriented stock recommendations
    2. SHAP explainable AI reasoning
    3. Feasibility assessment
    4. Market regime analysis
    5. Transparent decision explanations
    
    Returns:
        Dict containing stocks, explanations, and analysis
    """
    
    if not WEALTHWISE_AVAILABLE:
        logger.warning("⚠️ WealthWise not available, using fallback recommendations")
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback"
        }
    
    try:
        logger.info("🎯 Initializing WealthWise enhanced recommendation system")
        
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
        
        # Step 4: Get goal-oriented stock recommendations
        logger.info("🤖 Generating goal-oriented stock recommendations")
        recommended_stocks = recommender.recommend_stocks(
            target_value, timeframe, risk_score, 
            current_investment, monthly_contribution
        )
        
        # Step 5: Generate SHAP explanations
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
        
        logger.info(f"✅ Enhanced recommendations complete: {len(recommended_stocks)} stocks selected")
        
        return {
            "stocks": recommended_stocks,
            "shap_explanation": shap_explanation,
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
            "error": str(e)
        }

def get_stock_explanation(ticker: str, recommendation_result: Dict[str, Any]) -> str:
    """
    Get explanation for why a specific stock was recommended.
    
    This uses the SHAP explanations and factor analysis to explain
    each stock selection in human-readable terms.
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
                if len(explanation) > 20:  # Non-empty explanation
                    return f"{ticker}: {explanation[:100]}..."
    
    # Fallback explanation
    return f"{ticker} recommended by AI analysis for your goals and risk profile"

# =============================================================================
# ENHANCED PORTFOLIO OPTIMIZATION
# =============================================================================

def calculate_enhanced_portfolio_weights(
    data: pd.DataFrame, risk_score: int, recommendation_result: Dict[str, Any]
) -> np.ndarray:
    """
    Calculate enhanced portfolio weights using WealthWise optimization.
    
    If WealthWise is available, use its correlation-based optimization.
    Otherwise, fall back to the original method.
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
# ENHANCED AI SUMMARY GENERATION
# =============================================================================

async def generate_enhanced_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> str:
    """
    Generate enhanced AI summary with SHAP explanations and goal analysis.
    
    This creates a comprehensive educational summary that includes:
    1. Portfolio performance results
    2. SHAP explanations for why stocks were chosen
    3. Goal feasibility analysis
    4. Market regime context
    5. Educational insights
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
    Generate AI summary with SHAP explanations integrated.
    
    This creates a prompt that includes SHAP explanations and uses
    the AI service to generate educational content.
    """
    
    # Extract context
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
    Generate enhanced simple summary with available SHAP context.
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
# ENHANCED DATABASE FUNCTIONS
# =============================================================================

def save_enhanced_simulation_to_db(
    db: Session, sim_input: Dict[str, Any], user_data: Dict[str, Any],
    risk_score: int, risk_label: str, ai_summary: str,
    stocks_picked: List[Dict], simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> models.Simulation:
    """
    Save enhanced simulation with SHAP explanations and goal analysis.
    """
    
    try:
        logger.info("💾 Saving enhanced simulation with SHAP data to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        # Create enhanced results object
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
            "methodology": "WealthWise SHAP-enhanced goal-oriented optimization"
        }
        
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
            results=enhanced_results
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        logger.info(f"✅ Enhanced simulation saved with SHAP data (ID: {simulation.id})")
        return simulation
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced simulation: {str(e)}")
        db.rollback()
        raise

def format_enhanced_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """
    Format enhanced simulation response with SHAP explanations.
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
        # Enhanced flags
        "wealthwise_enhanced": simulation.results.get("wealthwise_enhanced", False),
        "has_shap_explanations": bool(simulation.results.get("shap_explanation")),
        "methodology": simulation.results.get("methodology", "Standard simulation")
    }
    
    return response

# =============================================================================
# FALLBACK FUNCTIONS (unchanged from original)
# =============================================================================

async def simulate_portfolio_fallback(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Fallback to original simulation when enhanced version fails.
    This ensures the system always works even if WealthWise is unavailable.
    """
    
    logger.info("🔄 Running fallback simulation")
    
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
    
    # Continue with original simulation...
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

# Keep all original functions for backward compatibility
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
    """Original stock data download function - unchanged"""
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
    """Original portfolio weights calculation - unchanged"""
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
    """Original portfolio growth simulation - unchanged"""
    try:
        logger.info(f"📈 Simulating growth: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        normalized = data.div(data.iloc[0])
        weighted = normalized.dot(weights)
        
        portfolio_values = []
        contributions = []
        current_value = lump_sum
        total_contributions = lump_sum

        for i, (date, growth_factor) in enumerate(weighted.items()):
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
            
            if i > 0:
                growth_rate = growth_factor / weighted.iloc[i - 1]
                current_value *= growth_rate

            contributions.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(total_contributions), 2)
            })
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(current_value), 2)
            })

        end_value = float(current_value)
        starting_value = float(total_contributions)
        portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

        logger.info(f"💰 Simulation results: £{starting_value:,.2f} → £{end_value:,.2f} ({portfolio_return:.1%} return)")

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
        
        logger.info("🔄 Using fallback simulation with 7% annual growth")
        starting_value = lump_sum + monthly * 12 * timeframe
        end_value = starting_value * (1.07 ** timeframe)
        
        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": 0.07 * timeframe,
            "timeline": {
                "contributions": [{"date": datetime.today().strftime("%Y-%m-%d"), "value": starting_value}],
                "portfolio": [{"date": datetime.today().strftime("%Y-%m-%d"), "value": end_value}]
            }
        }

def generate_simple_summary(stocks_picked: List[Dict], user_data: Dict[str, Any], 
                          risk_score: int, risk_label: str, 
                          simulation_results: Dict[str, Any]) -> str:
    """Original simple summary generation - unchanged"""
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

def get_company_name(ticker: str) -> str:
    """Original company name mapping - unchanged"""
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
    """Original database save function - unchanged"""
    try:
        logger.info("💾 Saving simulation results to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
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
            results={
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
    """Original response formatting - unchanged"""
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
# NEW API ENDPOINTS FOR ENHANCED FEATURES
# =============================================================================

async def get_shap_visualization(simulation_id: int, db: Session) -> Optional[str]:
    """
    Generate SHAP visualization for a specific simulation.
    
    This endpoint allows you to create visual explanations of why
    the AI made specific recommendations for a simulation.
    
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
    
    This combines the simulation results with current news analysis
    for the recommended stocks.
    
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
# MIGRATION NOTES FOR YOUR EXISTING SYSTEM
# =============================================================================

"""
INTEGRATION STEPS:

1. REPLACE YOUR EXISTING portfolio_simulator.py WITH THIS FILE

2. UPDATE YOUR API ROUTES (api/routers/ai_analysis.py):
   Add these new endpoints:

   @router.get("/simulation/{simulation_id}/shap-visualization")
   async def get_simulation_shap_viz(simulation_id: int, db: Session = Depends(get_db)):
       viz_path = await get_shap_visualization(simulation_id, db)
       if viz_path:
           return {"visualization_path": viz_path}
       else:
           raise HTTPException(status_code=404, detail="SHAP visualization not available")

   @router.get("/simulation/{simulation_id}/news-analysis")
   async def get_simulation_news_analysis(simulation_id: int, db: Session = Depends(get_db)):
       analysis = await analyze_simulation_with_news(simulation_id, db)
       return analysis

3. BENEFITS YOU'LL GET:
   - ✅ Goal-oriented portfolio optimization (matches user's specific target)
   - ✅ SHAP explainable AI (transparent reasoning for every recommendation)
   - ✅ Market regime detection (adapts to current market conditions)
   - ✅ Multi-factor analysis (professional-grade stock evaluation)
   - ✅ Enhanced educational summaries (explains WHY stocks were chosen)
   - ✅ Backward compatibility (falls back to original system if WealthWise fails)
   - ✅ Database integration (stores SHAP explanations for future analysis)

4. WHAT CHANGES FOR YOUR USERS:
   - Same API interface - no frontend changes needed
   - Enhanced portfolio recommendations based on their specific goals
   - Detailed explanations of why each stock was recommended
   - Better educational content in AI summaries
   - Optional new endpoints for SHAP visualizations and news analysis

5. FALLBACK PROTECTION:
   - If WealthWise fails to load, system automatically uses original logic
   - All existing functionality preserved
   - Graceful degradation ensures system reliability

6. TESTING:
   - Test with WEALTHWISE_AVAILABLE = False to ensure fallback works
   - Test with WEALTHWISE_AVAILABLE = True to see enhanced features
   - Monitor logs for any integration issues

NEXT STEPS:
1. Install the WealthWise package in your project directory
2. Replace portfolio_simulator.py with this enhanced version
3. Test the enhanced simulation endpoint
4. Add the new API routes for SHAP visualizations
5. Monitor performance and user feedback
"""