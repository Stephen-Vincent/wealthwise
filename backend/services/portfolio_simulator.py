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
    Get enhanced AI recommendations with factor analysis, SHAP explanations and goal analysis.
    
    This function provides:
    1. Goal-oriented analysis and feasibility assessment
    2. Market regime-aware recommendations
    3. Factor-based stock selection and ranking
    4. SHAP explainable AI reasoning
    5. Comprehensive decision transparency
    
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
        logger.info("🎯 Initializing WealthWise enhanced recommendation system with factor analysis")
        
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
        candidate_stocks = set(initial_recommendations)  # Start with goal-oriented picks
        
        # Add risk-appropriate candidates
        if risk_score < 35:  # Conservative
            candidate_stocks.update(["VTI", "BND", "VEA", "VTEB", "VWO", "AGG", "VNQ", "SCHD", "VYM"])
        elif risk_score < 70:  # Moderate
            candidate_stocks.update(["VTI", "VEA", "VWO", "VNQ", "BND", "VUG", "VGT", "VOO", "VXUS"])
        else:  # Aggressive
            candidate_stocks.update(["VTI", "VGT", "VUG", "ARKK", "VEA", "QQQ", "ARKQ", "TQQQ", "SOXL"])
        
        # Adjust for market regime
        if market_regime.get('regime') == 'bear':
            # Add defensive stocks in bear markets
            candidate_stocks.update(["VYM", "SCHD", "VDC", "VHT"])
        elif market_regime.get('regime') == 'bull':
            # Add growth stocks in bull markets
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
            
            # Select top stocks based on factor scores
            num_stocks = min(6, len(ranked_stocks))  # Diversify with up to 6 stocks
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

# 🔄 REPLACE your existing generate_enhanced_ai_summary function with this:

async def generate_enhanced_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> str:
    """
    🚀 INTEGRATED AI Summary: Combines SHAP explanations + News Analysis + Market Events
    
    This is the MASTER function that creates comprehensive educational content including:
    1. Portfolio performance results
    2. SHAP explanations for why stocks were chosen  
    3. ⭐ NEWS ANALYSIS and market event correlation ⭐
    4. Goal feasibility analysis
    5. Market regime context
    6. Educational insights with real-world examples
    """
    
    try:
        logger.info("🧠 Generating INTEGRATED AI summary with SHAP + News Analysis")
        
        # Import the enhanced AI Analysis Service
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        # ⭐ STEP 1: Get the full news analysis first (THIS IS THE MISSING PIECE!)
        logger.info("📰 Getting comprehensive news and market analysis...")
        
        # Call our enhanced news analysis function
        portfolio_news_analysis = await ai_service._analyze_portfolio_news_history(
            stocks_picked, user_data, simulation_results
        )
        
        # ⭐ STEP 2: Create the INTEGRATED prompt with BOTH SHAP and News
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
            portfolio_news_analysis=portfolio_news_analysis  # ⭐ NEWS DATA!
        )
        
        # ⭐ STEP 3: Generate the comprehensive summary
        logger.info("🤖 Generating integrated SHAP + News summary...")
        integrated_summary = await ai_service._get_groq_response(integrated_prompt)
        
        logger.info("✅ Integrated AI summary with SHAP + News generated successfully!")
        return ai_service._format_ai_response(integrated_summary)
        
    except Exception as e:
        logger.warning(f"⚠️ Integrated AI summary failed: {e}. Trying fallback methods...")
        
        # Fallback 1: Try just the enhanced news analysis
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
            
            # Fallback 2: Simple SHAP summary (your original logic)
            return generate_simple_enhanced_summary(
                stocks_picked, user_data, risk_score, risk_label, 
                simulation_results, shap_explanation, goal_analysis, feasibility_assessment
            )

# 🆕 ADD this new function (it doesn't exist in your current code):

def create_integrated_shap_news_prompt(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict], goal_analysis: Optional[Dict],
    feasibility_assessment: Optional[Dict], market_regime: Optional[Dict],
    portfolio_news_analysis: Dict[str, Any]
) -> str:
    """
    🎯 Create the ULTIMATE prompt that combines EVERYTHING:
    - SHAP AI explanations
    - News and market events  
    - Goal analysis  
    - Market regime
    - Educational context
    """
    
    # Extract basic info
    goal = user_data.get("goal", "wealth building")
    lump_sum = user_data.get("lump_sum", 0)
    monthly = user_data.get("monthly", 0)
    timeframe = user_data.get("timeframe", 10)
    target_value = user_data.get("target_value", 50000)
    
    end_value = simulation_results.get("end_value", 0)
    total_contributed = lump_sum + (monthly * timeframe * 12)
    target_achieved = end_value >= target_value
    
    # Extract portfolio symbols
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
        
        # Add portfolio quality score if available
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

    # ⭐ Format news analysis - THIS IS THE KEY MISSING PIECE!
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

    # Format historical events
    historical_context = ""
    if 'historical_market_events' in portfolio_news_analysis:
        historical = portfolio_news_analysis['historical_market_events']
        
        corrections = historical.get('major_corrections', [])
        rallies = historical.get('bull_market_periods', [])
        economic_events = historical.get('economic_events', [])
        
        if corrections or rallies or economic_events:
            historical_context = f"""
📊 MAJOR MARKET EVENTS YOUR PORTFOLIO LIVED THROUGH:

Market Corrections Your Portfolio Survived:"""
            
            for correction in corrections[:2]:
                historical_context += f"""
• {correction.get('type', 'Market Event')}: {correction.get('portfolio_impact', 'N/A')} impact
  Caused by: {', '.join(correction.get('likely_news_themes', [])[:2])}"""
            
            historical_context += f"""

Bull Market Rallies That Boosted Your Returns:"""
            for rally in rallies[:2]:
                historical_context += f"""
• {rally.get('type', 'Market Rally')}: {rally.get('portfolio_impact', 'N/A')} gain
  Driven by: {', '.join(rally.get('likely_news_themes', [])[:2])}"""
            
            if economic_events:
                historical_context += f"""

Major Economic Events During Your Investment Period:"""
                for event in economic_events[:2]:
                    historical_context += f"""
• {event.get('event', 'Economic Event')} ({event.get('timeline', 'Timeline')})
  Impact: {event.get('impact', 'Market impact')}"""

    # Create the ultimate comprehensive prompt
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

{historical_context}

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
- Use emojis and formatting for engagement
- Focus on EDUCATION about both AI decision-making and market behavior

This should be comprehensive and detailed - explain both the AI's transparent reasoning AND the real-world market context!
"""

# =============================================================================
# ENHANCED SIMPLE SUMMARY GENERATION
# =============================================================================

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
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def serialize_for_json(data: Any) -> Any:
    """
    Recursively convert NumPy arrays, pandas objects, and other non-serializable objects to JSON-compatible types.
    
    This function handles all the common serialization issues when saving AI model results to databases.
    Specifically designed for WealthWise SHAP explanations and portfolio optimization results.
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
    that need special handling.
    """
    if not shap_data:
        return {}
    
    try:
        cleaned_shap = {}
        
        # Handle common SHAP fields
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
        import copy
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

# Updated save_enhanced_simulation_to_db function
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
    Enhanced version of save_simulation_to_db with proper JSON serialization.
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
        
        # Import the models here to avoid circular imports
        from database import models
        
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
            
            from database import models
            
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

# Example usage in your portfolio_simulator.py:
"""
Replace the existing save_enhanced_simulation_to_db call with:

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
"""

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
    Enhanced version of save_simulation_to_db with proper JSON serialization.
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

# 4. ALSO UPDATE YOUR REGULAR save_simulation_to_db FUNCTION:

def save_simulation_to_db(db, sim_input: Dict[str, Any], user_data: Dict[str, Any],
                         risk_score: int, risk_label: str, ai_summary: str,
                         stocks_picked: List[Dict], simulation_results: Dict[str, Any]):
    """
    Original database save function with JSON serialization fix.
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
        
        # ⭐ Clean the results before saving ⭐
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
            results=cleaned_results  # ⭐ Use cleaned results ⭐
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


def format_enhanced_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """
    Format enhanced simulation response with SHAP explanations properly exposed.
    
    *** FIXED VERSION *** - Ensures SHAP data reaches frontend
    """
    
    # Get the results data
    results = simulation.results or {}
    
    # Extract SHAP data with debugging
    shap_explanation = results.get("shap_explanation")
    has_shap_explanations = bool(shap_explanation)
    
    # Debug logging
    logger.info(f"🔍 Formatting response for simulation {simulation.id}")
    logger.info(f"📊 Results keys: {list(results.keys())}")
    logger.info(f"🔍 SHAP explanation exists: {has_shap_explanations}")
    if shap_explanation:
        logger.info(f"📊 SHAP keys: {list(shap_explanation.keys())}")
        logger.info(f"💯 Portfolio quality score: {shap_explanation.get('portfolio_quality_score', 'N/A')}")
    
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
        "results": results,  # Include full results
        "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat(),
        
        # *** CRITICAL FIX *** - Expose SHAP data at top level for frontend
        "shap_explanation": shap_explanation,  # ⭐ ADD THIS LINE
        "has_shap_explanations": has_shap_explanations,
        "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
        "methodology": results.get("methodology", "Standard simulation"),
        
        # Additional SHAP metadata for debugging
        "shap_debug": {
            "shap_in_results": bool(results.get("shap_explanation")),
            "shap_keys": list(shap_explanation.keys()) if shap_explanation else [],
            "portfolio_quality_score": shap_explanation.get("portfolio_quality_score") if shap_explanation else None,
            "human_readable_available": bool(shap_explanation.get("human_readable_explanation")) if shap_explanation else False
        }
    }
    
    # Final verification
    logger.info(f"✅ Response SHAP status: has_shap={response['has_shap_explanations']}, data_exists={bool(response['shap_explanation'])}")
    
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
    """Enhanced portfolio growth simulation with comprehensive debugging"""
    try:
        logger.info(f"📈 Starting simulation: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        # STEP 1: Debug input data quality
        logger.info(f"📊 Input data shape: {data.shape}")
        logger.info(f"📊 Columns: {list(data.columns)}")
        logger.info(f"⚖️ Weights: {weights}")
        logger.info(f"📊 Weights sum: {weights.sum()}")
        
        # Check for data quality issues
        logger.info(f"📊 First day values:\n{data.iloc[0]}")
        logger.info(f"📊 Any zeros in first day: {(data.iloc[0] == 0).any()}")
        logger.info(f"📊 Any NaN in data: {data.isna().any().any()}")
        logger.info(f"📊 Data date range: {data.index[0]} to {data.index[-1]}")
        
        # STEP 2: Normalize with safety checks
        first_day_values = data.iloc[0]
        
        # Check for zero or negative values that would break normalization
        problematic_stocks = first_day_values[first_day_values <= 0]
        if len(problematic_stocks) > 0:
            logger.error(f"❌ Zero/negative prices on first day: {problematic_stocks}")
            # Replace zeros with small positive value
            first_day_values = first_day_values.replace(0, 0.01)
            logger.warning(f"⚠️ Replaced zero prices with 0.01")
        
        normalized = data.div(first_day_values)
        logger.info(f"📊 Normalized first values: {normalized.iloc[0]}")
        logger.info(f"📊 Any NaN after normalization: {normalized.isna().any().any()}")
        
        # Fill any remaining NaN values
        if normalized.isna().any().any():
            logger.warning("⚠️ Found NaN values after normalization, forward filling...")
            normalized = normalized.fillna(method='ffill').fillna(1.0)
        
        # STEP 3: Calculate weighted portfolio with debugging
        weighted = normalized.dot(weights)
        logger.info(f"📊 Weighted portfolio first 5 values:\n{weighted.head()}")
        logger.info(f"📊 Weighted portfolio last 5 values:\n{weighted.tail()}")
        logger.info(f"📊 Any NaN in weighted data: {weighted.isna().any()}")
        logger.info(f"📊 Any zero in weighted data: {(weighted == 0).any()}")
        
        # STEP 4: Run simulation with enhanced debugging
        portfolio_values = []
        contributions = []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        logger.info(f"📊 Starting simulation with current_value: {current_value}")

        for i, (date, growth_factor) in enumerate(weighted.items()):
            # Add monthly contributions (every ~21 trading days)
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
                if i < 100:  # Log first few months
                    logger.info(f"📅 Month {i//21}: Added £{monthly}, total contributions now £{total_contributions}")
            
            # Apply growth with safety checks
            if i > 0:
                prev_value = weighted.iloc[i - 1]
                
                # Enhanced debugging for growth calculation
                if prev_value <= 0:
                    logger.error(f"❌ Previous weighted value is zero/negative on {date}: {prev_value}")
                    growth_rate = 1.0  # No growth if previous value is invalid
                else:
                    growth_rate = growth_factor / prev_value
                
                # Detect suspicious growth rates
                if growth_rate <= 0:
                    logger.warning(f"⚠️ Zero/negative growth rate on {date}: {growth_rate} (current: {growth_factor}, prev: {prev_value})")
                    growth_rate = 1.0  # Prevent negative/zero growth
                elif growth_rate > 5.0:  # 500% daily growth is suspicious
                    logger.warning(f"⚠️ Extreme growth rate on {date}: {growth_rate} (current: {growth_factor}, prev: {prev_value})")
                    growth_rate = min(growth_rate, 2.0)  # Cap at 200% daily growth
                
                current_value *= growth_rate
                
                # Log extreme portfolio values
                if current_value <= 0:
                    logger.error(f"❌ Portfolio value became zero/negative on {date}: {current_value}")
                    current_value = total_contributions  # Reset to contributions if value goes to zero
                
                # Debug first few days
                if i < 10:
                    logger.info(f"📅 Day {i} ({date}): growth_factor={growth_factor:.4f}, growth_rate={growth_rate:.4f}, portfolio_value=£{current_value:.2f}")

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

        # STEP 5: Calculate final results with validation
        end_value = float(current_value)
        starting_value = float(total_contributions)
        
        # Ensure reasonable results
        if end_value <= 0:
            logger.error(f"❌ Final portfolio value is zero: {end_value}")
            logger.error("🔧 This indicates a fundamental calculation error")
            # Emergency fallback
            end_value = starting_value * (1.05 ** timeframe)  # 5% annual growth fallback
            logger.warning(f"⚠️ Using emergency fallback value: £{end_value:.2f}")
        
        portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

        # Final validation logging
        logger.info(f"💰 FINAL RESULTS:")
        logger.info(f"   Starting value (contributions): £{starting_value:,.2f}")
        logger.info(f"   Ending value (portfolio): £{end_value:,.2f}")
        logger.info(f"   Total return: {portfolio_return:.1%}")
        logger.info(f"   Timeline entries: {len(portfolio_values)}")
        logger.info(f"   Last portfolio entry: {portfolio_values[-1] if portfolio_values else 'None'}")

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
        logger.error(f"📊 Data info: shape={data.shape if 'data' in locals() else 'N/A'}")
        logger.error(f"⚖️ Weights info: {weights if 'weights' in locals() else 'N/A'}")
        
        # Enhanced fallback with better calculations
        logger.warning("🔄 Using enhanced fallback simulation")
        starting_value = lump_sum + monthly * 12 * timeframe
        
        # Use risk-adjusted return estimate
        annual_return = 0.07  # 7% default
        if hasattr(weights, '__len__') and len(weights) > 0:
            # Higher risk portfolio might have higher expected returns
            risk_factor = max(weights) if len(weights) > 0 else 0.5
            annual_return = 0.05 + (risk_factor * 0.05)  # 5-10% range
        
        end_value = starting_value * ((1 + annual_return) ** timeframe)
        
        # Create reasonable timeline
        dates = pd.date_range(start=datetime.today() - timedelta(days=timeframe*365), 
                            end=datetime.today(), freq='D')
        timeline_entries = min(len(dates), 1000)  # Limit entries
        
        contributions_timeline = []
        portfolio_timeline = []
        
        for i in range(0, timeline_entries, max(1, timeline_entries//252)):  # Roughly daily for a year
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
        
        logger.info(f"💰 Fallback results: £{starting_value:,.2f} → £{end_value:,.2f}")
        
        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": round((end_value - starting_value) / starting_value, 4),
            "timeline": {
                "contributions": contributions_timeline,
                "portfolio": portfolio_timeline
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