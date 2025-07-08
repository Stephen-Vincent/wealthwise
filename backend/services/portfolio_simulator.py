"""
Portfolio Simulator Service

This module handles the complete portfolio simulation workflow:
1. Extracts and validates user investment preferences
2. Uses AI to recommend appropriate stocks based on risk profile
3. Downloads historical market data for simulation
4. Calculates portfolio weights and simulates growth over time
5. Generates AI-powered educational summaries
6. Saves results to database

The service integrates with:
- Enhanced Stock Recommender AI (for stock selection)
- AI Analysis Service (for educational summaries)
- Yahoo Finance API (for historical data)
- Database models (for persistence)
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from database import models
import yfinance as yf
import numpy as np
import pandas as pd
import logging

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN PORTFOLIO SIMULATION FUNCTION
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Main portfolio simulation function that orchestrates the entire process.
    
    This function:
    1. Validates and extracts user investment data
    2. Gets AI-recommended stocks based on risk profile
    3. Downloads historical market data
    4. Simulates portfolio growth over the specified timeframe
    5. Generates educational AI summary
    6. Saves results to database
    
    Args:
        sim_input: Dictionary containing user onboarding data including:
                  - goal, target_value, lump_sum, monthly, timeframe
                  - risk_score, risk_label (from risk assessment)
                  - user demographics and preferences
        db: Database session for saving results
    
    Returns:
        Dict containing complete simulation results including:
        - Portfolio performance metrics
        - Stock allocations
        - AI educational summary
        - Timeline data for charts
    """
    
    try:
        logger.info("🚀 Starting portfolio simulation")
        
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

        # STEP 2: Get AI-powered stock recommendations based on user profile
        logger.info("🤖 Getting AI stock recommendations")
        tickers = get_ai_stock_recommendations(
            target_value=user_data["target_value"],
            timeframe=user_data["timeframe"],
            risk_score=risk_score,
            risk_label=risk_label
        )
        logger.info(f"📈 AI recommended stocks: {tickers}")

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
        
        # STEP 5: Calculate optimal portfolio weights based on risk profile
        logger.info("⚖️ Calculating portfolio allocation weights")
        weights = calculate_portfolio_weights(stock_data, risk_score)
        
        # STEP 6: Create structured list of selected stocks with allocations
        logger.info("📋 Creating final stock allocation list")
        stocks_picked = [
            {
                "symbol": ticker, 
                "name": get_company_name(ticker),
                "allocation": round(float(weight), 4)
            }
            for ticker, weight in zip(tickers, weights)
        ]
        
        logger.info("💼 Final portfolio allocation:")
        for stock in stocks_picked:
            logger.info(f"   {stock['symbol']}: {stock['allocation']*100:.1f}% ({stock['name']})")

        # STEP 7: Run the portfolio growth simulation
        logger.info("📈 Running portfolio growth simulation")
        simulation_results = simulate_portfolio_growth(
            stock_data, weights, lump_sum, monthly, timeframe
        )
        
        # STEP 8: Generate AI-powered educational summary
        logger.info("🧠 Generating AI educational summary")
        ai_summary = await generate_ai_enhanced_summary(
            stocks_picked, user_data, risk_score, risk_label, simulation_results
        )

        # STEP 9: Save complete simulation results to database
        logger.info("💾 Saving simulation to database")
        simulation = save_simulation_to_db(
            db=db,
            sim_input=sim_input,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            ai_summary=ai_summary,
            stocks_picked=stocks_picked,
            simulation_results=simulation_results
        )

        logger.info(f"✅ Portfolio simulation completed successfully (ID: {simulation.id})")
        return format_simulation_response(simulation)

    except Exception as e:
        logger.error(f"❌ Portfolio simulation failed: {str(e)}")
        db.rollback()
        raise ValueError(f"Portfolio simulation failed: {str(e)}")

# =============================================================================
# AI STOCK RECOMMENDATION FUNCTIONS
# =============================================================================

def get_ai_stock_recommendations(target_value: float, timeframe: int, 
                               risk_score: int, risk_label: str) -> List[str]:
    """
    Get AI-powered stock recommendations using the Enhanced Stock Recommender.
    
    This function:
    1. Attempts to use the sophisticated AI stock model
    2. Considers user's target value, timeframe, and risk tolerance
    3. Falls back to rule-based selection if AI model fails
    4. Ensures reliable stock recommendations in all scenarios
    
    Args:
        target_value: User's investment target in GBP
        timeframe: Investment period in years
        risk_score: Risk tolerance score (0-100)
        risk_label: Human-readable risk level
    
    Returns:
        List of stock ticker symbols recommended for the user
    """
    try:
        # Import the correct AI stock recommendation model
        from backend.ai_models.stock_model.enhanced_stock_recommender import train_and_recommend, save_last_input_features
        
        logger.info("🤖 Using AI Stock Recommender from train_stock_model")
        
        # Get AI-powered stock recommendations based on user profile
        tickers = train_and_recommend(
            target_value=float(target_value),
            timeframe=int(timeframe),
            risk_score=float(risk_score)
        )
        
        # Save the input parameters for audit trail and model improvement
        save_last_input_features(
            target_value=float(target_value),
            timeframe=int(timeframe),
            risk_score=float(risk_score)
        )
        
        logger.info(f"✅ AI stock model recommended {len(tickers)} stocks: {tickers}")
        
        # Validate that we received valid recommendations
        if not tickers or len(tickers) == 0:
            logger.warning("⚠️ AI model returned empty list, using fallback method")
            return get_fallback_stocks_by_risk_profile(risk_score, risk_label)
        
        return tickers
        
    except ImportError as e:
        logger.error(f"❌ Failed to import AI Stock Recommender: {str(e)}")
        logger.warning("🔄 Falling back to rule-based stock selection")
        return get_fallback_stocks_by_risk_profile(risk_score, risk_label)
    
    except Exception as e:
        logger.error(f"❌ AI stock model failed: {str(e)}")
        logger.warning("🔄 Falling back to rule-based stock selection")
        return get_fallback_stocks_by_risk_profile(risk_score, risk_label)


def get_fallback_stocks_by_risk_profile(risk_score: int, risk_label: str) -> List[str]:
    """
    Fallback stock selection based on simple risk-based rules.
    
    This function provides reliable stock recommendations when the AI model fails.
    Uses well-established ETFs appropriate for different risk levels.
    
    Args:
        risk_score: Risk tolerance score (0-100)
        risk_label: Human-readable risk level (for logging)
    
    Returns:
        List of ETF ticker symbols appropriate for the risk level
    """
    logger.info(f"📊 Using fallback selection for {risk_label} risk profile (score: {risk_score})")
    
    if risk_score < 35:  # Conservative portfolio
        return ["VTI", "BND", "VEA", "VTEB", "VWO"]  # Index funds, bonds, international
    elif risk_score < 70:  # Moderate portfolio
        return ["VTI", "VEA", "VWO", "VNQ", "BND"]   # Mix of stocks and bonds
    else:  # Aggressive portfolio
        return ["VTI", "VGT", "VUG", "ARKK", "VEA"]  # Growth stocks, tech focus

# =============================================================================
# MARKET DATA FUNCTIONS
# =============================================================================

def download_stock_data(tickers: List[str], timeframe: int) -> pd.DataFrame:
    """
    Download historical stock price data for portfolio simulation.
    
    This function:
    1. Calculates appropriate date range for the simulation
    2. Downloads closing price data from Yahoo Finance
    3. Cleans and validates the data
    4. Ensures sufficient data quality for simulation
    
    Args:
        tickers: List of stock ticker symbols
        timeframe: Investment period in years
    
    Returns:
        DataFrame with cleaned historical closing prices
        
    Raises:
        ValueError: If no valid stock data is available
    """
    try:
        # Calculate date range - ensure we have enough historical data
        days_needed = max(timeframe * 365, 365)  # At least 1 year minimum
        start_date = (datetime.today() - timedelta(days=days_needed)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')

        logger.info(f"📅 Downloading data from {start_date} to {end_date} for {len(tickers)} stocks")

        # Download closing price data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        
        # Handle single stock case (converts Series to DataFrame)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # Data quality control - remove stocks with insufficient data
        threshold = len(data) * 0.7  # Require at least 70% of trading days
        data = data.dropna(axis=1, thresh=threshold)
        
        logger.info(f"📊 Downloaded data shape: {data.shape}")
        logger.info(f"✅ Data quality check: {len(data.columns)} stocks with sufficient data")
        
        if data.empty:
            raise ValueError("No valid stock data available after quality filtering")
            
        return data
        
    except Exception as e:
        logger.error(f"❌ Error downloading stock data: {str(e)}")
        raise ValueError(f"Failed to download stock data: {str(e)}")

# =============================================================================
# PORTFOLIO OPTIMIZATION FUNCTIONS
# =============================================================================

def calculate_portfolio_weights(data: pd.DataFrame, risk_score: int) -> np.ndarray:
    """
    Calculate portfolio allocation weights based on user's risk tolerance.
    
    This function determines how much of the portfolio to allocate to each stock
    based on the user's risk profile. Higher risk tolerance leads to more
    concentrated positions in growth assets.
    
    Args:
        data: DataFrame with historical stock prices
        risk_score: User's risk tolerance (0-100)
    
    Returns:
        Array of weights that sum to 1.0, representing portfolio allocation
    """
    num_assets = len(data.columns)
    logger.info(f"⚖️ Calculating weights for {num_assets} assets (risk score: {risk_score})")
    
    if risk_score < 35:  # Conservative approach
        # Equal weighting for maximum diversification
        weights = np.array([1 / num_assets] * num_assets)
        logger.info("📊 Using equal weights (conservative approach)")
        
    elif risk_score < 70:  # Moderate approach
        # Slight bias toward first assets (typically larger, more stable)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:num_assets])
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        logger.info("📊 Using moderate bias weighting")
        
    else:  # Aggressive approach
        # More concentrated positions in top holdings
        weights = np.array([0.4, 0.3, 0.2, 0.1][:num_assets])
        
        # Handle cases with more than 4 assets
        if len(weights) < num_assets:
            remaining = num_assets - len(weights)
            additional_weights = np.array([0.05] * remaining)
            weights = np.concatenate([weights, additional_weights])
        
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        logger.info("📊 Using concentrated weighting (aggressive approach)")
    
    return weights

# =============================================================================
# PORTFOLIO SIMULATION FUNCTIONS
# =============================================================================

def simulate_portfolio_growth(data: pd.DataFrame, weights: np.ndarray, 
                            lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
    """
    Simulate portfolio growth over time using historical data.
    
    This function:
    1. Normalizes historical price data to starting values
    2. Applies portfolio weights to create a blended performance
    3. Simulates monthly contributions over the investment period
    4. Calculates final portfolio value and returns
    5. Creates timeline data for charting
    
    Args:
        data: Historical price data
        weights: Portfolio allocation weights
        lump_sum: Initial investment amount
        monthly: Monthly contribution amount
        timeframe: Investment period in years
    
    Returns:
        Dictionary containing simulation results including timeline data
    """
    try:
        logger.info(f"📈 Simulating growth: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        # STEP 1: Normalize price data to starting values (all start at 1.0)
        normalized = data.div(data.iloc[0])
        
        # STEP 2: Apply portfolio weights to create blended performance
        weighted = normalized.dot(weights)
        
        # STEP 3: Initialize tracking variables
        portfolio_values = []
        contributions = []
        current_value = lump_sum
        total_contributions = lump_sum

        # STEP 4: Simulate daily performance with monthly contributions
        for i, (date, growth_factor) in enumerate(weighted.items()):
            # Add monthly contribution (approximately every 21 trading days)
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
            
            # Apply market growth/decline
            if i > 0:
                growth_rate = growth_factor / weighted.iloc[i - 1]
                current_value *= growth_rate

            # Record data for timeline charts
            contributions.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(total_contributions), 2)
            })
            portfolio_values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(current_value), 2)
            })

        # STEP 5: Calculate final performance metrics
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
        
        # FALLBACK: Simple growth calculation if simulation fails
        logger.info("🔄 Using fallback simulation with 7% annual growth")
        starting_value = lump_sum + monthly * 12 * timeframe
        end_value = starting_value * (1.07 ** timeframe)  # 7% annual growth
        
        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": 0.07 * timeframe,
            "timeline": {
                "contributions": [{"date": datetime.today().strftime("%Y-%m-%d"), "value": starting_value}],
                "portfolio": [{"date": datetime.today().strftime("%Y-%m-%d"), "value": end_value}]
            }
        }

# =============================================================================
# AI SUMMARY GENERATION FUNCTIONS
# =============================================================================

async def generate_ai_enhanced_summary(stocks_picked: List[Dict], user_data: Dict[str, Any], 
                                     risk_score: int, risk_label: str, 
                                     simulation_results: Dict[str, Any]) -> str:
    """
    Generate an AI-powered educational summary of the portfolio simulation.
    
    This function:
    1. Attempts to use the AI Analysis Service for detailed explanations
    2. Provides educational content about investing principles
    3. Falls back to a simple summary if AI service fails
    4. Ensures users always receive meaningful feedback
    
    Args:
        stocks_picked: List of selected stocks with allocations
        user_data: User investment preferences and goals
        risk_score: Risk tolerance score
        risk_label: Human-readable risk level
        simulation_results: Portfolio performance results
    
    Returns:
        Educational summary explaining the simulation results
    """
    try:
        logger.info("🧠 Generating AI-powered educational summary")
        
        # Import and use the AI Analysis Service for sophisticated summaries
        from services.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        # Generate detailed AI summary with educational content
        ai_summary = await ai_service.generate_portfolio_summary(
            stocks_picked=stocks_picked,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            simulation_results=simulation_results
        )
        
        logger.info("✅ AI summary generated successfully")
        return ai_summary
        
    except Exception as e:
        logger.warning(f"⚠️ AI summary generation failed: {e}. Using fallback summary.")
        # Fall back to simple summary to ensure users get feedback
        return generate_simple_summary(stocks_picked, user_data, risk_score, risk_label, simulation_results)


def generate_simple_summary(stocks_picked: List[Dict], user_data: Dict[str, Any], 
                          risk_score: int, risk_label: str, 
                          simulation_results: Dict[str, Any]) -> str:
    """
    Generate a simple fallback summary when AI service is unavailable.
    
    This ensures users always receive a meaningful explanation of their
    portfolio simulation, even if the advanced AI summary fails.
    
    Args:
        stocks_picked: List of selected stocks with allocations
        user_data: User investment preferences and goals  
        risk_score: Risk tolerance score
        risk_label: Human-readable risk level
        simulation_results: Portfolio performance results
    
    Returns:
        Basic summary of simulation results
    """
    logger.info("📝 Generating simple fallback summary")
    
    # Extract key information for summary
    goal = user_data.get("goal", "wealth building")
    timeframe = user_data.get("timeframe", 10)
    start_value = simulation_results.get("starting_value", 0)
    end_value = simulation_results.get("end_value", 0)
    target_value = user_data.get("target_value", 50000)
    
    # Create readable stock list
    stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
    target_achieved = end_value >= target_value
    
    return f"""
Portfolio simulation completed for your {goal} goal. Your {risk_label.lower()} risk portfolio, 
invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years. 
Your target of £{target_value:,.2f} was {'achieved' if target_achieved else 'not achieved'}. 
This simulation demonstrates how diversified investing can help build wealth over time.
""".strip()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_company_name(ticker: str) -> str:
    """
    Map ticker symbols to human-readable company/fund names.
    
    This provides user-friendly names for the selected investments
    instead of just showing ticker symbols.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Human-readable company or fund name
    """
    name_mapping = {
        # Common ETFs and their full names
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

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def save_simulation_to_db(db: Session, sim_input: Dict[str, Any], user_data: Dict[str, Any],
                         risk_score: int, risk_label: str, ai_summary: str,
                         stocks_picked: List[Dict], simulation_results: Dict[str, Any]) -> models.Simulation:
    """
    Save complete simulation results to the database.
    
    This function:
    1. Creates a new Simulation record with all relevant data
    2. Stores both user inputs and calculated results
    3. Handles database errors gracefully
    4. Returns the saved simulation object
    
    Args:
        db: Database session
        sim_input: Original user input data
        user_data: Processed user investment data
        risk_score: Risk tolerance score
        risk_label: Human-readable risk level
        ai_summary: Generated educational summary
        stocks_picked: Selected stocks with allocations
        simulation_results: Portfolio performance results
    
    Returns:
        Saved Simulation model instance
        
    Raises:
        Exception: If database save fails
    """
    try:
        logger.info("💾 Saving simulation results to database")
        
        # Determine if investment target was achieved
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        # Create comprehensive simulation record
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
        
        # Save to database with error handling
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
    Format the database simulation record for API response.
    
    This function converts the database model into a clean dictionary
    that can be serialized to JSON for the frontend.
    
    Args:
        simulation: Saved simulation database record
    
    Returns:
        Dictionary formatted for API response
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