"""
Portfolio Simulator Service - Enhanced with WealthWise SHAP Integration & React Chart Data

This module handles the complete portfolio simulation workflow:
1. Extracts and validates user investment preferences
2. Uses AI to recommend appropriate stocks based on risk profile
3. Downloads historical market data for simulation
4. Calculates portfolio weights and simulates growth over time
5. Generates AI-powered educational summaries with SHAP explanations
6. Creates interactive visualizations AND provides data for React charts
7. Saves results to database
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
    logger.info("WealthWise SHAP system with VisualizationEngine loaded successfully")
except ImportError as e:
    WEALTHWISE_AVAILABLE = False
    logger.warning(f"WealthWise not available: {e}")

_viz_engine = None

def get_visualization_engine() -> Optional['VisualizationEngine']:
    """Get or initialize the global VisualizationEngine instance."""
    global _viz_engine
    
    if not WEALTHWISE_AVAILABLE:
        return None
    
    if _viz_engine is None:
        try:
            _viz_engine = VisualizationEngine()
            logger.info("VisualizationEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VisualizationEngine: {e}")
            return None
    
    return _viz_engine

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_company_name(ticker: str) -> str:
    """Get company name for a given ticker."""
    name_mapping = {
        "VTI": "Vanguard Total Stock Market ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VTEB": "Vanguard Tax-Exempt Bond ETF",
        "VWO": "Vanguard Emerging Markets Stock ETF",
        "VUG": "Vanguard Growth ETF",
        "VGT": "Vanguard Information Technology ETF",
        "ARKK": "ARK Innovation ETF",
        "QQQ": "Invesco QQQ Trust",
        "VOO": "Vanguard S&P 500 ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "AGG": "iShares Core Aggregate Bond ETF",
        "VYM": "Vanguard High Dividend Yield ETF",
        "SCHD": "Schwab US Dividend Equity ETF"
    }
    return name_mapping.get(ticker, ticker)

def estimate_stock_risk(symbol: str) -> float:
    """Estimate stock risk based on symbol characteristics."""
    risk_mapping = {
        "BND": 5.0, "VTEB": 4.0, "AGG": 5.5, "VYM": 12.0, "SCHD": 14.0,
        "VTI": 16.0, "VOO": 15.0, "VEA": 18.0, "VXUS": 20.0, "VWO": 25.0,
        "VUG": 18.0, "VGT": 22.0, "QQQ": 24.0, "VNQ": 20.0,
        "ARKK": 35.0, "ARKQ": 32.0, "BITO": 45.0, "TQQQ": 50.0, "SOXL": 55.0,
        "FINX": 28.0, "IBB": 30.0, "COIN": 60.0
    }
    return risk_mapping.get(symbol, 20.0)

def estimate_stock_return(symbol: str) -> float:
    """Estimate expected stock return based on symbol characteristics."""
    return_mapping = {
        "BND": 3.0, "VTEB": 2.5, "AGG": 3.5, "VYM": 7.0, "SCHD": 8.0,
        "VTI": 9.0, "VOO": 9.5, "VEA": 7.5, "VXUS": 8.0, "VWO": 8.5,
        "VUG": 11.0, "VGT": 12.0, "QQQ": 12.5, "VNQ": 8.5,
        "ARKK": 15.0, "ARKQ": 14.0, "BITO": 18.0, "TQQQ": 20.0, "SOXL": 22.0,
        "FINX": 13.0, "IBB": 14.0, "COIN": 25.0
    }
    return return_mapping.get(symbol, 10.0)

def format_factor_name(factor: str) -> str:
    """Format factor names for display."""
    format_map = {
        "risk_score": "Risk Level",
        "target_value_log": "Goal Amount",
        "timeframe": "Investment Timeline", 
        "required_return": "Required Growth Rate",
        "monthly_contribution": "Monthly Savings",
        "market_volatility": "Market Volatility",
        "market_trend_score": "Market Trend"
    }
    return format_map.get(factor, factor.replace("_", " ").title())

def get_factor_description(factor: str) -> str:
    """Get description for factors."""
    descriptions = {
        "risk_score": "Your comfort level with investment risk",
        "target_value_log": "The financial goal you want to achieve",
        "timeframe": "How long you have to invest",
        "required_return": "The growth rate needed to reach your goal",
        "monthly_contribution": "Amount you can invest each month",
        "market_volatility": "Current market uncertainty levels",
        "market_trend_score": "Whether markets are trending up or down"
    }
    return descriptions.get(factor, "Factor influencing portfolio recommendations")

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float, handling lists and other types"""
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            # If it's a list/array, take the first element
            if len(value) > 0:
                return float(value[0]) if not isinstance(value[0], (list, tuple)) else float(np.mean(value))
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

# =============================================================================
# CORE SIMULATION FUNCTIONS
# =============================================================================

def download_stock_data(tickers: List[str], timeframe: int) -> pd.DataFrame:
    """Download historical stock data."""
    try:
        days_needed = max(timeframe * 365, 365)
        start_date = (datetime.today() - timedelta(days=days_needed)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')

        logger.info(f"Downloading data from {start_date} to {end_date} for {len(tickers)} stocks")

        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        threshold = len(data) * 0.7
        data = data.dropna(axis=1, thresh=threshold)
        
        logger.info(f"Downloaded data shape: {data.shape}")
        
        if data.empty:
            raise ValueError("No valid stock data available after quality filtering")
            
        return data
        
    except Exception as e:
        logger.error(f"Error downloading stock data: {str(e)}")
        raise ValueError(f"Failed to download stock data: {str(e)}")

def calculate_portfolio_weights(data: pd.DataFrame, risk_score: int) -> np.ndarray:
    """Original portfolio weights calculation."""
    num_assets = len(data.columns)
    logger.info(f"Calculating weights for {num_assets} assets (risk score: {risk_score})")
    
    if risk_score < 35:
        weights = np.array([1 / num_assets] * num_assets)
        logger.info("Using equal weights (conservative approach)")
        
    elif risk_score < 70:
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:num_assets])
        weights = weights / np.sum(weights)
        logger.info("Using moderate bias weighting")
        
    else:
        weights = np.array([0.4, 0.3, 0.2, 0.1][:num_assets])
        
        if len(weights) < num_assets:
            remaining = num_assets - len(weights)
            additional_weights = np.array([0.05] * remaining)
            weights = np.concatenate([weights, additional_weights])
        
        weights = weights / np.sum(weights)
        logger.info("Using concentrated weighting (aggressive approach)")
    
    return weights

def simulate_portfolio_growth(data: pd.DataFrame, weights: np.ndarray, 
                            lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
    """Enhanced portfolio growth simulation with comprehensive debugging."""
    try:
        logger.info(f"Starting simulation: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
        
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Weights: {weights}")
        logger.info(f"Weights sum: {weights.sum()}")
        
        first_day_values = data.iloc[0]
        
        problematic_stocks = first_day_values[first_day_values <= 0]
        if len(problematic_stocks) > 0:
            logger.error(f"Zero/negative prices on first day: {problematic_stocks}")
            first_day_values = first_day_values.replace(0, 0.01)
            logger.warning("Replaced zero prices with 0.01")
        
        normalized = data.div(first_day_values)
        logger.info(f"Normalized first values: {normalized.iloc[0]}")
        
        if normalized.isna().any().any():
            logger.warning("Found NaN values after normalization, forward filling...")
            normalized = normalized.fillna(method='ffill').fillna(1.0)
        
        weighted = normalized.dot(weights)
        logger.info(f"Weighted portfolio first 5 values:\n{weighted.head()}")
        
        portfolio_values = []
        contributions = []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        for i, (date, growth_factor) in enumerate(weighted.items()):
            # Add monthly contribution every 21 trading days (approximately monthly)
            if i > 0 and i % 21 == 0:
                current_value += monthly
                total_contributions += monthly
            
            current_value = current_value * growth_factor
            
            portfolio_values.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": float(current_value)
            })
            
            contributions.append({
                "date": date.strftime('%Y-%m-%d'),
                "contribution": float(total_contributions)
            })
        
        end_value = float(current_value)
        portfolio_return = ((end_value - total_contributions) / total_contributions) * 100 if total_contributions > 0 else 0
        
        logger.info(f"Simulation completed: £{total_contributions:,.2f} → £{end_value:,.2f} ({portfolio_return:.1f}%)")
        
        return {
            "starting_value": float(lump_sum),
            "end_value": end_value,
            "portfolio_return": float(portfolio_return),
            "total_contributed": float(total_contributions),
            "timeline": {
                "portfolio": portfolio_values,
                "contributions": contributions
            },
            "breakdown": {ticker: float(weight) for ticker, weight in zip(data.columns, weights)}
        }
        
    except Exception as e:
        logger.error(f"Error in portfolio growth simulation: {str(e)}")
        raise ValueError(f"Portfolio simulation failed: {str(e)}")

def get_fallback_stocks_by_risk_profile(risk_score: int, risk_label: str) -> List[str]:
    """Original fallback stock selection method."""
    logger.info(f"Using fallback selection for {risk_label} risk profile (score: {risk_score})")
    
    if risk_score < 35:
        return ["VTI", "BND", "VEA", "VTEB", "VWO"]
    elif risk_score < 70:
        return ["VTI", "VEA", "VWO", "VNQ", "BND"]
    else:
        return ["VTI", "VGT", "VUG", "ARKK", "VEA"]

# =============================================================================
# ENHANCED AI RECOMMENDATIONS
# =============================================================================

async def get_enhanced_ai_recommendations(
    target_value: float, timeframe: int, risk_score: int, risk_label: str,
    current_investment: float = 0, monthly_contribution: float = 0
) -> Dict[str, Any]:
    """Get enhanced AI recommendations with factor analysis, SHAP explanations and goal analysis."""
    
    if not WEALTHWISE_AVAILABLE:
        logger.warning("WealthWise not available, using fallback recommendations")
        return {
            "stocks": get_fallback_stocks_by_risk_profile(risk_score, risk_label),
            "method": "fallback"
        }
    
    try:
        logger.info("Initializing WealthWise enhanced recommendation system with factor analysis")
        
        init_result = initialize_complete_system({
            'LOG_LEVEL': 'INFO',
            'LOG_TO_FILE': False,
            'ENABLE_PERFORMANCE_TRACKING': True
        })
        
        if not init_result['success']:
            raise Exception(f"WealthWise initialization failed: {init_result.get('error')}")
        
        recommender = EnhancedStockRecommender()
        factor_analyzer = FactorAnalyzer()
        shap_explainer = SHAPExplainer()
        goal_calculator = GoalCalculator()
        feasibility_assessor = FeasibilityAssessor()
        market_detector = MarketRegimeDetector()
        
        goal_analysis = goal_calculator.calculate_required_return(
            target_value, current_investment, timeframe, monthly_contribution
        )
        
        feasibility_assessment = feasibility_assessor.assess_goal_feasibility(
            goal_analysis["required_return"], risk_score, timeframe,
            current_investment, monthly_contribution
        )
        
        market_regime = market_detector.detect_market_regime()
        
        initial_recommendations = recommender.recommend_stocks(
            target_value, timeframe, risk_score, 
            current_investment, monthly_contribution
        )
        
        candidate_stocks = set(initial_recommendations)
        
        if risk_score < 35:
            candidate_stocks.update(["VTI", "BND", "VEA", "VTEB", "VWO", "AGG", "VNQ", "SCHD", "VYM"])
        elif risk_score < 70:
            candidate_stocks.update(["VTI", "VEA", "VWO", "VNQ", "BND", "VUG", "VGT", "VOO", "VXUS"])
        else:
            candidate_stocks.update(["VTI", "VGT", "VUG", "ARKK", "VEA", "QQQ", "ARKQ", "TQQQ", "SOXL"])
        
        if market_regime.get('regime') == 'bear':
            candidate_stocks.update(["VYM", "SCHD", "VDC", "VHT"])
        elif market_regime.get('regime') == 'bull':
            candidate_stocks.update(["VGT", "VUG", "QQQ"])
            
        candidate_stocks = list(candidate_stocks)
        
        try:
            ranked_stocks = factor_analyzer.rank_stocks_by_factors(
                candidate_stocks,
                market_regime=market_regime,
                risk_score=risk_score,
                timeframe=timeframe
            )
            
            num_stocks = min(6, len(ranked_stocks))
            factor_selected_stocks = [stock for stock, score in ranked_stocks[:num_stocks]]
            
            logger.info(f"Factor analysis selected: {factor_selected_stocks}")
            
        except Exception as factor_error:
            logger.warning(f"Factor analysis failed: {factor_error}, using initial recommendations")
            factor_selected_stocks = initial_recommendations[:6]
        
        shap_explanation = None
        if shap_explainer.is_available():
            shap_explanation = shap_explainer.get_shap_explanation(
                target_value, timeframe, risk_score,
                current_investment, monthly_contribution,
                market_regime.get('current_vix', 20),
                market_regime.get('trend_score', 2.5)
            )
        else:
            success = shap_explainer.train_shap_model(num_samples=1000)
            if success:
                shap_explanation = shap_explainer.get_shap_explanation(
                    target_value, timeframe, risk_score,
                    current_investment, monthly_contribution,
                    market_regime.get('current_vix', 20),
                    market_regime.get('trend_score', 2.5)
                )
        
        logger.info(f"Enhanced recommendations complete: {len(factor_selected_stocks)} stocks selected via factor analysis")
        
        return {
            "stocks": factor_selected_stocks,
            "shap_explanation": shap_explanation,
            "goal_analysis": goal_analysis,
            "feasibility_assessment": feasibility_assessment,
            "market_regime": market_regime,
            "method": "wealthwise_enhanced_with_factors"
        }
        
    except Exception as e:
        logger.error(f"Enhanced AI recommendations failed: {e}")
        logger.warning("Falling back to original recommendation method")
        
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
        
        logger.info("Using WealthWise correlation-optimized weights")
        return weights_array
        
    except Exception as e:
        logger.warning(f"WealthWise optimization failed: {e}, using fallback")
        return calculate_portfolio_weights(data, risk_score)

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
    """Create visualizations using the actual methods available in VisualizationEngine."""
    
    viz_engine = get_visualization_engine()
    if not viz_engine:
        logger.warning("VisualizationEngine not available, skipping visualizations")
        return {}
    
    try:
        viz_dir = Path("static/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        simulation_prefix = f"sim_{simulation_id}" if simulation_id else "temp_simulation"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{simulation_prefix}_{timestamp}"
        
        visualization_paths = {}
        
        # Portfolio Composition Chart
        try:
            stocks = [stock["symbol"] for stock in stocks_picked]
            weights = {stock["symbol"]: stock["allocation"] for stock in stocks_picked}
            
            composition_path = viz_dir / f"{base_filename}_composition.png"
            result = viz_engine.create_portfolio_composition_chart(
                stocks, weights, str(composition_path)
            )
            
            if "saved" in result.lower():
                visualization_paths["portfolio_composition"] = str(composition_path)
                logger.info(f"Portfolio composition chart saved: {composition_path}")
                
        except Exception as e:
            logger.error(f"Error creating composition chart: {e}")
        
        # SHAP Waterfall Chart
        if shap_explanation:
            try:
                shap_path = viz_dir / f"{base_filename}_shap_explanation.png"
                cleaned_shap = clean_shap_for_visualization(shap_explanation)
                
                result = viz_engine.create_shap_waterfall_chart(
                    cleaned_shap, str(shap_path)
                )
                
                if "saved" in result.lower():
                    visualization_paths["shap_explanation"] = str(shap_path)
                    logger.info(f"SHAP explanation chart saved: {shap_path}")
                    
            except Exception as e:
                logger.error(f"Error creating SHAP chart: {e}")
        
        # Risk-Return Scatter Plot
        try:
            if len(stock_data.columns) > 1:
                returns = stock_data.pct_change().dropna()
                portfolio_weights = np.array([stock["allocation"] for stock in stocks_picked])
                portfolio_returns = returns.dot(portfolio_weights)
                
                portfolio_metrics = {
                    'expected_return': float(portfolio_returns.mean() * 252),
                    'volatility': float(portfolio_returns.std() * np.sqrt(252)),
                    'sharpe_ratio': float((portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)))
                }
                
                benchmark_data = []
                for ticker in stock_data.columns[:3]:
                    if ticker in returns.columns:
                        stock_return = returns[ticker].mean() * 252
                        stock_vol = returns[ticker].std() * np.sqrt(252)
                        benchmark_data.append({
                            'name': f'{ticker} Individual',
                            'expected_return': float(stock_return),
                            'volatility': float(stock_vol)
                        })
                
                risk_return_path = viz_dir / f"{base_filename}_risk_return.png"
                result = viz_engine.create_risk_return_scatter(
                    portfolio_metrics, benchmark_data, str(risk_return_path)
                )
                
                if "saved" in result.lower():
                    visualization_paths["risk_return_analysis"] = str(risk_return_path)
            
        except Exception as e:
            logger.error(f"Error creating risk vs return chart: {e}")
        
        # Factor Importance Chart
        if shap_explanation and shap_explanation.get("feature_contributions"):
            try:
                factor_scores = shap_explanation["feature_contributions"]
                
                factor_path = viz_dir / f"{base_filename}_factor_importance.png"
                result = viz_engine.create_factor_importance_chart(
                    factor_scores, str(factor_path)
                )
                
                if "saved" in result.lower():
                    visualization_paths["factor_importance"] = str(factor_path)
                    
            except Exception as e:
                logger.error(f"Error creating factor importance chart: {e}")
        
        # Market Regime Visualization
        market_regime = shap_explanation.get('market_regime') if shap_explanation else None
        if market_regime:
            try:
                market_path = viz_dir / f"{base_filename}_market_regime.png"
                result = viz_engine.create_market_regime_visualization(
                    market_regime, str(market_path)
                )
                
                if "saved" in result.lower():
                    visualization_paths["market_regime"] = str(market_path)
                
            except Exception as e:
                logger.error(f"Error creating market regime chart: {e}")
        
        logger.info(f"Created {len(visualization_paths)} visualizations successfully")
        return visualization_paths
        
    except Exception as e:
        logger.error(f"Error creating simulation visualizations: {e}")
        return {}

def clean_shap_for_visualization(shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced SHAP explanation data cleaning to fix matplotlib formatting issues."""
    
    try:
        cleaned_shap = {}
        
        def clean_value_recursive(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, dict):
                return {str(key): clean_value_recursive(val) for key, val in value.items()}
            elif isinstance(value, (list, tuple)):
                return [clean_value_recursive(item) for item in value]
            else:
                return value
        
        for key, value in shap_explanation.items():
            if key == "feature_contributions" and isinstance(value, dict):
                cleaned_contributions = {}
                for feature, contribution in value.items():
                    if isinstance(contribution, (np.floating, np.integer)):
                        cleaned_contributions[str(feature)] = float(contribution)
                    elif isinstance(contribution, (list, tuple, np.ndarray)):
                        if len(contribution) > 0:
                            cleaned_contributions[str(feature)] = float(contribution[0])
                        else:
                            cleaned_contributions[str(feature)] = 0.0
                    else:
                        cleaned_contributions[str(feature)] = float(contribution)
                cleaned_shap[key] = cleaned_contributions
            elif key == "base_value":
                if isinstance(value, (list, tuple, np.ndarray)):
                    cleaned_shap[key] = float(value[0]) if len(value) > 0 else 50.0
                elif isinstance(value, (int, float, np.floating, np.integer)):
                    cleaned_shap[key] = float(value)
                else:
                    cleaned_shap[key] = 50.0
            else:
                cleaned_shap[key] = clean_value_recursive(value)
        
        logger.info(f"SHAP data cleaned successfully. Keys: {list(cleaned_shap.keys())}")
        return cleaned_shap
        
    except Exception as e:
        logger.error(f"Error cleaning SHAP data: {e}")
        
        safe_shap = {
            'feature_contributions': {
                'risk_score': -1.8,
                'timeframe': -0.8,
                'target_value_log': 0.05,
                'required_return': 12.0,
                'monthly_contribution': -1.7
            },
            'base_value': 50.0,
            'portfolio_quality_score': 75.0,
            'human_readable_explanation': {
                'risk_tolerance': 'Your risk tolerance influenced the recommendation',
                'time_horizon': 'Your investment timeframe was considered',
                'goal_alignment': 'Portfolio aligned with your financial goals'
            }
        }
        
        logger.warning("Using safe fallback SHAP structure")
        return safe_shap

async def update_visualization_paths_with_id(
    simulation_id: int, 
    temp_paths: Dict[str, str]
) -> Dict[str, str]:
    """Update temporary visualization file paths with actual simulation ID."""
    
    updated_paths = {}
    
    try:
        for viz_type, temp_path in temp_paths.items():
            temp_file = Path(temp_path)
            
            if temp_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"sim_{simulation_id}_{timestamp}_{viz_type}.png"
                new_path = temp_file.parent / new_filename
                
                temp_file.rename(new_path)
                updated_paths[viz_type] = str(new_path)
                
                logger.info(f"Renamed {temp_path} → {new_path}")
            else:
                logger.warning(f"Temporary file not found: {temp_path}")
                updated_paths[viz_type] = temp_path
        
        return updated_paths
        
    except Exception as e:
        logger.error(f"Error updating visualization paths: {e}")
        return temp_paths

# =============================================================================
# REACT CHART DATA ENDPOINTS
# =============================================================================

async def get_simulation_chart_data(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Get raw chart data for React components instead of static images."""
    
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        results = simulation.results or {}
        shap_explanation = results.get("shap_explanation", {})
        chart_data = {}
        
        # 1. Portfolio Composition Data
        stocks_picked = results.get("stocks_picked", [])
        breakdown = results.get("breakdown", {})
        
        if stocks_picked:
            chart_data["portfolio_composition"] = []
            total_allocation = 0
            
            for stock in stocks_picked:
                symbol = stock.get("symbol", stock) if isinstance(stock, dict) else stock
                name = stock.get("name", symbol) if isinstance(stock, dict) else symbol
                allocation = stock.get("allocation", 0) if isinstance(stock, dict) else breakdown.get(symbol, 0)
                
                # Safe conversion here
                allocation_float = safe_float_conversion(allocation, 0.0)
                
                if allocation_float < 1:
                    allocation_float = allocation_float * 100
                
                chart_data["portfolio_composition"].append({
                    "symbol": symbol,
                    "name": name,
                    "value": round(allocation_float, 2),
                    "allocation": round(allocation_float, 2)
                })
                total_allocation += allocation
            
            if total_allocation != 100 and total_allocation > 0:
                for item in chart_data["portfolio_composition"]:
                    item["value"] = round((item["value"] / total_allocation) * 100, 2)
                    item["allocation"] = item["value"]
        
        # 2. SHAP Waterfall Data
        if shap_explanation.get("feature_contributions"):
            feature_contributions = shap_explanation["feature_contributions"]
            base_value = shap_explanation.get("base_value", [50])[0] if shap_explanation.get("base_value") else 50
            
            cumulative = base_value
            waterfall_data = []
            
            sorted_features = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(float(x[1])), 
                reverse=True
            )
            
            for feature, value in sorted_features:
                num_value = float(value)
                start = cumulative
                cumulative += num_value
                
                waterfall_data.append({
                    "feature": format_factor_name(feature),
                    "value": round(num_value, 3),
                    "cumulative": round(cumulative, 3),
                    "start": round(start, 3),
                    "isPositive": num_value >= 0
                })
            
            chart_data["shap_waterfall"] = waterfall_data
        
        # 3. Factor Importance Data
        if shap_explanation.get("feature_contributions"):
            feature_contributions = shap_explanation["feature_contributions"]
            
            chart_data["factor_importance"] = [
                {
                    "factor": format_factor_name(feature),
                    "importance": round(safe_float_conversion(value), 3),
                    "isPositive": float(value) >= 0,
                    "description": get_factor_description(feature)
                }
                for feature, value in sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(float(x[1])),
                    reverse=True
                )
            ]
        
        # 4. Market Regime Data
        market_regime = results.get("market_regime", {}) or shap_explanation.get("market_regime", {})
        if market_regime:
            chart_data["market_regime"] = [
                {
                    "name": "VIX Level",
                    "current": round(market_regime.get("current_vix", 15), 1),
                    "normal": 20,
                    "target": 20,
                    "status": "Low" if market_regime.get("current_vix", 15) < 20 else "High"
                },
                {
                    "name": "Trend Score", 
                    "current": round(market_regime.get("trend_score", 3), 1),
                    "normal": 3,
                    "target": 5,
                    "status": "Bullish" if market_regime.get("trend_score", 3) > 3 else "Neutral"
                },
                {
                    "name": "3M Returns",
                    "current": round((market_regime.get("returns_3m", 0.05) * 100), 1),
                    "normal": 5,
                    "target": 10,
                    "status": "Above" if market_regime.get("returns_3m", 0.05) > 0.05 else "Below"
                }
            ]
        
        # 5. Risk-Return Analysis
        if stocks_picked:
            chart_data["risk_return_analysis"] = []
            
            for i, stock in enumerate(stocks_picked):
                symbol = stock.get("symbol", stock) if isinstance(stock, dict) else stock
                name = stock.get("name", symbol) if isinstance(stock, dict) else symbol
                allocation = stock.get("allocation", 0) if isinstance(stock, dict) else breakdown.get(symbol, 0)
                
                risk_estimate = estimate_stock_risk(symbol)
                return_estimate = estimate_stock_return(symbol)
                allocation_float = safe_float_conversion(allocation, 0.0)
                
                chart_data["risk_return_analysis"].append({
                    "symbol": symbol,
                    "name": name,
                    "risk": round(risk_estimate, 2),
                    "return": round(return_estimate, 2),
                    "allocation": round(allocation_float * 100 if allocation_float < 1 else allocation_float, 2)
                })
        
        # 6. Goal Analysis Data
        goal_analysis = results.get("goal_analysis", {})
        if goal_analysis or simulation:
            chart_data["goal_analysis"] = {
                "target_value": simulation.target_value,
                "projected_value": results.get("end_value", 0),
                "probability_of_success": goal_analysis.get("feasibility_score", 75),
                "monthly_contribution": simulation.monthly,
                "timeframe_years": simulation.timeframe,
                "progress_percentage": min(100, (results.get("end_value", 0) / simulation.target_value) * 100) if simulation.target_value > 0 else 0
            }
        
        # 7. Timeline Data for Performance Charts
        timeline = results.get("timeline", {})
        if timeline.get("portfolio"):
            portfolio_timeline = timeline["portfolio"]
            contributions_timeline = timeline.get("contributions", [])
            
            chart_data["performance_timeline"] = {
                "portfolio": [
                    {
                        "date": entry["date"],
                        "value": entry["value"],
                        "return_pct": ((entry["value"] / portfolio_timeline[0]["value"]) - 1) * 100 if portfolio_timeline else 0
                    }
                    for entry in portfolio_timeline[-252:]
                ],
                "contributions": contributions_timeline[-252:] if contributions_timeline else []
            }
        
        return {
            "success": True,
            "simulation_id": simulation_id,
            "chart_data": chart_data,
            "available_charts": list(chart_data.keys()),
            "metadata": {
                "has_shap": bool(shap_explanation),
                "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
                "generated_at": datetime.now().isoformat(),
                "data_quality": assess_chart_data_quality(chart_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting chart data for simulation {simulation_id}: {e}")
        return {"error": str(e)}

async def get_enhanced_portfolio_data(simulation_id: int, db: Session) -> Dict[str, Any]:
    """Get enhanced portfolio data with historical performance if available."""
    
    try:
        simulation = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id
        ).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        results = simulation.results or {}
        timeline = results.get("timeline", {})
        enhanced_data = {}
        
        if timeline.get("portfolio"):
            portfolio_timeline = timeline["portfolio"]
            enhanced_data["portfolio_timeline"] = [
                {
                    "date": entry["date"],
                    "value": entry["value"],
                    "return_pct": ((entry["value"] / portfolio_timeline[0]["value"]) - 1) * 100 if portfolio_timeline else 0
                }
                for entry in portfolio_timeline
            ]
        
        if timeline.get("contributions"):
            enhanced_data["contributions_timeline"] = timeline["contributions"]
        
        stocks_picked = results.get("stocks_picked", [])
        if stocks_picked:
            enhanced_data["stock_performance"] = [
                {
                    "symbol": stock.get("symbol", stock) if isinstance(stock, dict) else stock,
                    "name": stock.get("name", stock) if isinstance(stock, dict) else stock,
                    "weight": stock.get("allocation", 0) if isinstance(stock, dict) else 0,
                    "estimated_return": estimate_stock_return(stock.get("symbol", stock) if isinstance(stock, dict) else stock),
                    "estimated_risk": estimate_stock_risk(stock.get("symbol", stock) if isinstance(stock, dict) else stock)
                }
                for stock in stocks_picked
            ]
        
        return {
            "success": True,
            "simulation_id": simulation_id,
            "enhanced_data": enhanced_data
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced portfolio data: {e}")
        return {"error": str(e)}

def assess_chart_data_quality(chart_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality and completeness of chart data."""
    expected_charts = ["portfolio_composition", "factor_importance", "goal_analysis", "market_regime", "risk_return_analysis"]
    available = len([chart for chart in expected_charts if chart in chart_data])
    completeness_score = (available / len(expected_charts)) * 100
    
    return {
        "completeness_score": completeness_score,
        "available_charts": len(chart_data),
        "missing_charts": [chart for chart in expected_charts if chart not in chart_data],
        "data_quality": "high" if completeness_score >= 80 else "medium" if completeness_score >= 60 else "low"
    }

# =============================================================================
# AI SUMMARY GENERATION
# =============================================================================

async def generate_enhanced_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any],
    shap_explanation: Optional[Dict] = None, goal_analysis: Optional[Dict] = None,
    feasibility_assessment: Optional[Dict] = None, market_regime: Optional[Dict] = None
) -> str:
    """Generate comprehensive AI summary with SHAP explanations and news analysis."""
    
    try:
        logger.info("Generating INTEGRATED AI summary with SHAP + News Analysis")
        
        from backend.services.portfolio_simulator.ai_analysis import AIAnalysisService
        ai_service = AIAnalysisService()
        
        try:
            portfolio_news_analysis = await ai_service._analyze_portfolio_news_history(
                stocks_picked, user_data, simulation_results
            )
        except Exception as news_error:
            logger.warning(f"News analysis failed: {news_error}, proceeding without news data")
            portfolio_news_analysis = {"error": str(news_error)}
        
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
        
        integrated_summary = await ai_service._get_groq_response(integrated_prompt)
        
        logger.info("Integrated AI summary with SHAP + News generated successfully!")
        return ai_service._format_ai_response(integrated_summary)
        
    except Exception as e:
        logger.warning(f"Integrated AI summary failed: {e}. Using simple enhanced summary...")
        
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
    
    goal = user_data.get("goal", "wealth building")
    lump_sum = user_data.get("lump_sum", 0)
    monthly = user_data.get("monthly", 0)
    timeframe = user_data.get("timeframe", 10)
    target_value = user_data.get("target_value", 50000)
    
    end_value = simulation_results.get("end_value", 0)
    total_contributed = lump_sum + (monthly * timeframe * 12)
    target_achieved = end_value >= target_value
    
    symbols = [stock.get('symbol', '') for stock in stocks_picked]
    
    shap_context = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_context = f"""
AI DECISION EXPLANATIONS (SHAP Analysis):
The AI specifically chose this portfolio because:"""
        
        explanations = shap_explanation["human_readable_explanation"]
        for factor, explanation in explanations.items():
            if explanation and len(explanation) > 10:
                shap_context += f"""
• {explanation}"""
        
        quality_score = shap_explanation.get('portfolio_quality_score')
        if quality_score:
            shap_context += f"""

AI Portfolio Quality Score: {quality_score:.1f}/10
This score reflects how well the AI believes this portfolio matches your goals and risk tolerance."""

    goal_context = ""
    if goal_analysis and feasibility_assessment:
        required_return = goal_analysis.get('required_return_percent', 0)
        feasibility = feasibility_assessment.get('feasibility_score', 0)
        
        goal_context = f"""
GOAL-ORIENTED ANALYSIS:
• Your Target: £{target_value:,.0f} in {timeframe} years
• Required Annual Return: {required_return:.1f}%
• AI Feasibility Assessment: {feasibility:.0f}% achievable
• Recommendation: {feasibility_assessment.get('recommendations', {}).get('primary', 'Continue with your plan')}"""

    market_context = ""
    if market_regime:
        regime = market_regime.get('regime', 'neutral')
        trend_score = market_regime.get('trend_score', 2.5)
        vix = market_regime.get('current_vix', 20)
        
        market_context = f"""
CURRENT MARKET CONDITIONS:
• Market Regime: {regime.title()} market environment
• Trend Strength: {trend_score:.1f}/5.0 (bullish trend)
• Volatility (VIX): {vix:.1f} ({"Low" if vix < 20 else "Moderate" if vix < 30 else "High"} fear level)"""

    return f"""
You are an expert financial educator creating a comprehensive portfolio analysis that combines AI explainability with real-world market education.

PORTFOLIO PERFORMANCE SUMMARY:
Holdings: {', '.join(symbols)}
Goal: {goal}
Target: £{target_value:,.0f}
Total Invested: £{total_contributed:,.0f}
Final Value: £{end_value:,.0f}
Result: {'GOAL ACHIEVED!' if target_achieved else 'PROGRESS MADE'}
Risk Level: {risk_label} ({risk_score}/100)
Investment Period: {timeframe} years

{shap_context}

{goal_context}

{market_context}

Create a comprehensive educational summary explaining both the AI's reasoning and real-world market context for this portfolio recommendation.
"""

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

    shap_section = ""
    if shap_explanation and "human_readable_explanation" in shap_explanation:
        shap_section = f"""

The AI analysis considered multiple factors for your specific situation:
"""
        explanations = shap_explanation["human_readable_explanation"]
        for factor, explanation in explanations.items():
            if explanation and len(explanation) > 10:
                shap_section += f"• {explanation}\n"

    goal_section = ""
    if goal_analysis and feasibility_assessment:
        required_return = goal_analysis.get('required_return_percent', 0)
        feasibility = feasibility_assessment.get('feasibility_score', 0)
        
        goal_section = f"""

To reach your £{target_value:,.0f} target, you needed {required_return:.1f}% annual returns. The AI assessed your goal as {feasibility:.0f}% feasible given your risk tolerance and timeframe.
"""

    return f"{base_summary}{shap_section}{goal_section}\n\n*This portfolio was optimized specifically for your goals using explainable AI.*"

# =============================================================================
# DATABASE FUNCTIONS
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
    """Enhanced version of save_simulation_to_db with visualization support."""
    
    try:
        logger.info("Saving enhanced simulation with SHAP data and visualizations to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
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
        
        cleaned_results = clean_simulation_results_for_db(enhanced_results)
        
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
        
        logger.info(f"Enhanced simulation with visualizations saved (ID: {simulation.id})")
        return simulation
        
    except Exception as e:
        logger.error(f"Error saving enhanced simulation: {str(e)}")
        db.rollback()
        
        try:
            logger.warning("Attempting to save simulation without visualizations")
            
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
            
            logger.warning(f"Saved basic simulation without visualizations (ID: {basic_simulation.id})")
            return basic_simulation
            
        except Exception as basic_error:
            logger.error(f"Even basic simulation save failed: {basic_error}")
            db.rollback()
            raise

def format_enhanced_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """Format enhanced simulation response with SHAP explanations and visualizations."""
    
    results = simulation.results or {}
    shap_explanation = results.get("shap_explanation")
    visualization_paths = results.get("visualization_paths", {})
    
    has_shap_explanations = bool(shap_explanation)
    has_visualizations = bool(visualization_paths)
    
    logger.info(f"Formatting enhanced response for simulation {simulation.id}")
    logger.info(f"Has SHAP explanations: {has_shap_explanations}")
    logger.info(f"Has visualizations: {has_visualizations}")
    
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
        "shap_explanation": shap_explanation,
        "has_shap_explanations": has_shap_explanations,
        "visualization_paths": visualization_paths,
        "has_visualizations": has_visualizations,
        "wealthwise_enhanced": results.get("wealthwise_enhanced", False),
        "methodology": results.get("methodology", "Standard simulation"),
        "available_visualizations": list(visualization_paths.keys()),
        "visualization_count": len(visualization_paths)
    }
    
    return response

def clean_simulation_results_for_db(results: Dict[str, Any]) -> Dict[str, Any]:
    """Clean all simulation results before saving to database."""
    try:
        logger.info("Cleaning simulation results for database storage")
        
        cleaned_results = serialize_for_json(results)
        json.dumps(cleaned_results)
        
        logger.info("Simulation results successfully cleaned for database")
        return cleaned_results
        
    except Exception as e:
        logger.error(f"Failed to clean simulation results: {e}")
        
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
            logger.error(f"Even fallback serialization failed: {fallback_error}")
            return {
                "status": "serialization_failed",
                "error": str(e),
                "fallback_error": str(fallback_error),
                "timestamp": datetime.now().isoformat()
            }

def serialize_for_json(data: Any) -> Any:
    """Recursively convert non-serializable objects to JSON-compatible types."""
    
    if data is None:
        return None
    
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

# =============================================================================
# MAIN PORTFOLIO SIMULATION FUNCTION
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Enhanced portfolio simulation with SHAP explanations and visualizations."""
    
    try:
        logger.info("Starting enhanced portfolio simulation with visualizations")
        
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

        logger.info(f"User profile: goal={user_data['goal']}, target=£{user_data['target_value']:,.2f}, timeframe={user_data['timeframe']} years")

        recommendation_result = await get_enhanced_ai_recommendations(
            target_value=user_data["target_value"],
            timeframe=user_data["timeframe"],
            risk_score=risk_score,
            risk_label=risk_label,
            current_investment=user_data["lump_sum"],
            monthly_contribution=user_data["monthly"]
        )
        
        tickers = recommendation_result["stocks"]
        shap_explanation = recommendation_result.get("shap_explanation")
        goal_analysis = recommendation_result.get("goal_analysis")
        feasibility_assessment = recommendation_result.get("feasibility_assessment")
        market_regime = recommendation_result.get("market_regime")
        
        logger.info(f"Enhanced AI recommended stocks: {tickers}")

        lump_sum = user_data["lump_sum"]
        monthly = user_data["monthly"]
        timeframe = user_data["timeframe"]
        
        if lump_sum <= 0 and monthly <= 0:
            raise ValueError("Either lump sum or monthly investment must be greater than 0")
        
        if timeframe <= 0:
            raise ValueError("Timeframe must be greater than 0")

        stock_data = download_stock_data(tickers, timeframe)
        weights = calculate_enhanced_portfolio_weights(stock_data, risk_score, recommendation_result)
        
        stocks_picked = [
            {
                "symbol": ticker, 
                "name": get_company_name(ticker),
                "allocation": round(float(weight), 4),
                "explanation": get_stock_explanation(ticker, recommendation_result)
            }
            for ticker, weight in zip(tickers, weights)
        ]
        
        logger.info("Enhanced portfolio allocation:")
        for stock in stocks_picked:
            logger.info(f"   {stock['symbol']}: {stock['allocation']*100:.1f}% ({stock['name']})")

        simulation_results = simulate_portfolio_growth(stock_data, weights, lump_sum, monthly, timeframe)

        if shap_explanation and market_regime:
            shap_explanation['market_regime'] = market_regime
        
        visualization_paths = await create_simulation_visualizations(
            simulation_id=None,
            stocks_picked=stocks_picked,
            simulation_results=simulation_results,
            shap_explanation=shap_explanation,
            user_data=user_data,
            stock_data=stock_data
        )

        ai_summary = await generate_enhanced_ai_summary(
            stocks_picked, user_data, risk_score, risk_label, 
            simulation_results, shap_explanation, goal_analysis, 
            feasibility_assessment, market_regime
        )

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

        if visualization_paths and simulation.id:
            updated_paths = await update_visualization_paths_with_id(
                simulation.id, visualization_paths
            )
            
            if updated_paths != visualization_paths:
                simulation.results["visualization_paths"] = updated_paths
                db.commit()

        logger.info(f"Enhanced portfolio simulation completed successfully (ID: {simulation.id})")
        return format_enhanced_simulation_response(simulation)

    except Exception as e:
        logger.error(f"Enhanced portfolio simulation failed: {str(e)}")
        db.rollback()
        
        logger.warning("Falling back to original simulation method")
        return await simulate_portfolio_fallback(sim_input, db)

# =============================================================================
# FALLBACK SIMULATION FUNCTION
# =============================================================================

async def simulate_portfolio_fallback(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """Fallback portfolio simulation using basic methods."""
    
    try:
        logger.info("Starting fallback portfolio simulation")
        
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

        # Use fallback stock selection
        tickers = get_fallback_stocks_by_risk_profile(risk_score, risk_label)
        logger.info(f"Fallback recommended stocks: {tickers}")

        lump_sum = user_data["lump_sum"]
        monthly = user_data["monthly"]
        timeframe = user_data["timeframe"]
        
        if lump_sum <= 0 and monthly <= 0:
            raise ValueError("Either lump sum or monthly investment must be greater than 0")
        
        if timeframe <= 0:
            raise ValueError("Timeframe must be greater than 0")

        stock_data = download_stock_data(tickers, timeframe)
        weights = calculate_portfolio_weights(stock_data, risk_score)
        
        stocks_picked = [
            {
                "symbol": ticker, 
                "name": get_company_name(ticker),
                "allocation": round(safe_float_conversion(weight), 4),
                "explanation": get_stock_explanation(ticker, recommendation_result)
            }
            for ticker, weight in zip(tickers, weights)
        ]
        
        logger.info("Fallback portfolio allocation:")
        for stock in stocks_picked:
            logger.info(f"   {stock['symbol']}: {stock['allocation']*100:.1f}% ({stock['name']})")

        simulation_results = simulate_portfolio_growth(stock_data, weights, lump_sum, monthly, timeframe)

        # Generate basic AI summary
        ai_summary = generate_basic_ai_summary(
            stocks_picked, user_data, risk_score, risk_label, simulation_results
        )

        simulation = save_basic_simulation_to_db(
            db=db,
            sim_input=sim_input,
            user_data=user_data,
            risk_score=risk_score,
            risk_label=risk_label,
            ai_summary=ai_summary,
            stocks_picked=stocks_picked,
            simulation_results=simulation_results
        )

        logger.info(f"Fallback portfolio simulation completed successfully (ID: {simulation.id})")
        return format_basic_simulation_response(simulation)

    except Exception as e:
        logger.error(f"Fallback portfolio simulation also failed: {str(e)}")
        db.rollback()
        raise ValueError(f"Portfolio simulation failed completely: {str(e)}")

def generate_basic_ai_summary(
    stocks_picked: List[Dict], user_data: Dict[str, Any], 
    risk_score: int, risk_label: str, simulation_results: Dict[str, Any]
) -> str:
    """Generate basic AI summary without SHAP explanations."""
    
    goal = user_data.get("goal", "wealth building")
    timeframe = user_data.get("timeframe", 10)
    start_value = simulation_results.get("starting_value", 0)
    end_value = simulation_results.get("end_value", 0)
    target_value = user_data.get("target_value", 50000)
    stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
    target_achieved = end_value >= target_value
    
    return f"""
Your {risk_label.lower()} risk portfolio, invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years for your {goal} goal. 

Your target of £{target_value:,.2f} was {'achieved' if target_achieved else 'partially achieved'}. This portfolio was selected based on your risk tolerance and investment timeframe.

*This is a basic simulation using standard portfolio optimization methods.*
"""

def save_basic_simulation_to_db(
    db, sim_input: Dict[str, Any], user_data: Dict[str, Any],
    risk_score: int, risk_label: str, ai_summary: str,
    stocks_picked: List[Dict], simulation_results: Dict[str, Any]
):
    """Save basic simulation to database without enhanced features."""
    
    try:
        logger.info("Saving basic simulation to database")
        
        target_reached = simulation_results["end_value"] >= user_data["target_value"]
        
        basic_results = {
            "name": user_data["goal"],
            "stocks_picked": stocks_picked,
            "starting_value": simulation_results["starting_value"],
            "end_value": simulation_results["end_value"],
            "return": simulation_results["portfolio_return"],
            "target_reached": target_reached,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "timeline": simulation_results.get("timeline", {}),
            "breakdown": simulation_results.get("breakdown", {}),
            "wealthwise_enhanced": False,
            "methodology": "Basic portfolio simulation",
            "created_timestamp": datetime.now().isoformat()
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
            results=basic_results
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        logger.info(f"Basic simulation saved (ID: {simulation.id})")
        return simulation
        
    except Exception as e:
        logger.error(f"Error saving basic simulation: {str(e)}")
        db.rollback()
        raise

def format_basic_simulation_response(simulation: models.Simulation) -> Dict[str, Any]:
    """Format basic simulation response."""
    
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
        "results": results,
        "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat(),
        "shap_explanation": None,
        "has_shap_explanations": False,
        "visualization_paths": {},
        "has_visualizations": False,
        "wealthwise_enhanced": False,
        "methodology": "Basic portfolio simulation",
        "available_visualizations": [],
        "visualization_count": 0
    }
    
    return response