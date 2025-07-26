# ai_models/stock_model/core/recommender.py

"""
Enhanced Stock Recommender - Main Recommendation Engine

This is the core class that orchestrates all AI components to provide
intelligent, transparent portfolio recommendations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import configuration
from .config import (
    ASSET_UNIVERSES, 
    BACKUP_TICKERS, 
    get_risk_category,
    get_risk_score,
    validate_risk_category
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedStockRecommender:
    """
    AI-Powered Stock Recommendation System with Explainable AI
    
    This class implements a comprehensive investment advisory system that uses
    multiple AI techniques to provide transparent, goal-oriented portfolio
    recommendations. Unlike traditional "black box" systems, this recommender
    explains its reasoning using SHAP values and educational narratives.
    
    Core Philosophy:
    - Education over automation
    - Transparency over complexity  
    - Goal-oriented over generic advice
    - Risk-appropriate over one-size-fits-all
    
    AI Techniques Used:
    1. SHAP Explainable AI - Makes ML decisions transparent and educational
    2. Market Regime Detection - Time series analysis for market conditions
    3. Multi-Factor Analysis - Quantitative evaluation across multiple dimensions
    4. Correlation Analysis - Portfolio optimization for diversification
    5. Statistical Learning - Risk assessment and portfolio metrics
    6. Goal-Oriented Optimization - What return is needed to reach user's target
    """
    
    def __init__(self):
        """
        Initialize the Enhanced Stock Recommender System
        
        Sets up all the AI components and configuration needed for
        intelligent portfolio recommendations with explainable AI.
        """
        # Core system state
        self._last_recommendation_metadata = {}
        
        # Initialize AI components (lazy loading for better performance)
        self._market_detector = None
        self._factor_analyzer = None  
        self._portfolio_optimizer = None
        self._shap_explainer = None
        self._goal_calculator = None
        self._feasibility_assessor = None
        
        logger.info("✅ Enhanced Stock Recommender initialized")
    
    # ===================================================================
    # MAIN RECOMMENDATION METHOD
    # ===================================================================
    
    def recommend_stocks(self, target_value: float, timeframe: int, risk_score: float,
                        current_investment: float = 0, monthly_contribution: float = 0) -> List[str]:
        """
        AI-ENHANCED Stock Recommendation Engine with Explainable AI
        
        This is the main function that combines all AI techniques to provide
        intelligent, transparent portfolio recommendations.
        
        Process:
        1. Detect current market regime (time series analysis)
        2. Calculate goal requirements (goal-oriented optimization)
        3. Adjust risk based on market conditions (adaptive risk management)
        4. Analyze stocks using multiple factors (quantitative analysis)
        5. Optimize portfolio for diversification (correlation analysis)
        6. Generate SHAP explanations (explainable AI)
        
        Args:
            target_value: User's financial goal (e.g., £50,000)
            timeframe: Years to reach goal (e.g., 10)
            risk_score: Risk tolerance 0-100 (e.g., 65)
            current_investment: Starting amount (e.g., £5,000)
            monthly_contribution: Regular investments (e.g., £300/month)
            
        Returns:
            List of recommended stock/ETF tickers: ["VTI", "BND", "VEA", ...]
        """
        
        try:
            logger.info(f"🚀 AI-ENHANCED Portfolio Optimization Starting...")
            logger.info(f"💡 Goal: £{target_value:,} in {timeframe} years (Risk: {risk_score}/100)")
            logger.info(f"💰 Starting: £{current_investment:,} + £{monthly_contribution}/month")
            
            # === STEP 1: MARKET REGIME DETECTION ===
            market_regime = self._get_market_detector().get_cached_or_detect()
            logger.info(f"📊 Market Regime: {market_regime['regime'].upper()} (confidence: {market_regime['confidence']:.0%})")
            
            # === STEP 2: GOAL-ORIENTED ANALYSIS ===
            goal_calculator = self._get_goal_calculator()
            required_return = goal_calculator.calculate_required_return(
                target_value, current_investment, timeframe, monthly_contribution
            )
            
            feasibility_assessor = self._get_feasibility_assessor()
            goal_assessment = feasibility_assessor.assess_goal_feasibility(required_return, risk_score)
            
            # === STEP 3: RISK CATEGORY DETERMINATION ===
            base_risk_category = get_risk_category(risk_score)
            adjusted_risk_category = self._adjust_risk_for_market_regime(base_risk_category, market_regime)
            
            logger.info(f"🎯 Risk Category: {base_risk_category.replace('_', ' ').title()}")
            if adjusted_risk_category != base_risk_category:
                logger.info(f"🔄 Market-Adjusted: {adjusted_risk_category.replace('_', ' ').title()}")
            
            # === STEP 4: STOCK SELECTION AND OPTIMIZATION ===
            # Get initial stock universe
            stock_universe = self._get_stock_universe(adjusted_risk_category)
            
            # Apply factor analysis and ranking (if available)
            try:
                factor_analyzer = self._get_factor_analyzer()
                factor_weights = self._get_factor_weights_for_regime(market_regime['regime'], timeframe)
                ranked_stocks = factor_analyzer.rank_stocks_by_factors(stock_universe, factor_weights)
                
                # Select top-ranked stocks (6-10 holdings for optimal diversification)
                target_portfolio_size = min(10, max(6, len(stock_universe)))
                top_stocks = [stock for stock, score in ranked_stocks[:target_portfolio_size]]
                
                logger.info(f"🏆 Factor analysis selected {len(top_stocks)} top-ranked stocks")
                
            except ImportError:
                # Fallback if advanced modules not available
                logger.warning("⚠️ Advanced factor analysis not available, using basic selection")
                top_stocks = stock_universe[:8]  # Simple selection
            
            # === STEP 5: VALIDATE STOCKS ===
            valid_stocks = self._validate_stocks(top_stocks)
            
            if len(valid_stocks) < 4:
                logger.warning("⚠️ Too few valid stocks, using backup tickers")
                backup_stocks = BACKUP_TICKERS.get(adjusted_risk_category, BACKUP_TICKERS["moderate"])
                valid_stocks = self._validate_stocks(backup_stocks)
            
            # === STEP 6: PORTFOLIO OPTIMIZATION ===
            try:
                portfolio_optimizer = self._get_portfolio_optimizer()
                initial_weights = {stock: 1.0/len(valid_stocks) for stock in valid_stocks}
                optimized_weights = portfolio_optimizer.optimize_for_diversification(valid_stocks, initial_weights)
                portfolio_metrics = portfolio_optimizer.calculate_portfolio_metrics(valid_stocks, optimized_weights)
                
                logger.info(f"📈 Expected Return: {portfolio_metrics['expected_return']:.1%}")
                logger.info(f"📊 Portfolio Volatility: {portfolio_metrics['volatility']:.1%}")
                
            except ImportError:
                logger.warning("⚠️ Advanced portfolio optimization not available")
                optimized_weights = {stock: 1.0/len(valid_stocks) for stock in valid_stocks}
                portfolio_metrics = {"expected_return": 0.08, "volatility": 0.15, "sharpe_ratio": 0.5}
            
            # === STEP 7: STORE COMPREHENSIVE METADATA ===
            self._store_recommendation_metadata({
                "stocks": valid_stocks,
                "weights": optimized_weights,
                "market_regime": market_regime,
                "portfolio_metrics": portfolio_metrics,
                "goal_assessment": goal_assessment,
                "required_return": required_return,
                "risk_category": adjusted_risk_category,
                "ai_enhanced": True,
                "timestamp": datetime.now(),
                "methodology": "AI-enhanced multi-factor analysis with goal optimization"
            })
            
            # === FINAL RESULTS ===
            logger.info(f"✅ AI-ENHANCED Portfolio Created Successfully:")
            logger.info(f"   🎯 Goal Feasibility: {goal_assessment['feasibility_score']:.0f}%")
            logger.info(f"   🏆 Selected Stocks: {valid_stocks}")
            logger.info(f"   💡 Recommendation: {goal_assessment['recommendation']}")
            
            return valid_stocks
            
        except Exception as e:
            logger.error(f"❌ AI-enhanced recommendation failed: {str(e)}")
            # Fallback to simple recommendation for reliability
            return self._fallback_recommendation(risk_score)
    
    # ===================================================================
    # SHAP EXPLAINABLE AI METHODS
    # ===================================================================
    
    def get_shap_explanation(self, target_value: float, timeframe: int, risk_score: float,
                           current_investment: float = 0, monthly_contribution: float = 0) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for portfolio recommendations.
        
        This provides transparent explanations for why the AI made specific
        recommendations, making the decision-making process educational.
        
        Returns:
            Dict with SHAP explanation and human-readable reasoning
        """
        try:
            shap_explainer = self._get_shap_explainer()
            return shap_explainer.explain_recommendation(
                target_value, timeframe, risk_score, current_investment, monthly_contribution
            )
        except ImportError:
            logger.warning("⚠️ SHAP explainability not available")
            return {
                "error": "SHAP not available. Install with: pip install shap matplotlib seaborn",
                "portfolio_quality_score": 50,
                "explanation": "Install SHAP for transparent AI explanations"
            }
    
    def create_shap_visualization(self, target_value: float, timeframe: int, risk_score: float,
                                current_investment: float = 0, monthly_contribution: float = 0,
                                save_path: str = None) -> str:
        """
        Create visual SHAP explanation chart.
        
        Generates professional charts showing why the AI made its recommendations.
        Perfect for user education and transparency.
        """
        try:
            from ..explainable_ai.visualization import SHAPVisualizer
            visualizer = SHAPVisualizer()
            return visualizer.create_explanation_chart(
                target_value, timeframe, risk_score, current_investment, 
                monthly_contribution, save_path
            )
        except ImportError:
            return "SHAP visualization not available. Install: pip install shap matplotlib seaborn"
    
    def get_last_recommendation_explanation(self) -> Dict[str, Any]:
        """
        Get comprehensive explanation for the last portfolio recommendation.
        
        Returns all the AI analysis and explanations for the most recent
        portfolio recommendation, including SHAP values, market analysis,
        and goal feasibility assessment.
        """
        metadata = self._last_recommendation_metadata
        
        if not metadata:
            return {
                "error": "No recommendation available. Make a recommendation first.",
                "suggestion": "Call recommend_stocks() first to generate recommendations and explanations."
            }
        
        return {
            "shap_explanation": metadata.get("shap_explanation"),
            "market_regime": metadata.get("market_regime"),
            "goal_assessment": metadata.get("goal_assessment"),
            "portfolio_metrics": metadata.get("portfolio_metrics"),
            "methodology": metadata.get("methodology", "Enhanced AI portfolio optimization"),
            "ai_enhanced": metadata.get("ai_enhanced", True),
            "timestamp": metadata.get("timestamp")
        }
    
    # ===================================================================
    # GOAL-ORIENTED OPTIMIZATION METHODS
    # ===================================================================
    
    def calculate_required_return(self, target_value: float, current_investment: float, 
                                timeframe: int, monthly_contribution: float = 0) -> float:
        """
        Calculate the annual return required to reach the user's financial goal.
        
        This is the core innovation: instead of just matching portfolios to risk
        tolerance, we calculate exactly what return is needed to reach the user's
        specific goal and design the portfolio around that requirement.
        
        Args:
            target_value: The user's financial goal (e.g., £50,000)
            current_investment: Money they're starting with (e.g., £5,000)
            timeframe: Years to reach the goal (e.g., 10)
            monthly_contribution: Regular monthly investments (e.g., £300)
            
        Returns:
            Required annual return as decimal (e.g., 0.08 = 8%)
        """
        goal_calculator = self._get_goal_calculator()
        return goal_calculator.calculate_required_return(
            target_value, current_investment, timeframe, monthly_contribution
        )
    
    def assess_goal_feasibility(self, required_return: float, risk_score: float) -> Dict[str, Any]:
        """
        Assess whether the user's goal is realistic given their risk tolerance.
        
        Provides honest feedback about goal achievability and suggests adjustments.
        """
        feasibility_assessor = self._get_feasibility_assessor()
        return feasibility_assessor.assess_goal_feasibility(required_return, risk_score)
    
    # ===================================================================
    # MARKET ANALYSIS METHODS
    # ===================================================================
    
    def detect_market_regime(self) -> Dict[str, Any]:
        """
        Get current market regime analysis using AI-powered detection.
        
        Returns market condition classification and adjustment factors.
        """
        market_detector = self._get_market_detector()
        return market_detector.get_cached_or_detect()
    
    # ===================================================================
    # COMPONENT INITIALIZATION (LAZY LOADING)
    # ===================================================================
    
    def _get_market_detector(self):
        """Lazy load market regime detector."""
        if self._market_detector is None:
            try:
                from ..analysis.market_regime import MarketRegimeDetector
                self._market_detector = MarketRegimeDetector()
            except ImportError:
                logger.warning("Market regime detection not available")
                # Create mock detector
                self._market_detector = self._create_mock_market_detector()
        return self._market_detector
    
    def _get_factor_analyzer(self):
        """Lazy load factor analyzer."""
        if self._factor_analyzer is None:
            from ..analysis.factor_analysis import FactorAnalyzer
            self._factor_analyzer = FactorAnalyzer()
        return self._factor_analyzer
    
    def _get_portfolio_optimizer(self):
        """Lazy load portfolio optimizer."""
        if self._portfolio_optimizer is None:
            from ..analysis.portfolio_optimizer import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer()
        return self._portfolio_optimizer
    
    def _get_shap_explainer(self):
        """Lazy load SHAP explainer."""
        if self._shap_explainer is None:
            from ..explainable_ai.shap_explainer import SHAPExplainer
            self._shap_explainer = SHAPExplainer()
        return self._shap_explainer
    
    def _get_goal_calculator(self):
        """Lazy load goal calculator."""
        if self._goal_calculator is None:
            from ..goal_optimization.goal_calculator import GoalCalculator
            self._goal_calculator = GoalCalculator()
        return self._goal_calculator
    
    def _get_feasibility_assessor(self):
        """Lazy load feasibility assessor."""
        if self._feasibility_assessor is None:
            from ..goal_optimization.feasibility_assessor import FeasibilityAssessor
            self._feasibility_assessor = FeasibilityAssessor()
        return self._feasibility_assessor
    
    # ===================================================================
    # SUPPORTING METHODS
    # ===================================================================
    
    def _get_stock_universe(self, risk_category: str) -> List[str]:
        """Get expanded stock universe for the given risk category."""
        if not validate_risk_category(risk_category):
            risk_category = "moderate"
        
        universe = ASSET_UNIVERSES[risk_category]
        all_stocks = []
        
        # Collect all stocks from all categories in the universe
        for category, stocks in universe.items():
            if isinstance(stocks, list):
                all_stocks.extend(stocks)
        
        # Add backup tickers for more options
        backup_stocks = BACKUP_TICKERS.get(risk_category, BACKUP_TICKERS["moderate"])
        all_stocks.extend(backup_stocks)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_stocks))
    
    def _adjust_risk_for_market_regime(self, risk_category: str, market_regime: Dict) -> str:
        """
        AI-powered risk adjustment based on current market conditions.
        
        Adjusts the user's base risk tolerance based on market conditions.
        For example, becomes slightly more conservative in bear markets.
        """
        regime = market_regime['regime']
        
        # Define risk category hierarchy
        risk_levels = ["ultra_conservative", "conservative", "moderate", 
                       "moderate_aggressive", "aggressive", "ultra_aggressive"]
        
        current_index = risk_levels.index(risk_category) if risk_category in risk_levels else 2
        
        # AI-based adjustment logic
        if regime == "bear" or regime == "high_volatility":
            adjustment = -1  # Move toward more conservative in bad markets
        elif regime == "strong_bull":
            adjustment = 1   # Move slightly more aggressive in strong bull markets
        else:
            adjustment = 0   # No adjustment for neutral conditions
        
        new_index = max(0, min(len(risk_levels) - 1, current_index + adjustment))
        adjusted_category = risk_levels[new_index]
        
        if adjusted_category != risk_category:
            logger.info(f"🔄 AI Market Adjustment: {risk_category} → {adjusted_category}")
        
        return adjusted_category
    
    def _get_factor_weights_for_regime(self, regime: str, timeframe: int) -> Dict[str, float]:
        """
        AI-optimized factor weights based on market regime and timeframe.
        
        Dynamically adjusts how different factors are weighted based on
        current market conditions and the user's investment timeframe.
        """
        from .config import DEFAULT_FACTOR_WEIGHTS
        
        base_weights = DEFAULT_FACTOR_WEIGHTS.copy()
        
        # Market regime adjustments
        if regime in ["bear", "high_volatility"]:
            # Emphasize quality and low volatility in bad markets
            base_weights["quality"] += 0.15
            base_weights["volatility"] += 0.10
            base_weights["momentum"] -= 0.15
            base_weights["technical"] -= 0.10
        elif regime in ["strong_bull", "bull"]:
            # Emphasize momentum in good markets
            base_weights["momentum"] += 0.15
            base_weights["technical"] += 0.05
            base_weights["volatility"] -= 0.10
            base_weights["quality"] -= 0.10
        elif regime == "sideways":
            # Emphasize value in sideways markets
            base_weights["value"] += 0.15
            base_weights["momentum"] -= 0.15
        
        # Timeframe adjustments
        if timeframe <= 3:
            # Short timeframe: emphasize quality and low volatility
            base_weights["quality"] += 0.10
            base_weights["volatility"] += 0.10
            base_weights["momentum"] -= 0.20
        elif timeframe >= 15:
            # Long timeframe: can emphasize momentum and value
            base_weights["momentum"] += 0.10
            base_weights["value"] += 0.05
            base_weights["volatility"] -= 0.15
        
        # Normalize weights to sum to 1.0
        total_weight = sum(base_weights.values())
        return {k: v/total_weight for k, v in base_weights.items()}
    
    def _validate_stocks(self, stocks: List[str]) -> List[str]:
        """
        Validate stock symbols and filter out invalid ones.
        
        Ensures all recommended stocks are actually tradeable and have
        current market data available.
        """
        try:
            from ..utils.data_validation import StockValidator
            validator = StockValidator()
            return validator.validate_and_filter_stocks(stocks)
        except ImportError:
            # Basic validation fallback
            logger.warning("⚠️ Advanced stock validation not available, using basic validation")
            return [stock for stock in stocks if len(stock) <= 5 and stock.isalpha()]
    
    def _store_recommendation_metadata(self, metadata: Dict) -> None:
        """Store comprehensive recommendation metadata for analysis and explanations."""
        try:
            self._last_recommendation_metadata = metadata
            logger.debug("📝 Comprehensive AI metadata stored")
        except Exception as e:
            logger.warning(f"Failed to store recommendation metadata: {e}")
    
    def _fallback_recommendation(self, risk_score: float) -> List[str]:
        """
        Reliable fallback recommendation when AI enhancement fails.
        
        Provides simple but sensible portfolio recommendations based on
        risk tolerance when advanced AI systems are unavailable.
        """
        logger.warning("🔄 Using fallback recommendation method")
        
        risk_category = get_risk_category(risk_score)
        return BACKUP_TICKERS.get(risk_category, BACKUP_TICKERS["moderate"])
    
    def _create_mock_market_detector(self):
        """Create a mock market detector when the real one isn't available."""
        class MockMarketDetector:
            def get_cached_or_detect(self):
                return {
                    "regime": "neutral",
                    "confidence": 0.50,
                    "trend_score": 2.5,
                    "current_vix": 20,
                    "adjustment_factor": {"growth_tilt": 0, "defensive_tilt": 0, "volatility_adjustment": 0}
                }
        return MockMarketDetector()
    
    # ===================================================================
    # LEGACY COMPATIBILITY METHODS
    # ===================================================================
    
    def risk_score_to_category(self, risk_score: float) -> str:
        """Legacy method for backward compatibility."""
        return get_risk_category(risk_score)
    
    def validate_and_filter_stocks(self, stocks: List[str]) -> List[str]:
        """Legacy method for backward compatibility."""
        return self._validate_stocks(stocks)


# ===================================================================
# UTILITY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ===================================================================

def train_and_recommend(target_value: float, timeframe: int, risk_score: float) -> List[str]:
    """
    Main function for backward compatibility with existing code.
    
    NOW AI-ENHANCED: Uses machine learning, market regime detection,
    factor analysis, and goal-oriented optimization.
    """
    recommender = EnhancedStockRecommender()
    return recommender.recommend_stocks(target_value, timeframe, risk_score)

def get_recommendation_explanation(target_value: float, timeframe: int, risk_score: float,
                                 current_investment: float = 0, monthly_contribution: float = 0) -> Dict[str, Any]:
    """
    Get SHAP-based explanation for portfolio recommendations.
    
    Provides transparent explanations for why the AI recommended specific portfolios.
    This is the key educational feature that makes AI decisions understandable.
    """
    recommender = EnhancedStockRecommender()
    return recommender.get_shap_explanation(target_value, timeframe, risk_score, 
                                          current_investment, monthly_contribution)

def create_explanation_chart(target_value: float, timeframe: int, risk_score: float,
                           current_investment: float = 0, monthly_contribution: float = 0,
                           save_path: str = None) -> str:
    """
    Create visual SHAP explanation chart.
    
    Generates professional charts showing why the AI made its recommendations.
    Perfect for user education and transparency.
    """
    recommender = EnhancedStockRecommender()
    return recommender.create_shap_visualization(target_value, timeframe, risk_score,
                                               current_investment, monthly_contribution, save_path)

def get_market_regime() -> Dict[str, Any]:
    """
    Get current market regime analysis.
    
    Provides access to the AI's market condition analysis.
    Useful for understanding why certain recommendations were made.
    """
    recommender = EnhancedStockRecommender()
    return recommender.detect_market_regime()


# ===================================================================
# TESTING FUNCTIONS
# ===================================================================

def test_recommender_system():
    """
    Test the core recommender system functionality.
    
    This function tests basic functionality to ensure the system works
    even when advanced AI modules aren't available.
    """
    print("🧪 Testing Enhanced Stock Recommender Core System...")
    
    try:
        recommender = EnhancedStockRecommender()
        
        # Test basic recommendation
        stocks = recommender.recommend_stocks(50000, 10, 60, 5000, 300)
        print(f"✅ Basic recommendation: {stocks}")
        
        # Test goal calculation
        required_return = recommender.calculate_required_return(50000, 5000, 10, 300)
        print(f"✅ Required return calculation: {required_return:.1%}")
        
        # Test market regime detection
        market_regime = recommender.detect_market_regime()
        print(f"✅ Market regime detection: {market_regime['regime']}")
        
        # Test SHAP explanation (may fail if SHAP not installed)
        try:
            explanation = recommender.get_shap_explanation(50000, 10, 60, 5000, 300)
            if "error" not in explanation:
                print(f"✅ SHAP explanation: Quality score {explanation.get('portfolio_quality_score', 0):.1f}/100")
            else:
                print(f"⚠️ SHAP explanation: {explanation['error']}")
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
        
        print("✅ Core recommender system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Core recommender system test failed: {e}")
        return False


if __name__ == "__main__":
    test_recommender_system()