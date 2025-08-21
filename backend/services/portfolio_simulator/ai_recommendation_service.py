"""
AI-powered stock recommendation service for the Portfolio Simulator.

This module integrates with the WealthWise AI system to provide intelligent
stock recommendations based on user goals, risk tolerance, and market conditions.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import numpy as np  # ✅ Needed by SHAPDataProcessor

from .config import get_config, get_stock_metadata, get_risk_profiles
from .exceptions import AIServiceError, SHAPExplanationError
from .validators import InputValidator

logger = logging.getLogger(__name__)


class AIRecommendationService:
    """
    Provides AI-powered stock recommendations with explainable decisions.
    
    This service integrates with external AI models to provide personalized
    investment recommendations based on user profiles and market conditions.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the AI recommendation service.
        
        Args:
            validator: Input validator instance
        """
        self.config = get_config()
        self.validator = validator or InputValidator()
        self.stock_metadata = get_stock_metadata()
        self.risk_profiles = get_risk_profiles()
        
        # Try to initialize WealthWise AI system
        self.wealthwise_available = False
        self.wealthwise_components = {}
        
        if self.config.ai_service.enable_shap:
            self._initialize_wealthwise()
    
    def _initialize_wealthwise(self) -> None:
        """Initialize WealthWise AI components if available."""
        try:
            # Try to import WealthWise components
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            from ai_models.stock_model.goal_optimization import GoalCalculator, FeasibilityAssessor
            from ai_models.stock_model.analysis import MarketRegimeDetector, FactorAnalyzer
            from ai_models.stock_model.utils import initialize_complete_system
            
            # Initialize the system
            init_result = initialize_complete_system({
                'LOG_LEVEL': self.config.log_level.value,
                'LOG_TO_FILE': False,
                'ENABLE_PERFORMANCE_TRACKING': True
            })
            
            if init_result.get('success'):
                self.wealthwise_components = {
                    'recommender': EnhancedStockRecommender(),
                    'shap_explainer': SHAPExplainer(),
                    'goal_calculator': GoalCalculator(),
                    'feasibility_assessor': FeasibilityAssessor(),
                    'market_detector': MarketRegimeDetector(),
                    'factor_analyzer': FactorAnalyzer()
                }
                self.wealthwise_available = True
                logger.info("WealthWise AI system initialized successfully")
            else:
                logger.warning(f"WealthWise initialization failed: {init_result.get('error')}")
                
        except ImportError as e:
            logger.info(f"WealthWise not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize WealthWise: {e}")
    
    async def get_recommendations(
        self,
        target_value: float,
        timeframe_years: int,
        risk_score: int,
        risk_label: str,
        current_investment: float = 0,
        monthly_contribution: float = 0,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered stock recommendations.
        
        Args:
            target_value: Investment target amount
            timeframe_years: Investment timeframe
            risk_score: Risk tolerance score (0-100)
            risk_label: Risk profile label
            current_investment: Current investment amount
            monthly_contribution: Monthly contribution amount
            user_preferences: Additional user preferences
            
        Returns:
            Dictionary containing recommendations and explanations
            
        Raises:
            AIServiceError: If recommendation generation fails
        """
        try:
            logger.info(
                f"Generating AI recommendations: target=£{target_value:,.2f}, "
                f"timeframe={timeframe_years}y, risk={risk_score}"
            )
            
            # Validate inputs
            validated_target = self.validator.validate_target_value(target_value)
            validated_timeframe = self.validator.validate_timeframe(timeframe_years)
            validated_risk_score = self.validator.validate_risk_score(risk_score)
            
            if self.wealthwise_available:
                return await self._get_enhanced_recommendations(
                    validated_target, validated_timeframe, validated_risk_score,
                    risk_label, current_investment, monthly_contribution, user_preferences
                )
            else:
                return self._get_fallback_recommendations(
                    validated_target, validated_timeframe, validated_risk_score, risk_label
                )
                
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            
            logger.error(f"AI recommendation failed: {str(e)}")
            raise AIServiceError(
                f"Failed to generate recommendations: {str(e)}",
                service="ai_recommendation"
            )
    
    async def _get_enhanced_recommendations(
        self,
        target_value: float,
        timeframe_years: int,
        risk_score: int,
        risk_label: str,
        current_investment: float,
        monthly_contribution: float,
        user_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get enhanced recommendations using WealthWise AI system.
        
        Args:
            target_value: Investment target
            timeframe_years: Investment timeframe
            risk_score: Risk score
            risk_label: Risk label
            current_investment: Current investment
            monthly_contribution: Monthly contribution
            user_preferences: User preferences
            
        Returns:
            Enhanced recommendation results with SHAP explanations
        """
        try:
            recommender = self.wealthwise_components['recommender']
            goal_calculator = self.wealthwise_components['goal_calculator']
            feasibility_assessor = self.wealthwise_components['feasibility_assessor']
            market_detector = self.wealthwise_components['market_detector']
            factor_analyzer = self.wealthwise_components['factor_analyzer']
            shap_explainer = self.wealthwise_components['shap_explainer']
            
            # Step 1: Goal analysis
            goal_analysis = goal_calculator.calculate_required_return(
                target_value, current_investment, timeframe_years, monthly_contribution
            )
            
            # Step 2: Feasibility assessment
            feasibility_assessment = feasibility_assessor.assess_goal_feasibility(
                goal_analysis.get("required_return", 0), risk_score, timeframe_years,
                current_investment, monthly_contribution
            )
            
            # Step 3: Market regime detection
            market_regime = market_detector.detect_market_regime()
            
            # Step 4: Get initial stock recommendations
            initial_stocks = recommender.recommend_stocks(
                target_value, timeframe_years, risk_score,
                current_investment, monthly_contribution
            )
            
            # Step 5: Enhance with factor analysis
            candidate_stocks = self._expand_candidate_pool(
                initial_stocks, risk_score, market_regime
            )
            # Keep only tickers we have metadata for
            candidate_stocks = [s for s in candidate_stocks if s in self.stock_metadata]
            
            final_stocks: List[str] = []
            try:
                ranked_stocks = None

                # Try several compatible signatures without the bad 'market_regime' kwarg
                signature_attempts = (
                    {"risk_score": risk_score, "timeframe": timeframe_years},
                    {"risk_score": risk_score, "timeframe_years": timeframe_years},
                )
                for kwargs in signature_attempts:
                    try:
                        ranked_stocks = factor_analyzer.rank_stocks_by_factors(
                            candidate_stocks, **kwargs
                        )
                        break
                    except TypeError as e:
                        # Try next signature
                        continue

                # If both kwarg attempts failed, try positional as a last resort
                if ranked_stocks is None:
                    try:
                        ranked_stocks = factor_analyzer.rank_stocks_by_factors(
                            candidate_stocks, risk_score, timeframe_years
                        )
                    except Exception as e:
                        ranked_stocks = None

                if ranked_stocks is not None:
                    # Normalize output: could be list[str], list[tuple], or dict
                    if isinstance(ranked_stocks, dict):
                        sorted_items = sorted(
                            ranked_stocks.items(), key=lambda x: x[1], reverse=True
                        )
                        ordered = [k for k, _ in sorted_items]
                    elif ranked_stocks and isinstance(ranked_stocks[0], (list, tuple)):
                        ordered = [s[0] for s in ranked_stocks]
                    else:
                        ordered = list(ranked_stocks)

                    ordered = [s for s in ordered if s in self.stock_metadata]
                    num_stocks = min(6, len(ordered))
                    final_stocks = ordered[:num_stocks]

                # If analyzer still didn’t yield usable results, fallback to initial
                if not final_stocks:
                    raise ValueError("No usable ranking from factor analyzer")

            except Exception as factor_error:
                logger.warning(f"Factor analysis failed: {factor_error}")
                final_stocks = [s for s in initial_stocks if s in self.stock_metadata][:6]
            
            # Step 6: Generate SHAP explanations
            shap_explanation = None
            try:
                if hasattr(shap_explainer, "is_available") and not shap_explainer.is_available():
                    logger.info("Training SHAP model...")
                    # Be conservative with samples to reduce latency
                    train_kwargs = {"num_samples": 1000} if "train_shap_model" in dir(shap_explainer) else {}
                    shap_explainer.train_shap_model(**train_kwargs)
                
                if not shap_explanation and hasattr(shap_explainer, "get_shap_explanation"):
                    shap_explanation = shap_explainer.get_shap_explanation(
                        target_value, timeframe_years, risk_score,
                        current_investment, monthly_contribution,
                        market_regime.get('current_vix', 20),
                        market_regime.get('trend_score', 2.5)
                    )
                    
            except Exception as shap_error:
                logger.warning(f"SHAP explanation failed: {shap_error}")
            
            return {
                "stocks": final_stocks,
                "shap_explanation": shap_explanation,
                "goal_analysis": goal_analysis,
                "feasibility_assessment": feasibility_assessment,
                "market_regime": market_regime,
                "method": "wealthwise_enhanced",
                "confidence_score": self._calculate_confidence_score(
                    feasibility_assessment, market_regime, len(final_stocks)
                ),
                "explanations": self._generate_stock_explanations(
                    final_stocks, shap_explanation, market_regime
                )
            }
            
        except Exception as e:
            logger.error(f"Enhanced recommendation failed: {e}")
            # Fallback to basic recommendations
            return self._get_fallback_recommendations(
                target_value, timeframe_years, risk_score, risk_label
            )
    
    def _get_fallback_recommendations(
        self,
        target_value: float,
        timeframe_years: int,
        risk_score: int,
        risk_label: str
    ) -> Dict[str, Any]:
        """
        Get fallback recommendations using rule-based approach.
        
        Args:
            target_value: Investment target
            timeframe_years: Investment timeframe
            risk_score: Risk score
            risk_label: Risk label
            
        Returns:
            Basic recommendation results
        """
        logger.info("Using fallback recommendation method")
        
        # Map risk score to risk profile
        profile_key = self._map_risk_score_to_profile(risk_score)
        profile = self.risk_profiles[profile_key]
        
        # Get suggested ETFs for this risk profile
        suggested_stocks = profile.get("suggested_etfs", [])
        
        # Ensure we have at least some recommendations
        if not suggested_stocks:
            if risk_score < 35:
                suggested_stocks = ["VTI", "BND", "VEA", "VTEB", "VYM"]
            elif risk_score < 70:
                suggested_stocks = ["VTI", "VEA", "VWO", "VNQ", "BND"]
            else:
                suggested_stocks = ["VTI", "VGT", "VUG", "ARKK", "VEA"]
        
        # Limit to available stocks in our metadata
        available_stocks = [
            stock for stock in suggested_stocks 
            if stock in self.stock_metadata
        ]
        
        if not available_stocks:
            # Ultimate fallback
            available_stocks = ["VTI", "BND", "VEA"]
        
        return {
            "stocks": available_stocks[:6],  # Limit to 6 stocks
            "shap_explanation": None,
            "goal_analysis": self._calculate_basic_goal_analysis(
                target_value, timeframe_years, risk_score
            ),
            "feasibility_assessment": self._assess_basic_feasibility(
                target_value, timeframe_years, risk_score
            ),
            "market_regime": self._get_default_market_regime(),
            "method": "fallback_rule_based",
            "confidence_score": 0.6,  # Moderate confidence for rule-based
            "explanations": self._generate_basic_explanations(available_stocks[:6])
        }
    
    def _expand_candidate_pool(
        self,
        initial_stocks: List[str],
        risk_score: int,
        market_regime: Dict[str, Any]
    ) -> List[str]:
        """
        Expand the candidate stock pool based on risk profile and market conditions.
        
        Args:
            initial_stocks: Initial stock recommendations
            risk_score: Risk tolerance score
            market_regime: Current market regime data
            
        Returns:
            Expanded list of candidate stocks
        """
        candidates = set(initial_stocks)
        
        # Add stocks based on risk profile
        if risk_score < 35:  # Conservative
            candidates.update(["VTI", "BND", "VEA", "VTEB", "VWO", "AGG", "VYM", "SCHD"])
        elif risk_score < 70:  # Moderate
            candidates.update(["VTI", "VEA", "VWO", "VNQ", "BND", "VUG", "VGT", "VOO"])
        else:  # Aggressive
            candidates.update(["VTI", "VGT", "VUG", "ARKK", "VEA", "QQQ", "TQQQ"])
        
        # Add stocks based on market regime
        regime = market_regime.get('regime', 'neutral')
        if regime == 'bear':
            candidates.update(["VYM", "SCHD", "BND", "VTEB"])  # Defensive assets
        elif regime == 'bull':
            candidates.update(["VGT", "VUG", "QQQ", "ARKK"])  # Growth assets
        
        # Filter to only include stocks we have metadata for
        valid_candidates = [
            stock for stock in candidates 
            if stock in self.stock_metadata
        ]
        
        return valid_candidates
    
    def _calculate_confidence_score(
        self,
        feasibility_assessment: Dict[str, Any],
        market_regime: Dict[str, Any],
        num_stocks: int
    ) -> float:
        """
        Calculate confidence score for recommendations.
        
        Args:
            feasibility_assessment: Goal feasibility data
            market_regime: Market regime data
            num_stocks: Number of recommended stocks
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.7
        
        # Adjust based on feasibility
        feasibility_score = feasibility_assessment.get('feasibility_score', 50)
        if feasibility_score > 80:
            base_confidence += 0.1
        elif feasibility_score < 40:
            base_confidence -= 0.2
        
        # Adjust based on market volatility
        vix = market_regime.get('current_vix', 20)
        if vix < 15:  # Low volatility
            base_confidence += 0.1
        elif vix > 30:  # High volatility
            base_confidence -= 0.1
        
        # Adjust based on diversification
        if num_stocks >= 5:
            base_confidence += 0.05
        elif num_stocks < 3:
            base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _generate_stock_explanations(
        self,
        stocks: List[str],
        shap_explanation: Optional[Dict[str, Any]],
        market_regime: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate explanations for why each stock was recommended.
        
        Args:
            stocks: List of recommended stocks
            shap_explanation: SHAP explanation data
            market_regime: Market regime data
            
        Returns:
            Dictionary mapping stocks to explanation strings
        """
        explanations = {}
        
        for stock in stocks:
            stock_info = self.stock_metadata.get(stock, {})
            
            if shap_explanation and shap_explanation.get("human_readable_explanation"):
                # Use AI-generated explanations if available
                ai_explanations = shap_explanation["human_readable_explanation"]
                explanation = next(
                    (exp for key, exp in ai_explanations.items() 
                     if stock.lower() in exp.lower()), 
                    None
                )
                
                if explanation:
                    explanations[stock] = explanation[:150] + "..." if len(explanation) > 150 else explanation
                    continue
            
            # Fallback to rule-based explanations
            category = stock_info.get('category', 'equity')
            stock_risk = stock_info.get('risk_score', 15)
            description = stock_info.get('description', f'{stock} investment')
            
            if category == 'bond':
                explanations[stock] = f"Selected for stability and income generation. {description}"
            elif category == 'equity_dividend':
                explanations[stock] = f"Chosen for dividend income and moderate growth. {description}"
            elif stock_risk > 30:
                explanations[stock] = f"Included for growth potential despite higher volatility. {description}"
            else:
                explanations[stock] = f"Selected for balanced risk-return profile. {description}"
        
        return explanations
    
    def _generate_basic_explanations(self, stocks: List[str]) -> Dict[str, str]:
        """Generate basic explanations for fallback recommendations."""
        explanations = {}
        
        for stock in stocks:
            stock_info = self.stock_metadata.get(stock, {})
            description = stock_info.get('description', f'{stock} investment')
            explanations[stock] = f"Selected based on risk profile matching. {description}"
        
        return explanations
    
    def _calculate_basic_goal_analysis(
        self,
        target_value: float,
        timeframe_years: int,
        risk_score: int
    ) -> Dict[str, Any]:
        """Calculate basic goal analysis without AI."""
        # Simple calculation of required return
        # Assumes some monthly contribution for calculation
        assumed_monthly = target_value / (timeframe_years * 24)  # Rough estimate
        total_contributions = assumed_monthly * 12 * timeframe_years
        
        if total_contributions > 0:
            required_growth = target_value - total_contributions
            required_return_pct = (required_growth / total_contributions) * 100 / timeframe_years
        else:
            required_return_pct = 8.0  # Default assumption
        
        return {
            "required_return_percent": max(0, required_return_pct),
            "estimated_contributions": total_contributions,
            "required_growth": max(0, target_value - total_contributions),
            "method": "basic_calculation"
        }
    
    def _assess_basic_feasibility(
        self,
        target_value: float,
        timeframe_years: int,
        risk_score: int
    ) -> Dict[str, Any]:
        """Assess basic feasibility without AI."""
        # Simple heuristic-based feasibility assessment
        goal_analysis = self._calculate_basic_goal_analysis(target_value, timeframe_years, risk_score)
        required_return = goal_analysis.get("required_return_percent", 8)
        
        # Feasibility based on required return and risk tolerance
        if required_return <= 5:
            feasibility_score = 90
            risk_assessment = "low"
        elif required_return <= 10:
            feasibility_score = 75
            risk_assessment = "moderate"
        elif required_return <= 15:
            feasibility_score = 60 if risk_score > 50 else 40
            risk_assessment = "high"
        else:
            feasibility_score = 30 if risk_score > 70 else 15
            risk_assessment = "very_high"
        
        # Adjust for timeframe
        if timeframe_years < 5:
            feasibility_score -= 15
        elif timeframe_years > 15:
            feasibility_score += 10
        
        feasibility_score = max(10, min(95, feasibility_score))
        
        return {
            "feasibility_score": feasibility_score,
            "risk_assessment": risk_assessment,
            "recommendations": {
                "primary": f"Goal is {'achievable' if feasibility_score > 60 else 'challenging'} with current parameters",
                "secondary": "Consider adjusting timeframe or contributions if needed"
            },
            "method": "heuristic_assessment"
        }
    
    def _get_default_market_regime(self) -> Dict[str, Any]:
        """Get default market regime data when AI is not available."""
        return {
            "regime": "neutral",
            "current_vix": 20.0,
            "trend_score": 2.5,
            "returns_3m": 0.05,
            "confidence": 0.5,
            "method": "default_values"
        }
    
    def _map_risk_score_to_profile(self, risk_score: int) -> str:
        """Map numerical risk score to risk profile name."""
        if risk_score < 35:
            return "conservative"
        elif risk_score < 70:
            return "moderate"
        else:
            return "aggressive"
    
    def get_stock_explanation(self, ticker: str, recommendation_result: Dict[str, Any]) -> str:
        """
        Get explanation for a specific stock recommendation.
        
        Args:
            ticker: Stock ticker symbol
            recommendation_result: Result from get_recommendations()
            
        Returns:
            Human-readable explanation for the stock selection
        """
        explanations = recommendation_result.get("explanations", {})
        
        if ticker in explanations:
            return explanations[ticker]
        
        # Fallback explanation
        method = recommendation_result.get("method", "unknown")
        if method == "fallback_rule_based":
            return f"{ticker} selected based on risk profile matching"
        else:
            return f"{ticker} recommended by AI analysis for your goals and risk profile"


class SHAPDataProcessor:
    """
    Processes and cleans SHAP explanation data for visualization and analysis.
    
    This class handles the complex task of converting SHAP model outputs into
    formats suitable for visualization and human interpretation.
    """
    
    def __init__(self):
        """Initialize the SHAP data processor."""
        self.feature_name_mapping = {
            "risk_score": "Risk Tolerance",
            "target_value_log": "Investment Goal",
            "timeframe": "Time Horizon",
            "required_return": "Required Growth Rate",
            "monthly_contribution": "Monthly Investment",
            "market_volatility": "Market Volatility",
            "market_trend_score": "Market Trend"
        }
        
        self.feature_descriptions = {
            "risk_score": "Your comfort level with investment risk and volatility",
            "target_value_log": "The financial goal you want to achieve",
            "timeframe": "How long you have to invest and reach your goal",
            "required_return": "The annual growth rate needed to reach your target",
            "monthly_contribution": "Amount you can invest each month",
            "market_volatility": "Current market uncertainty and volatility levels",
            "market_trend_score": "Whether markets are trending up or down"
        }
    
    def clean_shap_explanation(self, raw_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and process raw SHAP explanation data.
        
        Args:
            raw_explanation: Raw SHAP explanation from the model
            
        Returns:
            Cleaned and processed SHAP explanation
            
        Raises:
            SHAPExplanationError: If processing fails
        """
        try:
            cleaned = {}
            
            # Process feature contributions
            if "feature_contributions" in raw_explanation:
                cleaned["feature_contributions"] = self._clean_feature_contributions(
                    raw_explanation["feature_contributions"]
                )
            
            # Process base value
            cleaned["base_value"] = self._clean_numeric_value(
                raw_explanation.get("base_value", 50.0)
            )
            
            # Process portfolio quality score
            cleaned["portfolio_quality_score"] = self._clean_numeric_value(
                raw_explanation.get("portfolio_quality_score", 75.0)
            )
            
            # Generate human-readable explanations
            cleaned["human_readable_explanation"] = self._generate_human_explanations(
                cleaned.get("feature_contributions", {}),
                cleaned.get("portfolio_quality_score", 75.0)
            )
            
            # Add metadata
            cleaned["processed_timestamp"] = datetime.now().isoformat()
            cleaned["processor_version"] = "1.0"
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to clean SHAP explanation: {e}")
            raise SHAPExplanationError(
                f"SHAP explanation processing failed: {str(e)}"
            )
    
    def _clean_feature_contributions(self, contributions: Dict[str, Any]) -> Dict[str, float]:
        """Clean feature contribution values."""
        cleaned_contributions = {}
        
        for feature, value in contributions.items():
            try:
                # Handle various data types that might come from SHAP
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    # Array-like value, take the first element
                    numeric_value = float(list(value)[0]) if len(list(value)) > 0 else 0.0
                else:
                    # Scalar value
                    numeric_value = float(value)
                
                # Validate the value
                if not np.isfinite(numeric_value):
                    numeric_value = 0.0
                
                cleaned_contributions[str(feature)] = round(numeric_value, 4)
                
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Could not process feature contribution for {feature}: {value}")
                cleaned_contributions[str(feature)] = 0.0
        
        return cleaned_contributions
    
    def _clean_numeric_value(self, value: Any) -> float:
        """Clean a numeric value from SHAP output."""
        try:
            if hasattr(value, '__iter__') and not isinstance(value, str):
                # Array-like value
                numeric_value = float(list(value)[0]) if len(list(value)) > 0 else 50.0
            else:
                # Scalar value
                numeric_value = float(value)
            
            # Validate and constrain
            if not np.isfinite(numeric_value):
                numeric_value = 50.0
            
            return round(numeric_value, 2)
            
        except (ValueError, TypeError, IndexError):
            return 50.0
    
    def _generate_human_explanations(
        self, 
        contributions: Dict[str, float], 
        quality_score: float
    ) -> Dict[str, str]:
        """Generate human-readable explanations from SHAP contributions."""
        explanations = {}
        
        # Sort contributions by absolute impact
        sorted_contributions = sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        for feature, contribution in sorted_contributions[:5]:  # Top 5 factors
            feature_name = self.feature_name_mapping.get(feature, feature.replace("_", " ").title())
            
            if abs(contribution) < 0.1:  # Skip very small contributions
                continue
            
            if contribution > 0:
                impact = "increased"
                direction = "higher"
            else:
                impact = "decreased"
                direction = "lower"
                contribution = abs(contribution)
            
            if feature == "risk_score":
                explanations[feature] = (
                    f"Your risk tolerance {impact} the AI's confidence in growth-oriented investments. "
                    f"This led to {direction} allocation in volatile but potentially rewarding assets."
                )
            elif feature == "timeframe":
                explanations[feature] = (
                    f"Your investment timeframe {impact} the portfolio's risk level. "
                    f"{'Longer' if contribution > 0 else 'Shorter'} time horizons allow for "
                    f"{'more' if contribution > 0 else 'less'} aggressive strategies."
                )
            elif feature == "target_value_log":
                explanations[feature] = (
                    f"Your financial goal {impact} the required growth rate. "
                    f"This influenced the selection toward {'higher-return' if contribution > 0 else 'safer'} investments."
                )
            elif feature == "required_return":
                explanations[feature] = (
                    f"The required return rate {impact} portfolio aggressiveness. "
                    f"{'Higher' if contribution > 0 else 'Lower'} required returns necessitate "
                    f"{'riskier' if contribution > 0 else 'more conservative'} asset selection."
                )
            elif feature == "monthly_contribution":
                explanations[feature] = (
                    f"Your monthly investment amount {impact} the portfolio strategy. "
                    f"{'Regular' if contribution > 0 else 'Limited'} contributions allow for "
                    f"{'more' if contribution > 0 else 'less'} diversified approaches."
                )
            else:
                explanations[feature] = (
                    f"{feature_name} {impact} the portfolio recommendation by {contribution:.2f} points."
                )
        
        # Add overall assessment
        if quality_score > 80:
            explanations["overall"] = (
                "The AI has high confidence in this portfolio matching your goals and risk tolerance."
            )
        elif quality_score > 60:
            explanations["overall"] = (
                "The AI believes this portfolio is well-suited to your profile with some trade-offs."
            )
        else:
            explanations["overall"] = (
                "The AI found this portfolio challenging to optimize for your specific requirements."
            )
        
        return explanations