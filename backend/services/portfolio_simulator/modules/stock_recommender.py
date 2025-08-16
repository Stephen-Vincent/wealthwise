"""
Stock Recommender Module

This module provides enhanced AI-powered stock recommendations with SHAP explanations,
goal-oriented selection, and fallback mechanisms.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedStockRecommender:
    """
    Provides enhanced stock recommendations using AI analysis and goal optimization.
    
    Features:
    - Goal-oriented stock selection
    - SHAP explainable AI integration
    - Risk-based portfolio construction
    - Fallback recommendation strategies
    - Market regime awareness
    """
    
    def __init__(self):
        """Initialize the enhanced stock recommender."""
        self.wealthwise_available = self._check_wealthwise_availability()
        logger.info(f"🤖 EnhancedStockRecommender initialized (WealthWise: {'✅' if self.wealthwise_available else '❌'})")
    
    def get_recommendations(self, user_data: Dict[str, Any], risk_score: float, risk_level: str) -> Dict[str, Any]:
        """
        Get stock recommendations - main method called by portfolio simulator
        
        Args:
            user_data: Dictionary containing user information
            risk_score: Risk score (0-100) 
            risk_level: Risk level string (e.g., "Moderate Aggressive")
            
        Returns:
            Dictionary with stock recommendations
        """
        try:
            logger.info(f"🤖 Generating recommendations for risk level: {risk_level} (score: {risk_score})")
            
            # Extract relevant user information
            goal = user_data.get('goal', 'wealth building')
            timeframe = user_data.get('timeframe', 5)
            target_value = user_data.get('target_value', 10000)
            monthly = user_data.get('monthly', 100)
            lump_sum = user_data.get('lump_sum', 1000)
            
            # Use existing fallback method to get stocks
            stocks = self.get_fallback_stocks(int(risk_score))
            
            # Calculate allocations based on risk score
            allocations = self._calculate_risk_based_allocations(risk_score, len(stocks))
            
            # Create detailed recommendations structure
            recommendations = {
                'stocks_picked': [
                    {
                        'symbol': stock,
                        'allocation': allocation,
                        'explanation': self.explain_stock_selection(stock, {
                            'risk_score': risk_score,
                            'goal': goal,
                            'timeframe': timeframe
                        })
                    }
                    for stock, allocation in zip(stocks, allocations)
                ],
                'reasoning': self._generate_portfolio_reasoning(risk_score, risk_level, goal, timeframe),
                'methodology': 'Enhanced AI recommendation system with risk-based optimization',
                'risk_alignment': f"Optimized for {risk_level} risk profile",
                'confidence_score': min(95, 70 + (risk_score / 10)),
                'rebalancing_frequency': 'Quarterly' if risk_score > 60 else 'Semi-annually',
                'goal_alignment': self._assess_goal_alignment(user_data, risk_score),
                'diversification_score': self._calculate_diversification_score({'stocks_picked': [{'symbol': s} for s in stocks]}),
                'expected_return': self._estimate_expected_return(risk_score, stocks),
                'volatility_estimate': self._estimate_volatility_from_stocks(stocks)
            }
            
            logger.info(f"✅ Generated {len(stocks)} stock recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            # Return basic fallback
            return self._get_basic_fallback_recommendations_sync(user_data, risk_score, risk_level)

    def _calculate_risk_based_allocations(self, risk_score: float, num_stocks: int) -> List[float]:
        """Calculate allocations based on risk score and number of stocks"""
        
        if risk_score <= 30:  # Conservative
            if num_stocks == 5:
                return [25, 35, 20, 15, 5]  # Bond-heavy
            else:
                # Equal weight with bond bias
                base = 100 / num_stocks
                return [base] * num_stocks
                
        elif risk_score <= 50:  # Moderate
            if num_stocks == 5:
                return [30, 25, 20, 15, 10]  # Balanced
            else:
                return [100 / num_stocks] * num_stocks
                
        elif risk_score <= 70:  # Moderate Aggressive  
            if num_stocks == 5:
                return [35, 25, 20, 15, 5]  # Growth-focused
            else:
                return [100 / num_stocks] * num_stocks
                
        else:  # Aggressive
            if num_stocks == 5:
                return [40, 30, 15, 10, 5]  # High growth
            else:
                return [100 / num_stocks] * num_stocks

    def _generate_portfolio_reasoning(self, risk_score: float, risk_level: str, goal: str, timeframe: int) -> str:
        """Generate reasoning for portfolio composition"""
        
        if risk_score <= 30:
            return f"Conservative portfolio emphasizing capital preservation and steady growth for {goal} over {timeframe} years. Focus on stability with modest growth potential."
        elif risk_score <= 50:
            return f"Balanced portfolio mixing growth and stability for {goal}. Designed to provide steady returns while managing risk over {timeframe} years."
        elif risk_score <= 70:
            return f"Growth-focused portfolio targeting higher returns for {goal}. Accepts moderate volatility for enhanced growth potential over {timeframe} years."
        else:
            return f"Aggressive growth portfolio maximizing return potential for {goal}. Emphasizes capital appreciation with higher volatility tolerance over {timeframe} years."

    def _assess_goal_alignment(self, user_data: Dict[str, Any], risk_score: float) -> str:
        """Assess how well portfolio aligns with user's goal"""
        
        goal = user_data.get('goal', 'wealth building')
        timeframe = user_data.get('timeframe', 5)
        target_value = user_data.get('target_value', 10000)
        
        goal_assessments = {
            'retirement': f"Portfolio designed for long-term retirement wealth building with risk level appropriate for {timeframe}-year timeline",
            'house': f"Balanced approach for house purchase goal, managing growth needs with capital preservation for {timeframe}-year timeline", 
            'education': f"Growth-focused strategy for education funding, balancing appreciation with timeline requirements",
            'wealth building': f"Optimized for wealth accumulation over {timeframe} years with risk level matching your tolerance",
            'emergency fund': "Conservative approach prioritizing capital preservation and liquidity for emergency preparedness"
        }
        
        return goal_assessments.get(goal, f"Portfolio optimized for {goal} over {timeframe}-year investment horizon")

    def _estimate_expected_return(self, risk_score: float, stocks: List[str]) -> float:
        """Estimate expected annual return based on portfolio composition"""
        
        # Base returns by risk level
        if risk_score <= 30:
            base_return = 5.5  # Conservative
        elif risk_score <= 50:
            base_return = 7.0  # Moderate
        elif risk_score <= 70:
            base_return = 8.5  # Moderate Aggressive
        else:
            base_return = 10.0  # Aggressive
        
        # Adjust for specific holdings
        tech_allocation = sum(1 for stock in stocks if stock in ['QQQ', 'VGT', 'ARKK'])
        bond_allocation = sum(1 for stock in stocks if stock in ['BND', 'VTEB'])
        
        # Tech increases expected return
        if tech_allocation > 0:
            base_return += tech_allocation * 0.5
        
        # Bonds decrease expected return but add stability
        if bond_allocation > 0:
            base_return -= bond_allocation * 0.5
        
        return round(base_return, 1)

    def _estimate_volatility_from_stocks(self, stocks: List[str]) -> float:
        """Estimate portfolio volatility based on holdings"""
        
        base_volatility = 10.0
        
        # High volatility stocks
        high_vol_stocks = ['ARKK', 'VWO', 'VGT']
        low_vol_stocks = ['BND', 'VTEB', 'VIG']
        
        high_vol_count = sum(1 for stock in stocks if stock in high_vol_stocks)
        low_vol_count = sum(1 for stock in stocks if stock in low_vol_stocks)
        
        # Adjust volatility
        base_volatility += high_vol_count * 2.5
        base_volatility -= low_vol_count * 1.5
        
        return round(max(5.0, base_volatility), 1)

    def _get_basic_fallback_recommendations_sync(self, user_data: Dict[str, Any], risk_score: float, risk_level: str) -> Dict[str, Any]:
        """Synchronous basic fallback when everything fails"""
        
        stocks = ['VTI', 'BND', 'VEA']  # Ultra-basic portfolio
        allocations = [60, 30, 10]
        
        return {
            'stocks_picked': [
                {
                    'symbol': stock,
                    'allocation': allocation,
                    'explanation': f"{stock} - Basic allocation for balanced portfolio"
                }
                for stock, allocation in zip(stocks, allocations)
            ],
            'reasoning': f"Basic balanced portfolio for {risk_level} investor",
            'methodology': 'Emergency fallback allocation',
            'risk_alignment': f"Simplified allocation for {risk_level} profile",
            'confidence_score': 60,
            'rebalancing_frequency': 'Annually'
        }

    async def get_enhanced_recommendations(self, goal_analysis: Dict[str, Any],
                                         risk_profile: Dict[str, Any],
                                         user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get enhanced AI recommendations with SHAP explanations and goal analysis.
        
        Args:
            goal_analysis: Smart goal calculation results
            risk_profile: User risk assessment
            user_data: User investment parameters
            
        Returns:
            Enhanced recommendation results with explanations
        """
        
        try:
            logger.info("🤖 Getting enhanced AI stock recommendations")
            
            if self.wealthwise_available:
                # Use WealthWise enhanced recommendations
                return await self._get_wealthwise_recommendations(
                    goal_analysis, risk_profile, user_data
                )
            else:
                # Use enhanced fallback recommendations
                return await self._get_enhanced_fallback_recommendations(
                    goal_analysis, risk_profile, user_data
                )
                
        except Exception as e:
            logger.error(f"❌ Enhanced recommendations failed: {e}")
            return await self._get_basic_fallback_recommendations(
                risk_profile, user_data, str(e)
            )
    
    def _check_wealthwise_availability(self) -> bool:
        """Check if WealthWise system is available."""
        try:
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            from ai_models.stock_model.goal_optimization import FeasibilityAssessor
            from ai_models.stock_model.analysis import MarketRegimeDetector
            return True
        except ImportError as e:
            logger.info(f"ℹ️ WealthWise not available: {e}")
            return False
    
    async def _get_wealthwise_recommendations(self, goal_analysis: Dict[str, Any],
                                            risk_profile: Dict[str, Any],
                                            user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendations using the full WealthWise system.
        """
        
        try:
            logger.info("🎯 Using WealthWise enhanced recommendation system")
            
            # Import WealthWise components
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            from ai_models.stock_model.explainable_ai import SHAPExplainer
            from ai_models.stock_model.goal_optimization import FeasibilityAssessor
            from ai_models.stock_model.analysis import MarketRegimeDetector
            from ai_models.stock_model.utils import initialize_complete_system
            
            # Initialize system
            init_result = initialize_complete_system({
                'LOG_LEVEL': 'INFO',
                'LOG_TO_FILE': False,
                'ENABLE_PERFORMANCE_TRACKING': True
            })
            
            if not init_result['success']:
                raise Exception(f"WealthWise initialization failed: {init_result.get('error')}")
            
            # Initialize components
            recommender = EnhancedStockRecommender()
            shap_explainer = SHAPExplainer()
            feasibility_assessor = FeasibilityAssessor()
            market_detector = MarketRegimeDetector()
            
            # Extract parameters
            target_value = user_data["target_value"]
            timeframe = user_data["timeframe"]
            risk_score = risk_profile["score"]
            current_investment = user_data["lump_sum"]
            monthly_contribution = user_data["monthly"]
            
            # Assess goal feasibility using our smart goal analysis
            feasibility_assessment = feasibility_assessor.assess_goal_feasibility(
                goal_analysis["required_return"], risk_score, timeframe,
                current_investment, monthly_contribution
            )
            
            # Detect market regime
            market_regime = market_detector.detect_market_regime()
            
            # Get goal-oriented stock recommendations
            logger.info("🤖 Generating goal-oriented stock recommendations")
            recommended_stocks = recommender.recommend_stocks(
                target_value, timeframe, risk_score, 
                current_investment, monthly_contribution
            )
            
            # Generate SHAP explanations
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
            
            logger.info(f"✅ WealthWise recommendations complete: {len(recommended_stocks)} stocks selected")
            
            return {
                "stocks": recommended_stocks,
                "shap_explanation": shap_explanation,
                "feasibility_assessment": feasibility_assessment,
                "market_regime": market_regime,
                "method": "wealthwise_enhanced",
                "confidence_score": self._calculate_confidence_score(shap_explanation, feasibility_assessment)
            }
            
        except Exception as e:
            logger.error(f"❌ WealthWise recommendations failed: {e}")
            raise
    
    async def _get_enhanced_fallback_recommendations(self, goal_analysis: Dict[str, Any],
                                                   risk_profile: Dict[str, Any],
                                                   user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get enhanced recommendations using fallback algorithms.
        """
        
        try:
            logger.info("🔄 Using enhanced fallback recommendation system")
            
            # Get basic stock selection
            stocks = self.get_fallback_stocks(risk_profile["score"])
            
            # Create enhanced explanations
            enhanced_explanations = self._generate_fallback_explanations(
                stocks, goal_analysis, risk_profile, user_data
            )
            
            # Simulate market regime analysis
            market_regime = self._simulate_market_regime()
            
            # Create feasibility assessment
            feasibility_assessment = self._assess_goal_feasibility_fallback(
                goal_analysis, risk_profile, user_data
            )
            
            return {
                "stocks": stocks,
                "shap_explanation": enhanced_explanations,
                "feasibility_assessment": feasibility_assessment,
                "market_regime": market_regime,
                "method": "enhanced_fallback",
                "confidence_score": 0.75  # Moderate confidence for fallback
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced fallback recommendations failed: {e}")
            raise
    
    async def _get_basic_fallback_recommendations(self, risk_profile: Dict[str, Any],
                                                user_data: Dict[str, Any],
                                                error: str) -> Dict[str, Any]:
        """
        Get basic recommendations when all enhanced methods fail.
        """
        
        logger.warning("🔄 Using basic fallback recommendations")
        
        stocks = self.get_fallback_stocks(risk_profile["score"])
        
        return {
            "stocks": stocks,
            "method": "basic_fallback",
            "error": error,
            "confidence_score": 0.5,
            "explanation": "Using basic risk-based stock selection due to system limitations."
        }
    
    def get_fallback_stocks(self, risk_score: int) -> List[str]:
        """
        Get fallback stock recommendations based on risk profile.
        
        Args:
            risk_score: User risk score (0-100)
            
        Returns:
            List of recommended stock symbols
        """
        
        logger.info(f"📊 Selecting fallback stocks for risk score: {risk_score}")
        
        if risk_score < 35:
            # Conservative portfolio: bonds heavy, stable investments
            stocks = ["VTI", "BND", "VEA", "VTEB", "VWO"]
            logger.info("📊 Conservative allocation: bond-heavy with stable equity exposure")
            
        elif risk_score < 70:
            # Moderate portfolio: balanced approach
            stocks = ["VTI", "VEA", "VWO", "VNQ", "BND"]
            logger.info("📊 Moderate allocation: balanced equity/bond mix with diversification")
            
        else:
            # Aggressive portfolio: growth-focused
            stocks = ["VTI", "VGT", "VUG", "ARKK", "VEA"]
            logger.info("📊 Aggressive allocation: growth-focused with technology emphasis")
        
        return stocks
    
    def _generate_fallback_explanations(self, stocks: List[str], 
                                      goal_analysis: Dict[str, Any],
                                      risk_profile: Dict[str, Any],
                                      user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanations for fallback recommendations.
        """
        
        risk_score = risk_profile["score"]
        risk_label = risk_profile["label"]
        goal = user_data.get("goal", "wealth building")
        timeframe = user_data.get("timeframe", 10)
        
        # Create human-readable explanations
        explanations = {}
        
        if risk_score < 35:
            explanations["risk_approach"] = f"Conservative approach chosen due to {risk_label.lower()} risk tolerance. Emphasizing capital preservation with modest growth."
            explanations["bond_allocation"] = "Significant bond allocation (BND, VTEB) provides stability and income generation."
            explanations["equity_diversification"] = "Broad market exposure (VTI) combined with international diversification (VEA, VWO) for growth potential."
            
        elif risk_score < 70:
            explanations["balanced_approach"] = f"Balanced portfolio reflecting {risk_label.lower()} risk tolerance, mixing growth and stability."
            explanations["core_holdings"] = "Total market exposure (VTI) provides broad diversification as core holding."
            explanations["diversification_strategy"] = "International stocks (VEA, VWO) and REITs (VNQ) add diversification across asset classes."
            explanations["defensive_component"] = "Bond allocation (BND) provides defensive characteristics during market volatility."
            
        else:
            explanations["growth_focus"] = f"Growth-oriented portfolio aligned with {risk_label.lower()} risk tolerance and {timeframe}-year timeframe."
            explanations["technology_emphasis"] = "Technology sector exposure (VGT) and growth stocks (VUG) target higher returns."
            explanations["innovation_component"] = "Innovation-focused allocation (ARKK) provides exposure to disruptive technologies."
            explanations["global_exposure"] = "International diversification (VEA) balances US-heavy growth focus."
        
        # Add goal-specific explanations
        if goal_analysis.get("can_reach_with_contributions"):
            explanations["goal_alignment"] = f"Portfolio designed to exceed your {goal} goal while beating inflation, as your contributions alone nearly reach the target."
        else:
            required_return = goal_analysis.get("required_return_percent", 0)
            explanations["goal_alignment"] = f"Portfolio targets {required_return:.1f}% annual returns needed for your {goal} goal over {timeframe} years."
        
        # Create SHAP-style explanation structure
        shap_explanation = {
            "human_readable_explanation": explanations,
            "portfolio_quality_score": self._calculate_portfolio_quality_score(stocks, risk_score),
            "confidence_score": 0.8,  # High confidence in rule-based selection
            "methodology": "Risk-based allocation with goal optimization",
            "feature_importance": self._calculate_feature_importance_fallback(risk_score, goal_analysis)
        }
        
        return shap_explanation
    
    def _calculate_portfolio_quality_score(self, stocks: List[str], risk_score: int) -> float:
        """
        Calculate a quality score for the portfolio.
        """
        
        # Base score
        quality_score = 0.7
        
        # Diversification bonus
        if len(stocks) >= 5:
            quality_score += 0.15
        elif len(stocks) >= 4:
            quality_score += 0.1
        
        # Asset class diversity bonus
        has_bonds = any(ticker in ["BND", "VTEB"] for ticker in stocks)
        has_international = any(ticker in ["VEA", "VWO"] for ticker in stocks)
        has_reits = any(ticker in ["VNQ"] for ticker in stocks)
        
        if has_bonds:
            quality_score += 0.05
        if has_international:
            quality_score += 0.05
        if has_reits:
            quality_score += 0.03
        
        # Risk alignment bonus
        if risk_score < 35 and has_bonds:
            quality_score += 0.05  # Conservative investors get bonus for bonds
        elif risk_score > 70 and any(ticker in ["VGT", "VUG", "ARKK"] for ticker in stocks):
            quality_score += 0.05  # Aggressive investors get bonus for growth
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def _calculate_feature_importance_fallback(self, risk_score: int, 
                                             goal_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate feature importance for fallback explanations.
        """
        
        # Simulate SHAP-style feature importance
        importance = {
            "risk_tolerance": 0.35,
            "time_horizon": 0.20,
            "goal_feasibility": 0.25,
            "diversification_need": 0.15,
            "market_conditions": 0.05
        }
        
        # Adjust based on specific conditions
        if goal_analysis.get("can_reach_with_contributions"):
            importance["goal_feasibility"] += 0.1
            importance["risk_tolerance"] -= 0.05
        
        if risk_score > 80 or risk_score < 20:
            importance["risk_tolerance"] += 0.1
            importance["diversification_need"] -= 0.05
        
        return importance

    def _calculate_diversification_score(self, portfolio_data: Dict) -> float:
        """Calculate diversification score for portfolio"""
        stocks = portfolio_data.get('stocks_picked', [])
        if len(stocks) >= 5:
            return 0.9
        elif len(stocks) >= 3:
            return 0.75
        else:
            return 0.6
    
    def _simulate_market_regime(self) -> Dict[str, Any]:
        """
        Simulate market regime analysis for fallback mode.
        """
        
        # Create simulated market regime data
        return {
            "current_vix": 18.5,  # Moderate volatility
            "trend_score": 2.8,   # Slightly bullish
            "regime": "Normal Market",
            "volatility_regime": "Moderate",
            "trend_direction": "Slightly Bullish",
            "market_stress_level": "Low",
            "recommendation": "Normal diversification approach appropriate",
            "simulated": True
        }
    
    def _assess_goal_feasibility_fallback(self, goal_analysis: Dict[str, Any],
                                        risk_profile: Dict[str, Any],
                                        user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess goal feasibility using fallback logic.
        """
        
        required_return = goal_analysis.get("required_return_percent", 6)
        risk_score = risk_profile["score"]
        timeframe = user_data["timeframe"]
        
        # Simple feasibility scoring
        feasibility_score = 5.0  # Start with perfect score
        
        # Penalize based on required return vs risk tolerance
        if required_return > 12 and risk_score < 50:
            feasibility_score -= 2.0
        elif required_return > 8 and risk_score < 30:
            feasibility_score -= 1.5
        elif required_return > 15:
            feasibility_score -= 2.5
        
        # Bonus for longer timeframes
        if timeframe >= 15:
            feasibility_score += 0.5
        elif timeframe >= 10:
            feasibility_score += 0.3
        
        # Ensure score is within bounds
        feasibility_score = max(1.0, min(5.0, feasibility_score))
        
        # Generate assessment message
        if feasibility_score >= 4.5:
            assessment = "Highly feasible with recommended portfolio"
        elif feasibility_score >= 3.5:
            assessment = "Feasible with disciplined execution"
        elif feasibility_score >= 2.5:
            assessment = "Challenging but achievable"
        else:
            assessment = "Consider adjusting goals or increasing contributions"
        
        return {
            "feasibility_score": feasibility_score,
            "assessment": assessment,
            "risk_alignment": self._assess_risk_alignment(required_return, risk_score),
            "time_adequacy": "Adequate" if timeframe >= 10 else "Limited",
            "recommendations": self._generate_feasibility_recommendations(
                required_return, risk_score, timeframe
            )
        }
    
    def _assess_risk_alignment(self, required_return: float, risk_score: int) -> str:
        """Assess alignment between required return and risk tolerance."""
        
        if required_return <= 6 and risk_score >= 20:
            return "Well aligned"
        elif required_return <= 8 and risk_score >= 40:
            return "Well aligned"
        elif required_return <= 12 and risk_score >= 70:
            return "Well aligned"
        elif required_return > 12 and risk_score < 50:
            return "Misaligned - high return need vs low risk tolerance"
        elif required_return > 8 and risk_score < 30:
            return "Somewhat misaligned"
        else:
            return "Moderately aligned"
    
    def _generate_feasibility_recommendations(self, required_return: float,
                                            risk_score: int, timeframe: int) -> List[str]:
        """Generate recommendations for improving goal feasibility."""
        
        recommendations = []
        
        if required_return > 10 and risk_score < 50:
            recommendations.append("Consider increasing risk tolerance or extending timeframe")
        
        if required_return > 15:
            recommendations.append("Consider increasing monthly contributions")
            recommendations.append("Review if goal target is realistic")
        
        if timeframe < 10:
            recommendations.append("Consider extending investment timeframe if possible")
        
        if risk_score < 30 and required_return > 8:
            recommendations.append("Conservative investors may need to increase contributions")
        
        if not recommendations:
            recommendations.append("Goal appears well-aligned with risk tolerance and timeframe")
        
        return recommendations
    
    def _calculate_confidence_score(self, shap_explanation: Optional[Dict],
                                  feasibility_assessment: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for recommendations.
        """
        
        base_confidence = 0.8
        
        # Adjust based on SHAP availability
        if shap_explanation and shap_explanation.get("portfolio_quality_score"):
            quality_score = shap_explanation["portfolio_quality_score"]
            base_confidence = (base_confidence + quality_score) / 2
        
        # Adjust based on feasibility
        feasibility_score = feasibility_assessment.get("feasibility_score", 3.0)
        feasibility_factor = feasibility_score / 5.0
        
        # Weighted combination
        final_confidence = (base_confidence * 0.7) + (feasibility_factor * 0.3)
        
        return min(max(final_confidence, 0.3), 1.0)  # Ensure between 0.3 and 1.0
    
    def get_stock_categories(self) -> Dict[str, List[str]]:
        """
        Get categorized stock/ETF lists for different investment strategies.
        
        Returns:
            Dictionary of stock categories
        """
        
        return {
            "conservative": {
                "bonds": ["BND", "VTEB", "AGG", "TLT"],
                "dividend_stocks": ["VYM", "DVY", "VIG"],
                "large_cap_value": ["VTV", "VBR"],
                "international_bonds": ["BNDX", "VTEB"]
            },
            "moderate": {
                "total_market": ["VTI", "VTSAX", "ITOT"],
                "international": ["VEA", "VWO", "VXUS"],
                "bonds": ["BND", "AGG"],
                "reits": ["VNQ", "IYR"],
                "balanced_funds": ["VBIAX", "VTINX"]
            },
            "aggressive": {
                "growth": ["VUG", "VGT", "QQQ"],
                "small_cap": ["VB", "IWM", "VBK"],
                "emerging_markets": ["VWO", "IEMG"],
                "technology": ["VGT", "XLK", "ARKK"],
                "innovation": ["ARKK", "ARKW", "ARKG"]
            },
            "sector_specific": {
                "technology": ["VGT", "XLK", "FTEC"],
                "healthcare": ["VHT", "XLV", "IHI"],
                "financial": ["VFH", "XLF", "KBE"],
                "energy": ["VDE", "XLE", "IEO"],
                "real_estate": ["VNQ", "IYR", "REM"]
            }
        }
    
    def suggest_alternatives(self, primary_stocks: List[str], 
                           reason: str = "diversification") -> Dict[str, List[str]]:
        """
        Suggest alternative stocks/ETFs for given recommendations.
        
        Args:
            primary_stocks: List of primary recommended stocks
            reason: Reason for suggesting alternatives
            
        Returns:
            Dictionary of alternatives for each primary stock
        """
        
        alternatives = {}
        
        # Common alternatives mapping
        alt_mapping = {
            "VTI": ["ITOT", "SPTM", "VT"],
            "BND": ["AGG", "VTEB", "SCHZ"],
            "VEA": ["IEFA", "SCHF", "VTEB"],
            "VWO": ["IEMG", "SCHE", "VEU"],
            "VNQ": ["IYR", "SCHH", "RWR"],
            "VGT": ["XLK", "FTEC", "IYW"],
            "VUG": ["IVW", "SPYG", "VGT"],
            "ARKK": ["VGT", "QQQ", "IYW"],
            "VTEB": ["MUB", "PZA", "BND"]
        }
        
        for stock in primary_stocks:
            if stock in alt_mapping:
                alternatives[stock] = alt_mapping[stock]
                logger.info(f"💡 Alternatives for {stock}: {alternatives[stock]}")
        
        return alternatives
    
    def validate_recommendations(self, stocks: List[str], 
                               risk_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that stock recommendations align with user risk profile.
        
        Args:
            stocks: List of recommended stocks
            risk_profile: User risk assessment
            
        Returns:
            Validation results and warnings
        """
        
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "risk_alignment_score": 0.0
        }
        
        risk_score = risk_profile["score"]
        risk_label = risk_profile["label"]
        
        # Check risk alignment
        high_risk_stocks = ["ARKK", "VGT", "VUG", "QQQ"]
        conservative_stocks = ["BND", "VTEB", "AGG", "TLT"]
        
        high_risk_count = sum(1 for stock in stocks if stock in high_risk_stocks)
        conservative_count = sum(1 for stock in stocks if stock in conservative_stocks)
        
        # Risk alignment scoring
        if risk_score < 35:
            # Conservative investors should have more bonds
            if conservative_count == 0:
                validation_result["warnings"].append("Conservative investors should consider bond allocation")
                validation_result["suggestions"].append("Add bond ETFs like BND or VTEB")
            
            if high_risk_count > 1:
                validation_result["warnings"].append("Portfolio may be too aggressive for conservative risk tolerance")
            
            alignment_score = max(0, 1.0 - (high_risk_count * 0.3) + (conservative_count * 0.2))
            
        elif risk_score > 70:
            # Aggressive investors can handle more risk
            if conservative_count > 2:
                validation_result["warnings"].append("Portfolio may be too conservative for aggressive risk tolerance")
                validation_result["suggestions"].append("Consider reducing bond allocation")
            
            if high_risk_count == 0:
                validation_result["suggestions"].append("Consider adding growth-focused investments")
            
            alignment_score = max(0, 1.0 - (conservative_count * 0.2) + (high_risk_count * 0.15))
            
        else:
            # Moderate risk tolerance
            alignment_score = 0.8  # Generally good for balanced approach
            
            if high_risk_count > 2:
                validation_result["warnings"].append("Consider if portfolio aligns with moderate risk tolerance")
            
            if conservative_count == 0 and high_risk_count > 1:
                validation_result["suggestions"].append("Consider adding defensive assets for balance")
        
        validation_result["risk_alignment_score"] = min(max(alignment_score, 0.0), 1.0)
        
        # Overall validation
        if len(validation_result["warnings"]) > 2:
            validation_result["is_valid"] = False
        
        logger.info(f"✅ Recommendation validation: {validation_result['risk_alignment_score']:.2f} alignment score")
        
        return validation_result
    
    def explain_stock_selection(self, stock: str, context: Dict[str, Any]) -> str:
        """
        Provide detailed explanation for why a specific stock was selected.
        
        Args:
            stock: Stock symbol to explain
            context: Context including risk profile, goals, etc.
            
        Returns:
            Detailed explanation string
        """
        
        risk_score = context.get("risk_score", 50)
        goal = context.get("goal", "wealth building")
        timeframe = context.get("timeframe", 10)
        
        explanations = {
            "VTI": f"Total Stock Market ETF provides broad diversification across all US companies, perfect for {goal} with {timeframe}-year timeframe. Low fees and comprehensive market exposure.",
            
            "BND": f"Total Bond Market ETF adds stability and income to portfolio. Essential defensive component for risk management, especially important given your risk profile.",
            
            "VEA": f"Developed Markets ETF provides international diversification beyond US markets. Reduces geographic concentration risk and captures global growth opportunities.",
            
            "VWO": f"Emerging Markets ETF offers exposure to faster-growing developing economies. Higher risk but greater growth potential over {timeframe}-year period.",
            
            "VNQ": f"Real Estate ETF adds asset class diversification through REITs. Provides inflation protection and income generation for portfolio balance.",
            
            "VGT": f"Technology ETF captures growth potential of innovation sector. Aligned with aggressive growth objectives for {goal} over {timeframe} years.",
            
            "VUG": f"Growth ETF focuses on companies with strong earnings growth potential. Suitable for investors seeking capital appreciation over income.",
            
            "ARKK": f"Innovation ETF provides exposure to disruptive technologies and emerging themes. Higher risk but potential for significant long-term returns.",
            
            "VTEB": f"Tax-Exempt Bond ETF offers municipal bond exposure with tax advantages. Particularly beneficial for higher-income investors seeking after-tax returns."
        }
        
        base_explanation = explanations.get(stock, f"{stock} selected based on portfolio optimization analysis.")
        
        # Add risk context
        if risk_score < 35 and stock in ["BND", "VTEB", "VEA"]:
            base_explanation += " This conservative choice aligns well with your risk tolerance."
        elif risk_score > 70 and stock in ["VGT", "VUG", "ARKK"]:
            base_explanation += " This growth-focused selection matches your higher risk tolerance."
        
        return base_explanation