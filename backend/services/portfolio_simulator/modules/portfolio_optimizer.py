"""
Portfolio Optimizer Module

This module handles portfolio weight calculation and optimization
with enhanced algorithms and risk management features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')  # Suppress optimization warnings

class PortfolioOptimizer:
    """
    Optimizes portfolio weights using various strategies.
    
    Features:
    - Risk-based allocation
    - Mean reversion optimization
    - Correlation-based diversification
    - Minimum variance optimization
    - Maximum Sharpe ratio optimization
    - Risk parity allocation
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer."""
        logger.info("⚖️ PortfolioOptimizer initialized")
    
    def calculate_enhanced_weights(self, data: pd.DataFrame, risk_score: int, 
                                 recommendation_result: Dict[str, Any]) -> np.ndarray:
        """
        Calculate enhanced portfolio weights using multiple optimization methods.
        
        Args:
            data: Historical stock price data
            risk_score: User risk score (0-100)
            recommendation_result: Enhanced AI recommendation results
            
        Returns:
            Optimized portfolio weights
        """
        
        try:
            logger.info(f"⚖️ Calculating enhanced weights for {len(data.columns)} assets")
            
            # Check if WealthWise optimization is available
            if (recommendation_result.get("method") == "wealthwise_enhanced" and 
                self._has_wealthwise_optimization()):
                
                logger.info("🤖 Using WealthWise correlation-based optimization")
                return self._calculate_wealthwise_weights(data, recommendation_result)
            
            # Use our enhanced optimization methods
            optimization_method = self._select_optimization_method(risk_score, len(data.columns))
            
            if optimization_method == "min_variance":
                weights = self._calculate_minimum_variance_weights(data)
            elif optimization_method == "max_sharpe":
                weights = self._calculate_max_sharpe_weights(data)
            elif optimization_method == "risk_parity":
                weights = self._calculate_risk_parity_weights(data)
            elif optimization_method == "correlation_based":
                weights = self._calculate_correlation_based_weights(data)
            else:
                # Default to risk-based allocation
                weights = self._calculate_risk_based_weights(data, risk_score)
            
            # Apply constraints and validation
            weights = self._apply_weight_constraints(weights, risk_score)
            weights = self._validate_weights(weights)
            
            logger.info(f"✅ Enhanced weights calculated using {optimization_method}")
            return weights
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced optimization failed: {e}. Using basic allocation.")
            return self.calculate_basic_weights(data, risk_score)
    
    def calculate_basic_weights(self, data: pd.DataFrame, risk_score: int) -> np.ndarray:
        """
        Calculate basic portfolio weights using simple allocation rules.
        
        Args:
            data: Historical stock price data
            risk_score: User risk score (0-100)
            
        Returns:
            Basic portfolio weights
        """
        
        try:
            num_assets = len(data.columns)
            logger.info(f"⚖️ Calculating basic weights for {num_assets} assets (risk score: {risk_score})")
            
            if risk_score < 35:
                # Conservative allocation
                weights = self._calculate_conservative_weights(num_assets)
                logger.info("📊 Using conservative allocation")
                
            elif risk_score < 70:
                # Moderate allocation with slight bias
                weights = self._calculate_moderate_weights(num_assets)
                logger.info("📊 Using moderate allocation")
                
            else:
                # Aggressive allocation with concentration
                weights = self._calculate_aggressive_weights(num_assets)
                logger.info("📊 Using aggressive allocation")
            
            # Ensure weights are valid
            weights = self._validate_weights(weights)
            return weights
            
        except Exception as e:
            logger.error(f"❌ Error calculating basic weights: {e}")
            # Ultimate fallback: equal weights
            return np.array([1.0 / len(data.columns)] * len(data.columns))
    
    def _has_wealthwise_optimization(self) -> bool:
        """Check if WealthWise optimization is available."""
        try:
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            return True
        except ImportError:
            return False
    
    def _calculate_wealthwise_weights(self, data: pd.DataFrame, 
                                    recommendation_result: Dict[str, Any]) -> np.ndarray:
        """
        Calculate weights using WealthWise correlation-based optimization.
        """
        
        try:
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            
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
            
            return weights_array
            
        except Exception as e:
            logger.warning(f"⚠️ WealthWise optimization failed: {e}")
            raise
    
    def _select_optimization_method(self, risk_score: int, num_assets: int) -> str:
        """
        Select the most appropriate optimization method based on user profile.
        """
        
        # For very conservative investors, use minimum variance
        if risk_score < 25:
            return "min_variance"
        
        # For conservative investors with few assets, use equal weighting
        elif risk_score < 35 and num_assets <= 4:
            return "equal_weight"
        
        # For moderate investors, use correlation-based diversification
        elif risk_score < 70:
            return "correlation_based"
        
        # For aggressive investors with enough assets, try maximum Sharpe
        elif risk_score >= 70 and num_assets >= 4:
            return "max_sharpe"
        
        # Default to risk-based allocation
        else:
            return "risk_based"
    
    def _calculate_minimum_variance_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate minimum variance portfolio weights.
        """
        
        try:
            logger.info("📊 Calculating minimum variance weights")
            
            # Calculate returns and covariance matrix
            returns = data.pct_change().dropna()
            cov_matrix = returns.cov().values
            
            num_assets = len(data.columns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            bounds = tuple((0.05, 0.4) for _ in range(num_assets))  # 5% min, 40% max
            
            # Initial guess
            x0 = np.array([1.0 / num_assets] * num_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                logger.warning("⚠️ Minimum variance optimization failed, using equal weights")
                return np.array([1.0 / num_assets] * num_assets)
                
        except Exception as e:
            logger.warning(f"⚠️ Minimum variance calculation failed: {e}")
            raise
    
    def _calculate_max_sharpe_weights(self, data: pd.DataFrame, 
                                    risk_free_rate: float = 0.02) -> np.ndarray:
        """
        Calculate maximum Sharpe ratio portfolio weights.
        """
        
        try:
            logger.info("📊 Calculating maximum Sharpe ratio weights")
            
            # Calculate returns and statistics
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov().values * 252  # Annualized
            
            num_assets = len(data.columns)
            
            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(mean_returns.values * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if portfolio_std == 0:
                    return -np.inf
                
                sharpe = (portfolio_return - risk_free_rate) / portfolio_std
                return -sharpe  # Minimize negative Sharpe
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            bounds = tuple((0.05, 0.5) for _ in range(num_assets))  # 5% min, 50% max
            
            # Initial guess
            x0 = np.array([1.0 / num_assets] * num_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                logger.warning("⚠️ Max Sharpe optimization failed, using equal weights")
                return np.array([1.0 / num_assets] * num_assets)
                
        except Exception as e:
            logger.warning(f"⚠️ Max Sharpe calculation failed: {e}")
            raise
    
    def _calculate_risk_parity_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity (equal risk contribution) weights.
        """
        
        try:
            logger.info("📊 Calculating risk parity weights")
            
            # Calculate returns and covariance matrix
            returns = data.pct_change().dropna()
            cov_matrix = returns.cov().values
            
            num_assets = len(data.columns)
            
            # Objective function: minimize sum of squared differences in risk contributions
            def objective(weights):
                portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                
                if portfolio_var == 0:
                    return np.inf
                
                # Risk contributions
                marginal_contribs = np.dot(cov_matrix, weights)
                risk_contribs = weights * marginal_contribs / portfolio_var
                
                # Target: equal risk contribution (1/n for each asset)
                target_risk = 1.0 / num_assets
                
                # Sum of squared deviations from equal risk
                return np.sum((risk_contribs - target_risk) ** 2)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            bounds = tuple((0.01, 0.6) for _ in range(num_assets))  # 1% min, 60% max
            
            # Initial guess: inverse volatility weights
            volatilities = np.sqrt(np.diag(cov_matrix))
            x0 = (1.0 / volatilities) / np.sum(1.0 / volatilities)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                logger.warning("⚠️ Risk parity optimization failed, using inverse volatility weights")
                return x0
                
        except Exception as e:
            logger.warning(f"⚠️ Risk parity calculation failed: {e}")
            raise
    
    def _calculate_correlation_based_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate weights based on correlation diversification.
        """
        
        try:
            logger.info("📊 Calculating correlation-based diversification weights")
            
            # Calculate correlation matrix
            returns = data.pct_change().dropna()
            corr_matrix = returns.corr().values
            
            num_assets = len(data.columns)
            
            # Calculate diversification scores
            diversification_scores = []
            
            for i in range(num_assets):
                # Average correlation with other assets (lower is better for diversification)
                avg_corr = np.mean([corr_matrix[i, j] for j in range(num_assets) if i != j])
                diversification_score = 1.0 - abs(avg_corr)  # Higher score = more diversifying
                diversification_scores.append(diversification_score)
            
            # Convert to weights (normalize)
            diversification_scores = np.array(diversification_scores)
            weights = diversification_scores / np.sum(diversification_scores)
            
            # Apply smoothing to avoid extreme concentrations
            min_weight = 0.05
            max_weight = 0.4
            
            weights = np.maximum(weights, min_weight)
            weights = np.minimum(weights, max_weight)
            
            # Renormalize
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.warning(f"⚠️ Correlation-based calculation failed: {e}")
            raise
    
    def _calculate_conservative_weights(self, num_assets: int) -> np.ndarray:
        """Calculate conservative allocation weights."""
        
        if num_assets <= 3:
            # Simple equal weighting for few assets
            return np.array([1.0 / num_assets] * num_assets)
        elif num_assets == 4:
            # Typical conservative allocation: bonds heavy
            return np.array([0.3, 0.4, 0.2, 0.1])  # Example: stocks, bonds, intl, other
        elif num_assets == 5:
            return np.array([0.25, 0.35, 0.2, 0.15, 0.05])
        else:
            # Equal weights for many assets (conservative approach)
            return np.array([1.0 / num_assets] * num_assets)
    
    def _calculate_moderate_weights(self, num_assets: int) -> np.ndarray:
        """Calculate moderate allocation weights."""
        
        if num_assets <= 3:
            return np.array([1.0 / num_assets] * num_assets)
        elif num_assets == 4:
            # Moderate allocation: balanced
            return np.array([0.4, 0.3, 0.2, 0.1])
        elif num_assets == 5:
            return np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        else:
            # Slight bias toward first few assets
            weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1][:num_assets])
            if len(weights) < num_assets:
                # Add equal weights for remaining assets
                remaining = num_assets - len(weights)
                additional_weights = np.array([0.05] * remaining)
                weights = np.concatenate([weights, additional_weights])
            
            # Normalize
            return weights / np.sum(weights)
    
    def _calculate_aggressive_weights(self, num_assets: int) -> np.ndarray:
        """Calculate aggressive allocation weights."""
        
        if num_assets <= 3:
            return np.array([1.0 / num_assets] * num_assets)
        elif num_assets == 4:
            # Aggressive allocation: growth heavy
            return np.array([0.5, 0.3, 0.15, 0.05])
        elif num_assets == 5:
            return np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        else:
            # Concentrated in first few assets
            weights = np.array([0.4, 0.3, 0.2, 0.1][:num_assets])
            if len(weights) < num_assets:
                remaining = num_assets - len(weights)
                additional_weights = np.array([0.05] * remaining)
                weights = np.concatenate([weights, additional_weights])
            
            # Normalize
            return weights / np.sum(weights)
    
    def _calculate_risk_based_weights(self, data: pd.DataFrame, risk_score: int) -> np.ndarray:
        """
        Calculate weights based on user risk tolerance and asset volatility.
        """
        
        try:
            logger.info("📊 Calculating risk-based weights")
            
            # Calculate asset volatilities
            returns = data.pct_change().dropna()
            volatilities = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Risk tolerance factor (0 = very conservative, 1 = very aggressive)
            risk_tolerance = risk_score / 100.0
            
            if risk_tolerance < 0.3:
                # Conservative: prefer lower volatility assets
                weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)
            elif risk_tolerance > 0.7:
                # Aggressive: can handle higher volatility
                weights = volatilities / np.sum(volatilities)
            else:
                # Moderate: balanced approach
                inv_vol_weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)
                vol_weights = volatilities / np.sum(volatilities)
                # Blend based on risk tolerance
                blend_factor = (risk_tolerance - 0.3) / 0.4  # Scale to 0-1
                weights = (1 - blend_factor) * inv_vol_weights + blend_factor * vol_weights
            
            return weights
            
        except Exception as e:
            logger.warning(f"⚠️ Risk-based calculation failed: {e}")
            # Fallback to equal weights
            return np.array([1.0 / len(data.columns)] * len(data.columns))
    
    def _apply_weight_constraints(self, weights: np.ndarray, risk_score: int) -> np.ndarray:
        """
        Apply weight constraints based on risk profile.
        """
        
        try:
            # Define constraints based on risk score
            if risk_score < 35:
                min_weight, max_weight = 0.1, 0.4   # Conservative: well-diversified
            elif risk_score < 70:
                min_weight, max_weight = 0.05, 0.5  # Moderate: some concentration allowed
            else:
                min_weight, max_weight = 0.02, 0.6  # Aggressive: higher concentration allowed
            
            # Apply minimum weights
            weights = np.maximum(weights, min_weight)
            
            # Apply maximum weights
            weights = np.minimum(weights, max_weight)
            
            # Renormalize to sum to 1
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.warning(f"⚠️ Error applying constraints: {e}")
            return weights
    
    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Validate and fix portfolio weights.
        """
        
        try:
            # Ensure weights are not NaN or infinite
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                logger.warning("⚠️ Invalid weights detected, using equal weights")
                return np.array([1.0 / len(weights)] * len(weights))
            
            # Ensure weights are positive
            weights = np.maximum(weights, 0.001)  # Minimum 0.1%
            
            # Ensure weights sum to 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-6:
                weights = weights / weight_sum
            
            # Final validation
            if abs(np.sum(weights) - 1.0) > 1e-6:
                logger.warning("⚠️ Weight normalization failed, using equal weights")
                return np.array([1.0 / len(weights)] * len(weights))
            
            return weights
            
        except Exception as e:
            logger.error(f"❌ Weight validation failed: {e}")
            return np.array([1.0 / len(weights)] * len(weights))
    
    def analyze_portfolio_risk(self, data: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """
        Analyze risk characteristics of the portfolio.
        
        Args:
            data: Historical stock price data
            weights: Portfolio weights
            
        Returns:
            Risk analysis results
        """
        
        try:
            logger.info("📊 Analyzing portfolio risk characteristics")
            
            # Calculate returns
            returns = data.pct_change().dropna()
            portfolio_returns = returns.dot(weights)
            
            # Risk metrics
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_return = portfolio_returns.mean() * 252
            
            # Downside metrics
            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5)
            
            # Sharpe ratio
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Sortino ratio
            sortino_ratio = (portfolio_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
            
            # Diversification metrics
            correlation_matrix = returns.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Concentration risk (Herfindahl index)
            concentration_index = np.sum(weights ** 2)
            
            risk_analysis = {
                "return_metrics": {
                    "expected_annual_return": round(portfolio_return * 100, 2),
                    "annual_volatility": round(portfolio_volatility * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "sortino_ratio": round(sortino_ratio, 2)
                },
                "risk_metrics": {
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "value_at_risk_95": round(var_95 * 100, 2),
                    "downside_volatility": round(downside_volatility * 100, 2)
                },
                "diversification_metrics": {
                    "average_correlation": round(avg_correlation, 3),
                    "concentration_index": round(concentration_index, 3),
                    "effective_number_of_assets": round(1 / concentration_index, 1)
                },
                "risk_assessment": self._assess_risk_level(portfolio_volatility, max_drawdown, concentration_index)
            }
            
            logger.info("✅ Portfolio risk analysis completed")
            return risk_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing portfolio risk: {e}")
            return {"error": str(e)}
    
    def _assess_risk_level(self, volatility: float, max_drawdown: float, 
                          concentration: float) -> Dict[str, Any]:
        """
        Assess the overall risk level of the portfolio.
        """
        
        risk_score = 0
        risk_factors = []
        
        # Volatility assessment
        if volatility > 0.25:  # >25% volatility
            risk_score += 3
            risk_factors.append("High volatility")
        elif volatility > 0.15:  # 15-25% volatility
            risk_score += 2
            risk_factors.append("Moderate volatility")
        else:
            risk_score += 1
            risk_factors.append("Low volatility")
        
        # Drawdown assessment
        if abs(max_drawdown) > 0.4:  # >40% max drawdown
            risk_score += 3
            risk_factors.append("High drawdown risk")
        elif abs(max_drawdown) > 0.2:  # 20-40% max drawdown
            risk_score += 2
            risk_factors.append("Moderate drawdown risk")
        else:
            risk_score += 1
            risk_factors.append("Low drawdown risk")
        
        # Concentration assessment
        if concentration > 0.5:  # Highly concentrated
            risk_score += 3
            risk_factors.append("High concentration")
        elif concentration > 0.3:  # Moderately concentrated
            risk_score += 2
            risk_factors.append("Moderate concentration")
        else:
            risk_score += 1
            risk_factors.append("Well diversified")
        
        # Overall risk level
        if risk_score <= 4:
            risk_level = "Low"
        elif risk_score <= 6:
            risk_level = "Moderate"
        elif risk_score <= 8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return {
            "overall_risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level, risk_factors)
        }
    
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """
        Get risk management recommendations.
        """
        
        recommendations = {
            "Low": "Your portfolio has conservative risk characteristics. Consider if this aligns with your return expectations and time horizon.",
            "Moderate": "Your portfolio has balanced risk characteristics. Monitor performance and rebalance periodically.",
            "High": "Your portfolio has elevated risk. Ensure this aligns with your risk tolerance and consider diversification.",
            "Very High": "Your portfolio has very high risk characteristics. Consider reducing concentration and volatility."
        }
        
        base_rec = recommendations.get(risk_level, "Monitor portfolio risk regularly.")
        
        # Add specific recommendations based on risk factors
        specific_recs = []
        if "High concentration" in risk_factors:
            specific_recs.append("Consider adding more assets to improve diversification.")
        if "High volatility" in risk_factors:
            specific_recs.append("Consider adding defensive assets to reduce volatility.")
        if "High drawdown risk" in risk_factors:
            specific_recs.append("Consider implementing stop-loss or rebalancing strategies.")
        
        if specific_recs:
            return base_rec + " " + " ".join(specific_recs)
        
        return base_rec
    
    def optimize_for_target_return(self, data: pd.DataFrame, target_return: float) -> np.ndarray:
        """
        Optimize portfolio for a specific target return.
        
        Args:
            data: Historical stock price data
            target_return: Target annual return (e.g., 0.08 for 8%)
            
        Returns:
            Optimized weights for target return
        """
        
        try:
            logger.info(f"📊 Optimizing for target return: {target_return:.1%}")
            
            # Calculate returns and covariance matrix
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov().values * 252  # Annualized
            
            num_assets = len(data.columns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(mean_returns.values * x) - target_return}  # Target return
            ]
            
            bounds = tuple((0.01, 0.6) for _ in range(num_assets))  # 1% min, 60% max
            
            # Initial guess
            x0 = np.array([1.0 / num_assets] * num_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                logger.info(f"✅ Target return optimization successful")
                return result.x
            else:
                logger.warning(f"⚠️ Target return optimization failed: {result.message}")
                return self._calculate_risk_based_weights(data, 50)  # Fallback to moderate allocation
                
        except Exception as e:
            logger.error(f"❌ Target return optimization failed: {e}")
            return np.array([1.0 / len(data.columns)] * len(data.columns))
    
    def rebalance_portfolio(self, current_weights: np.ndarray, target_weights: np.ndarray,
                          threshold: float = 0.05) -> Dict[str, Any]:
        """
        Determine if portfolio needs rebalancing and calculate trades.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalancing threshold (e.g., 0.05 for 5%)
            
        Returns:
            Rebalancing analysis and recommendations
        """
        
        try:
            # Calculate weight differences
            weight_diffs = np.abs(current_weights - target_weights)
            max_deviation = np.max(weight_diffs)
            
            needs_rebalancing = max_deviation > threshold
            
            rebalancing_analysis = {
                "needs_rebalancing": needs_rebalancing,
                "max_deviation": round(max_deviation * 100, 2),
                "threshold_percent": threshold * 100,
                "deviations": [round(diff * 100, 2) for diff in weight_diffs],
                "trades_required": []
            }
            
            if needs_rebalancing:
                # Calculate required trades
                total_value = 100000  # Assume $100k portfolio for percentage calculations
                
                for i, (current, target, diff) in enumerate(zip(current_weights, target_weights, weight_diffs)):
                    if diff > threshold:
                        current_value = current * total_value
                        target_value = target * total_value
                        trade_amount = target_value - current_value
                        
                        rebalancing_analysis["trades_required"].append({
                            "asset_index": i,
                            "current_weight": round(current * 100, 2),
                            "target_weight": round(target * 100, 2),
                            "trade_amount": round(trade_amount, 2),
                            "action": "buy" if trade_amount > 0 else "sell"
                        })
            
            return rebalancing_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing rebalancing: {e}")
            return {"error": str(e), "needs_rebalancing": False}