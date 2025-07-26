"""
Multi-Factor Stock Analysis Module

This module implements quantitative factor analysis for stock selection.
Instead of randomly picking stocks, the AI evaluates each stock across
multiple financial factors and selects the best-scoring options.

Factors Analyzed:
1. Momentum - Recent price performance (6M and 12M returns)
2. Volatility - Risk characteristics (lower volatility = higher quality)
3. Quality - Consistency and reliability metrics
4. Value - Fundamental valuation (P/E, P/B ratios)
5. Size - Market capitalization effects
6. Technical - Short-term momentum indicators (RSI-based)
"""

import yfinance as yf
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FactorAnalyzer:
    """
    AI-Powered Multi-Factor Analysis for Stock Evaluation
    
    This class implements sophisticated quantitative analysis to evaluate
    stocks across multiple financial factors, similar to what professional
    investment managers use.
    """
    
    def __init__(self):
        self.factor_cache = {}
        self.cache_timeout = 1800  # Cache factor scores for 30 minutes
    
    def analyze_stock_factors(self, stock: str) -> Dict[str, float]:
        """
        Comprehensive multi-factor analysis for individual stock evaluation
        
        Each factor gets a score from -1 to +1, then combined into composite score
        
        Returns:
            Dict with individual factor scores and composite score:
            {
                "momentum": 0.3,      # Positive momentum
                "volatility": 0.2,    # Good volatility profile
                "quality": 0.1,       # Average quality
                "value": -0.1,        # Slightly expensive
                "size": 0.4,          # Good market cap
                "technical": 0.0,     # Neutral technical
                "composite": 0.15     # Overall score
            }
        """
        # Check cache first
        cache_key = f"{stock}_{datetime.now().strftime('%Y%m%d_%H')}"  # Hourly cache
        if cache_key in self.factor_cache:
            return self.factor_cache[cache_key]
        
        try:
            logger.debug(f"🔬 Analyzing {stock} across multiple factors...")
            
            # Download 2 years of stock data for comprehensive analysis
            ticker = yf.Ticker(stock)
            hist_data = ticker.history(period="2y")
            info = ticker.info
            
            # Validate sufficient data for analysis
            if len(hist_data) < 200:
                logger.warning(f"{stock}: Insufficient data for factor analysis")
                return self._default_factor_scores()
            
            # Extract price and volume data
            prices = hist_data['Close']
            volumes = hist_data['Volume']
            returns = prices.pct_change().dropna()
            
            # Calculate all factors
            factor_scores = {
                "momentum": self._calculate_momentum_factor(prices),
                "volatility": self._calculate_volatility_factor(returns),
                "quality": self._calculate_quality_factor(returns, volumes),
                "value": self._calculate_value_factor(info),
                "size": self._calculate_size_factor(info),
                "technical": self._calculate_technical_factor(returns)
            }
            
            # Calculate composite score with weighted combination
            factor_scores["composite"] = self._calculate_composite_score(factor_scores)
            
            # Cache the results
            self.factor_cache[cache_key] = factor_scores
            
            logger.debug(f"✓ {stock}: Composite score = {factor_scores['composite']:.3f}")
            return factor_scores
            
        except Exception as e:
            logger.warning(f"Factor analysis failed for {stock}: {e}")
            return self._default_factor_scores()
    
    def _calculate_momentum_factor(self, prices: pd.Series) -> float:
        """
        Calculate momentum factor based on recent price performance
        
        Theory: Stocks that have performed well recently tend to continue
        """
        try:
            # Calculate 6-month and 12-month momentum
            momentum_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) if len(prices) >= 126 else 0
            momentum_12m = (prices.iloc[-1] / prices.iloc[-252] - 1) if len(prices) >= 252 else 0
            
            # Combine with higher weight on recent performance
            combined_momentum = (momentum_6m * 0.6 + momentum_12m * 0.4) * 2
            
            # Normalize using tanh to bound between -1 and 1
            return float(np.tanh(combined_momentum))
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_factor(self, returns: pd.Series) -> float:
        """
        Calculate volatility factor (lower volatility = higher score)
        
        Theory: Less volatile stocks are more predictable and safer
        """
        try:
            annual_volatility = returns.std() * np.sqrt(252)
            # Score is higher for stocks with volatility around 20% (market average)
            volatility_score = max(-1, min(1, (0.20 - annual_volatility) / 0.15))
            return float(volatility_score)
            
        except Exception:
            return 0.0
    
    def _calculate_quality_factor(self, returns: pd.Series, volumes: pd.Series) -> float:
        """
        Calculate quality factor based on consistency metrics
        
        Theory: High-quality companies have consistent performance
        """
        try:
            # Return consistency (lower std dev of monthly returns = higher quality)
            monthly_returns = returns.rolling(21).sum().dropna()
            return_consistency = 1 - min(1, monthly_returns.std() * 4)
            
            # Volume consistency (more consistent trading = higher quality)
            volume_consistency = 1 - min(1, (volumes.std() / volumes.mean())) if volumes.mean() > 0 else 0
            
            # Combine consistency measures
            quality_score = (return_consistency * 0.7 + volume_consistency * 0.3)
            quality_score = (quality_score - 0.5) * 2  # Scale to [-1, 1]
            
            return float(quality_score)
            
        except Exception:
            return 0.0
    
    def _calculate_value_factor(self, info: Dict) -> float:
        """
        Calculate value factor based on fundamental ratios
        
        Theory: Cheaper stocks (lower P/E, P/B) often outperform
        """
        try:
            value_score = 0
            
            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            
            if pe_ratio and pe_ratio > 0:
                # Lower P/E = higher value score
                pe_score = max(-1, min(1, (25 - pe_ratio) / 20))
                value_score += pe_score * 0.6
            
            if pb_ratio and pb_ratio > 0:
                # Lower P/B = higher value score
                pb_score = max(-1, min(1, (3 - pb_ratio) / 2))
                value_score += pb_score * 0.4
            
            return float(value_score)
            
        except Exception:
            return 0.0
    
    def _calculate_size_factor(self, info: Dict) -> float:
        """
        Calculate size factor based on market capitalization
        
        Theory: Mid to large cap stocks often have better risk-adjusted returns
        """
        try:
            market_cap = info.get('marketCap', 0)
            if market_cap > 0:
                # Use log scale for market cap analysis
                log_cap = np.log10(market_cap)
                # Favor mid to large cap stocks (log scale 9-12, i.e., $1B-$1T)
                size_score = max(-1, min(1, (log_cap - 8) / 4 - 0.5))
                return float(size_score)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_technical_factor(self, returns: pd.Series) -> float:
        """
        Calculate technical factor using RSI-based analysis
        
        Theory: Oversold stocks (low RSI) may be due for a bounce
        """
        try:
            rsi_periods = 14
            if len(returns) >= rsi_periods:
                # Calculate RSI (Relative Strength Index)
                gains = returns.where(returns > 0, 0).rolling(rsi_periods).mean()
                losses = -returns.where(returns < 0, 0).rolling(rsi_periods).mean()
                rs = gains / (losses + 1e-10)  # Avoid division by zero
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Convert RSI to score: 30-70 neutral, <30 oversold (good), >70 overbought (bad)
                technical_score = max(-1, min(1, (50 - current_rsi) / 25))
                return float(technical_score)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_composite_score(self, factor_scores: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from individual factors
        
        Higher weights on momentum and quality (most predictive factors)
        """
        weights = {
            "momentum": 0.25,     # 25% weight on momentum
            "quality": 0.25,      # 25% weight on quality
            "volatility": 0.20,   # 20% weight on volatility
            "value": 0.15,        # 15% weight on value
            "size": 0.10,         # 10% weight on size
            "technical": 0.05     # 5% weight on technical
        }
        
        composite = sum(
            factor_scores.get(factor, 0) * weight 
            for factor, weight in weights.items()
        )
        
        return float(composite)
    
    def _default_factor_scores(self) -> Dict[str, float]:
        """Default factor scores when analysis fails"""
        return {
            "momentum": 0, "volatility": 0, "quality": 0,
            "value": 0, "size": 0, "technical": 0, "composite": 0
        }
    
    def rank_stocks_by_factors(self, stocks: List[str], 
                              factor_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        AI-Powered Multi-Criteria Decision Making for Stock Ranking
        
        Evaluates each stock across multiple factors and ranks them by composite score
        
        Args:
            stocks: List of stock tickers to rank
            factor_weights: Custom weights for each factor (optional)
            
        Returns:
            List of (stock, score) tuples sorted by score:
            [("VTI", 0.45), ("QQQ", 0.32), ("BND", 0.18), ...]
        """
        if factor_weights is None:
            factor_weights = {
                "momentum": 0.25,   # Strong predictor of future returns
                "quality": 0.25,    # Important for risk-adjusted returns
                "volatility": 0.20, # Risk management consideration
                "value": 0.15,      # Long-term performance factor
                "size": 0.10,       # Market cap considerations
                "technical": 0.05   # Short-term momentum
            }
        
        stock_scores = []
        
        logger.info(f"🏆 Ranking {len(stocks)} stocks using multi-factor analysis...")
        
        for stock in stocks:
            # Get factor scores for this stock
            factor_scores = self.analyze_stock_factors(stock)
            
            # Calculate weighted composite score
            composite_score = sum(
                factor_scores.get(factor, 0) * weight 
                for factor, weight in factor_weights.items()
            )
            
            stock_scores.append((stock, composite_score))
            logger.debug(f"{stock}: Composite score = {composite_score:.3f}")
        
        # Sort by score (highest first) - best stocks at the top
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ Stock ranking complete. Top 3: {[f'{s[0]}({s[1]:.2f})' for s in stock_scores[:3]]}")
        return stock_scores
    
    def get_factor_weights_for_regime(self, regime: str, timeframe: int) -> Dict[str, float]:
        """
        AI-optimized factor weights based on market regime and timeframe
        
        Dynamically adjusts how factors are weighted based on conditions
        """
        base_weights = {
            "momentum": 0.25, "quality": 0.25, "volatility": 0.20,
            "value": 0.15, "size": 0.10, "technical": 0.05
        }
        
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
        normalized_weights = {k: v/total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def get_expanded_universe(self, risk_category: str, asset_universes: Dict) -> List[str]:
        """Get expanded stock universe for comprehensive factor analysis"""
        universe = asset_universes.get(risk_category, asset_universes.get("moderate", {}))
        all_stocks = []
        
        # Collect all stocks from all categories in the universe
        for category, stocks in universe.items():
            if isinstance(stocks, list):
                all_stocks.extend(stocks)
        
        # Remove duplicates while preserving order
        expanded_universe = list(dict.fromkeys(all_stocks))
        
        return expanded_universe