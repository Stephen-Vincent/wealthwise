"""
Market Regime Detection Module

This module implements AI-powered market condition detection using time series analysis.
Instead of giving the same advice regardless of market conditions, the system adapts
recommendations based on current market regime.

AI Techniques Used:
1. Moving Average Analysis - Trend detection using 20, 50, 200-day averages
2. Volatility Analysis - VIX levels for fear/greed assessment
3. Momentum Analysis - Recent returns for trend strength
4. Pattern Recognition - Classifying market conditions using ML logic
"""

import yfinance as yf
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    AI-Powered Market Regime Detection System
    
    This class analyzes current market conditions to determine the "regime"
    (bull market, bear market, high volatility, etc.) and provides adjustment
    factors for portfolio recommendations.
    
    Market Regimes Detected:
    - strong_bull: Strong uptrend, low volatility, good returns
    - bull: Moderate uptrend, normal volatility
    - bear: Downtrend, usually higher volatility
    - high_volatility: High VIX regardless of direction
    - low_volatility: Low VIX, calm markets
    - sideways: No clear trend, mixed signals
    """
    
    def __init__(self):
        self.last_regime_data = None
        self.cache_timeout = 3600  # Cache results for 1 hour
        self.last_update = None
    
    def detect_market_regime(self) -> Dict[str, Any]:
        """
        AI-Powered Market Regime Detection using Time Series Analysis
        
        Returns:
            Dict containing:
            - regime: Market condition classification
            - confidence: How confident the AI is in this classification
            - current_vix: Current volatility level
            - trend_score: Strength of current trend (0-5)
            - returns_1m/3m: Recent performance
            - adjustment_factor: How to adjust portfolio allocation
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Using cached market regime data")
            return self.last_regime_data
        
        try:
            logger.info("🔍 Analyzing market conditions using AI regime detection...")
            
            # Download market data for analysis
            spy = yf.download("SPY", period="1y", progress=False)['Close']
            vix = yf.download("^VIX", period="3mo", progress=False)['Close']
            
            # Validate we have sufficient data for analysis
            if len(spy) < 50:
                logger.warning("Insufficient SPY data for regime detection")
                return self._default_market_regime()
            
            if len(vix) < 20:
                logger.warning("Insufficient VIX data for regime detection")
                return self._default_market_regime()
            
            # === TREND ANALYSIS USING MOVING AVERAGES ===
            current_price = float(spy.iloc[-1])
            sma_20 = float(spy.rolling(20).mean().iloc[-1])
            sma_50 = float(spy.rolling(50).mean().iloc[-1]) if len(spy) >= 50 else current_price
            sma_200 = float(spy.rolling(200).mean().iloc[-1]) if len(spy) >= 200 else current_price
            
            # Calculate trend strength score (0-5 scale)
            trend_score = self._calculate_trend_score(current_price, sma_20, sma_50, sma_200)
            
            # === VOLATILITY ANALYSIS ===
            current_vix = float(vix.iloc[-1])
            avg_vix = float(vix.mean())
            
            # === MOMENTUM ANALYSIS ===
            returns_1m, returns_3m = self._calculate_momentum(spy)
            
            # === AI-BASED REGIME CLASSIFICATION ===
            regime, confidence = self._classify_regime(
                trend_score, current_vix, returns_1m, returns_3m
            )
            
            logger.info(f"📊 Market Regime: {regime.upper()} (VIX: {current_vix:.1f}, Trend: {trend_score}/5, 3M Return: {returns_3m:.1%})")
            
            regime_data = {
                "regime": regime,
                "confidence": confidence,
                "trend_score": trend_score,
                "current_vix": current_vix,
                "avg_vix": avg_vix,
                "returns_1m": returns_1m,
                "returns_3m": returns_3m,
                "adjustment_factor": self._get_regime_adjustment(regime)
            }
            
            # Cache the results
            self.last_regime_data = regime_data
            self.last_update = datetime.now()
            
            return regime_data
            
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return self._default_market_regime()
    
    def _calculate_trend_score(self, current_price: float, sma_20: float, 
                              sma_50: float, sma_200: float) -> float:
        """
        Calculate trend strength score (0-5 scale)
        Each condition adds 1 point if true
        """
        trend_score = 0
        
        if current_price > sma_20:  # Price above short-term average
            trend_score += 1
        if current_price > sma_50:  # Price above medium-term average
            trend_score += 1
        if current_price > sma_200: # Price above long-term average
            trend_score += 1
        if sma_20 > sma_50:        # Short-term above medium-term (momentum)
            trend_score += 1
        if sma_50 > sma_200:       # Medium-term above long-term (trend strength)
            trend_score += 1
        
        return float(trend_score)
    
    def _calculate_momentum(self, spy_data) -> tuple:
        """Calculate recent returns for momentum assessment"""
        returns_1m = 0.0
        returns_3m = 0.0
        
        if len(spy_data) >= 21:  # 1 month ≈ 21 trading days
            returns_1m = float((spy_data.iloc[-1] / spy_data.iloc[-21]) - 1)
        if len(spy_data) >= 63:  # 3 months ≈ 63 trading days
            returns_3m = float((spy_data.iloc[-1] / spy_data.iloc[-63]) - 1)
        
        return returns_1m, returns_3m
    
    def _classify_regime(self, trend_score: float, current_vix: float,
                        returns_1m: float, returns_3m: float) -> tuple:
        """
        AI-based regime classification using machine learning logic
        
        Returns:
            tuple: (regime_name, confidence_level)
        """
        if trend_score >= 4 and current_vix < 25 and returns_3m > 0.05:
            # Strong trend + low fear + good returns = Strong Bull
            return "strong_bull", 0.85
        elif trend_score >= 3 and returns_1m > 0:
            # Good trend + positive recent returns = Bull
            return "bull", 0.75
        elif trend_score <= 1 and returns_3m < -0.10:
            # Weak trend + poor returns = Bear
            return "bear", 0.80
        elif current_vix > 30:
            # High fear regardless of other factors = High Volatility
            return "high_volatility", 0.70
        elif current_vix < 15:
            # Low fear = Low Volatility (calm markets)
            return "low_volatility", 0.65
        else:
            # Mixed signals = Sideways market
            return "sideways", 0.60
    
    def _get_regime_adjustment(self, regime: str) -> Dict[str, float]:
        """
        Get portfolio adjustments based on detected market regime
        
        Returns adjustment factors for:
        - growth_tilt: Increase/decrease allocation to growth assets
        - defensive_tilt: Increase/decrease allocation to defensive assets
        - volatility_adjustment: Adjust for expected volatility changes
        """
        adjustments = {
            "strong_bull": {
                "growth_tilt": 0.15,        # Increase growth allocation 15%
                "defensive_tilt": -0.10,    # Decrease defensive allocation 10%
                "volatility_adjustment": -0.05  # Expect lower volatility
            },
            "bull": {
                "growth_tilt": 0.10,        # Moderate increase in growth
                "defensive_tilt": -0.05,    # Slight decrease in defensive
                "volatility_adjustment": 0   # No volatility adjustment
            },
            "bear": {
                "growth_tilt": -0.15,       # Decrease growth allocation 15%
                "defensive_tilt": 0.20,     # Increase defensive allocation 20%
                "volatility_adjustment": 0.10  # Expect higher volatility
            },
            "high_volatility": {
                "growth_tilt": -0.10,       # Reduce growth exposure
                "defensive_tilt": 0.15,     # Increase defensive assets
                "volatility_adjustment": 0.15  # Account for high volatility
            },
            "low_volatility": {
                "growth_tilt": 0.05,        # Slight increase in growth
                "defensive_tilt": -0.05,    # Slight decrease in defensive
                "volatility_adjustment": -0.10  # Lower expected volatility
            },
            "sideways": {
                "growth_tilt": 0,           # No growth tilt
                "defensive_tilt": 0.05,     # Slight defensive bias
                "volatility_adjustment": 0   # Neutral volatility
            },
            "neutral": {
                "growth_tilt": 0,           # All neutral
                "defensive_tilt": 0,
                "volatility_adjustment": 0
            }
        }
        return adjustments.get(regime, adjustments["neutral"])
    
    def _default_market_regime(self) -> Dict[str, Any]:
        """
        Default market regime when detection fails
        Returns neutral/average market conditions
        """
        return {
            "regime": "neutral",
            "confidence": 0.50,
            "trend_score": 2.5,
            "current_vix": 20,
            "avg_vix": 20,
            "returns_1m": 0,
            "returns_3m": 0,
            "adjustment_factor": {"growth_tilt": 0, "defensive_tilt": 0, "volatility_adjustment": 0}
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cached regime data is still valid"""
        if self.last_regime_data is None or self.last_update is None:
            return False
        
        time_elapsed = (datetime.now() - self.last_update).total_seconds()
        return time_elapsed < self.cache_timeout
    
    def get_cached_or_detect(self) -> Dict[str, Any]:
        """
        Returns cached regime data if still valid; otherwise triggers detection.
        """
        if self._is_cache_valid():
            logger.debug("🧠 Using cached market regime data")
            return self.last_regime_data
        else:
            return self.detect_market_regime()
    
    def adjust_risk_for_market_regime(self, risk_category: str, market_regime: Dict) -> str:
        """
        AI-powered risk adjustment based on current market conditions
        
        Adjusts the user's base risk tolerance based on market conditions.
        For example, in bear markets, becomes slightly more conservative.
        """
        regime = market_regime['regime']
        
        # Define risk category hierarchy
        risk_levels = ["ultra_conservative", "conservative", "moderate", 
                       "moderate_aggressive", "aggressive", "ultra_aggressive"]
        
        current_index = risk_levels.index(risk_category) if risk_category in risk_levels else 2
        
        # AI-based adjustment logic
        if regime == "bear" or regime == "high_volatility":
            adjustment = -1  # Move toward more conservative in bad markets
        elif regime == "strong_bull" and market_regime['adjustment_factor']['growth_tilt'] > 0.1:
            adjustment = 1   # Move slightly more aggressive in strong bull markets
        else:
            adjustment = 0   # No adjustment for neutral conditions
        
        new_index = max(0, min(len(risk_levels) - 1, current_index + adjustment))
        adjusted_category = risk_levels[new_index]
        
        if adjusted_category != risk_category:
            logger.info(f"🔄 AI Market Adjustment: {risk_category} → {adjusted_category}")
        
        return adjusted_category