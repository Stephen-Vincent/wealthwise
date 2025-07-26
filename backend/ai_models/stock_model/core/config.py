# ai_models/stock_model/core/config.py

"""
Configuration and Asset Universe Definitions for WealthWise

This module contains all the static configuration data including
asset universes, risk mappings, and system parameters.
"""

from typing import Dict, List

# ===================================================================
# ASSET UNIVERSE CONFIGURATION
# ===================================================================
# Defines investment options for each risk tolerance level
# Each category has specific ETFs and expected returns based on historical data

ASSET_UNIVERSES = {
    "ultra_conservative": {
        # PRIMARY GOAL: Capital preservation with minimal volatility
        # TARGET INVESTOR: Risk-averse, near retirement, short timeframe
        "bonds_govt": ["TLT", "IEF", "SHY"],  # US Treasury bonds (different durations)
        "bonds_corporate": ["LQD", "VCIT", "BND"],  # High-grade corporate bonds
        "dividend_aristocrats": ["NOBL", "VIG", "DVY"],  # Dividend growth stocks
        "utilities": ["VPU", "XLU"],  # Utility sector (stable, defensive)
        "allocation": {
            "bonds_govt": 0.4, 
            "bonds_corporate": 0.3, 
            "dividend_aristocrats": 0.2, 
            "utilities": 0.1
        },
        "expected_annual_return": 0.05,  # 5% annual return expectation
        "volatility": 0.08  # 8% annual volatility (low risk)
    },
    "conservative": {
        # PRIMARY GOAL: Stability with moderate growth potential
        # TARGET INVESTOR: Risk-averse but wants some growth, medium timeframe
        "bonds": ["BND", "AGG", "VTEB"],  # Broad bond market exposure
        "large_cap_value": ["VTV", "VYM", "SCHV"],  # Value stocks with dividends
        "international_developed": ["VEA", "VXUS"],  # International diversification
        "reits": ["VNQ", "SCHH"],  # Real estate investment trusts
        "allocation": {
            "bonds": 0.4, 
            "large_cap_value": 0.35, 
            "international_developed": 0.15, 
            "reits": 0.1
        },
        "expected_annual_return": 0.07,  # 7% annual return expectation
        "volatility": 0.12  # 12% annual volatility
    },
    "moderate": {
        # PRIMARY GOAL: Balanced growth and income
        # TARGET INVESTOR: Moderate risk tolerance, long-term goals
        "large_cap_blend": ["VTI", "ITOT", "SWTSX"],  # Total market exposure
        "international_blend": ["VEA", "VWO", "VTIAX"],  # Global diversification
        "bonds": ["BND", "AGG"],  # Bond foundation for stability
        "sector_rotation": ["VGT", "VHT", "VFH"],  # Growth sectors (tech, health, finance)
        "allocation": {
            "large_cap_blend": 0.4, 
            "international_blend": 0.25, 
            "bonds": 0.2, 
            "sector_rotation": 0.15
        },
        "expected_annual_return": 0.09,  # 9% annual return expectation
        "volatility": 0.15  # 15% annual volatility
    },
    "moderate_aggressive": {
        # PRIMARY GOAL: Growth-focused with some stability
        # TARGET INVESTOR: Higher risk tolerance, long timeframe, growth goals
        "large_cap_growth": ["VUG", "MGK", "SCHG"],  # Growth-oriented stocks
        "small_cap": ["VB", "IWM", "VTI"],  # Small cap exposure (higher growth potential)
        "international_growth": ["VEA", "VWO", "IEMG"],  # International growth
        "tech_innovation": ["VGT", "ARKK", "QQQ"],  # Technology focus
        "allocation": {
            "large_cap_growth": 0.35, 
            "small_cap": 0.25, 
            "international_growth": 0.25, 
            "tech_innovation": 0.15
        },
        "expected_annual_return": 0.11,  # 11% annual return expectation
        "volatility": 0.18  # 18% annual volatility
    },
    "aggressive": {
        # PRIMARY GOAL: Maximum growth potential
        # TARGET INVESTOR: High risk tolerance, long timeframe, wealth building
        "growth_stocks": ["VUG", "QQQ", "VGT"],  # High-growth stocks
        "small_cap_growth": ["VBK", "IWO", "VTWO"],  # Small cap growth
        "emerging_markets": ["VWO", "IEMG", "EEM"],  # Emerging market exposure
        "innovation_themes": ["ARKK", "ARKQ", "ARKG"],  # Thematic/disruptive investing
        "allocation": {
            "growth_stocks": 0.4, 
            "small_cap_growth": 0.25, 
            "emerging_markets": 0.2, 
            "innovation_themes": 0.15
        },
        "expected_annual_return": 0.13,  # 13% annual return expectation
        "volatility": 0.22  # 22% annual volatility
    },
    "ultra_aggressive": {
        # PRIMARY GOAL: Highest risk/reward, maximum growth
        # TARGET INVESTOR: Very high risk tolerance, very long timeframe, speculative
        "high_growth_tech": ["ARKK", "WCLD", "SKYY"],  # High-growth technology
        "crypto_exposure": ["BITO", "COIN"],  # Cryptocurrency exposure
        "biotech_innovation": ["XBI", "IBB", "ARKG"],  # Biotech sector
        "disruptive_tech": ["ARKQ", "ROBO", "FINX"],  # Disruptive technologies
        "allocation": {
            "high_growth_tech": 0.4, 
            "crypto_exposure": 0.25, 
            "biotech_innovation": 0.2, 
            "disruptive_tech": 0.15
        },
        "expected_annual_return": 0.16,  # 16% annual return expectation
        "volatility": 0.28  # 28% annual volatility (high risk)
    }
}

# ===================================================================
# BACKUP TICKER CONFIGURATION
# ===================================================================
# Reliable ETFs for each risk category when primary selections fail
# These are well-established, liquid ETFs with good track records

BACKUP_TICKERS = {
    "ultra_conservative": ["BND", "VTI", "VEA", "VYM", "VTEB"],
    "conservative": ["VTI", "BND", "VEA", "VWO", "VYM"],
    "moderate": ["VTI", "VEA", "VWO", "BND", "VNQ"],
    "moderate_aggressive": ["VTI", "VUG", "VEA", "VWO", "QQQ"],
    "aggressive": ["QQQ", "VUG", "VWO", "ARKK", "VGT"],
    "ultra_aggressive": ["ARKK", "QQQ", "VUG", "VWO", "VGT"]
}

# ===================================================================
# RISK SCORE MAPPINGS
# ===================================================================
# Convert between numerical risk scores and categorical risk levels

RISK_SCORE_TO_CATEGORY = {
    (0, 15): "ultra_conservative",
    (15, 30): "conservative", 
    (30, 50): "moderate",
    (50, 70): "moderate_aggressive",
    (70, 85): "aggressive",
    (85, 100): "ultra_aggressive"
}

CATEGORY_TO_RISK_SCORE = {
    "ultra_conservative": 10,
    "conservative": 25,
    "moderate": 40,
    "moderate_aggressive": 60,
    "aggressive": 75,
    "ultra_aggressive": 90
}

# ===================================================================
# SHAP MODEL CONFIGURATION
# ===================================================================
# Feature names for SHAP explainable AI analysis

SHAP_FEATURE_NAMES = [
    "target_value_log",     # Log of investment target (helps with scaling)
    "timeframe",            # Investment horizon in years
    "risk_score",           # User's risk tolerance (0-100)
    "required_return",      # Annual return needed to reach goal
    "monthly_contribution", # Regular monthly investment amount
    "market_volatility",    # Current market volatility (VIX level)
    "market_trend_score"    # Market momentum indicator (0-5)
]

# ===================================================================
# FACTOR ANALYSIS CONFIGURATION
# ===================================================================
# Default weights for multi-factor stock analysis

DEFAULT_FACTOR_WEIGHTS = {
    "momentum": 0.25,   # 25% - Recent price performance
    "quality": 0.25,    # 25% - Consistency and reliability
    "volatility": 0.20, # 20% - Risk characteristics
    "value": 0.15,      # 15% - Valuation metrics (P/E, P/B)
    "size": 0.10,       # 10% - Market capitalization effects
    "technical": 0.05   # 5% - Short-term technical indicators
}

# ===================================================================
# MARKET REGIME ADJUSTMENTS
# ===================================================================
# Portfolio adjustments based on detected market conditions

REGIME_ADJUSTMENTS = {
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
        "growth_tilt": 0,           # All neutral adjustments
        "defensive_tilt": 0,
        "volatility_adjustment": 0
    }
}

# ===================================================================
# RELIABLE BACKUP TICKERS
# ===================================================================
# Most reliable ETFs for fallback scenarios across all risk levels

RELIABLE_TICKERS = [
    'VTI',   # Total Stock Market ETF
    'BND',   # Total Bond Market ETF  
    'VEA',   # Developed Markets ETF
    'VWO',   # Emerging Markets ETF
    'VNQ',   # Real Estate ETF
    'QQQ',   # NASDAQ-100 ETF
    'SPY',   # S&P 500 ETF
    'VUG',   # Growth ETF
    'VYM',   # High Dividend Yield ETF
    'VGT'    # Technology Sector ETF
]

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_risk_category(risk_score: float) -> str:
    """
    Convert numerical risk score to risk category.
    
    Args:
        risk_score: Risk tolerance score from 0-100
        
    Returns:
        String risk category name
        
    Example:
        get_risk_category(65) -> "moderate_aggressive"
    """
    for (min_score, max_score), category in RISK_SCORE_TO_CATEGORY.items():
        if min_score <= risk_score < max_score:
            return category
    return "ultra_aggressive"  # Default for scores >= 85

def get_risk_score(category: str) -> float:
    """
    Convert risk category to representative numerical score.
    
    Args:
        category: Risk category name
        
    Returns:
        Float representing typical risk score for this category
        
    Example:
        get_risk_score("moderate_aggressive") -> 60.0
    """
    return CATEGORY_TO_RISK_SCORE.get(category, 40.0)

def get_expected_return(risk_category: str) -> float:
    """
    Get expected annual return for a risk category.
    
    Args:
        risk_category: Risk category name
        
    Returns:
        Expected annual return as decimal (e.g., 0.09 = 9%)
    """
    return ASSET_UNIVERSES.get(risk_category, ASSET_UNIVERSES["moderate"])["expected_annual_return"]

def get_expected_volatility(risk_category: str) -> float:
    """
    Get expected annual volatility for a risk category.
    
    Args:
        risk_category: Risk category name
        
    Returns:
        Expected annual volatility as decimal (e.g., 0.15 = 15%)
    """
    return ASSET_UNIVERSES.get(risk_category, ASSET_UNIVERSES["moderate"])["volatility"]

def validate_risk_category(category: str) -> bool:
    """
    Validate if a risk category is valid.
    
    Args:
        category: Risk category to validate
        
    Returns:
        True if valid, False otherwise
    """
    return category in ASSET_UNIVERSES

def get_all_tickers_for_category(risk_category: str) -> List[str]:
    """
    Get all tickers available for a specific risk category.
    
    Args:
        risk_category: Risk category name
        
    Returns:
        List of all ticker symbols for this category
    """
    if risk_category not in ASSET_UNIVERSES:
        return BACKUP_TICKERS.get(risk_category, RELIABLE_TICKERS[:5])
    
    universe = ASSET_UNIVERSES[risk_category]
    all_tickers = []
    
    for category, tickers in universe.items():
        if isinstance(tickers, list):
            all_tickers.extend(tickers)
    
    # Add backup tickers for more options
    backup_tickers = BACKUP_TICKERS.get(risk_category, [])
    all_tickers.extend(backup_tickers)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(all_tickers))