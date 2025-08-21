"""
Configuration management for the Portfolio Simulator Service.

This module centralizes all configuration settings and provides
type-safe access to environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Enumeration for logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class VisualizationConfig:
    """Visualization service configuration."""
    output_directory: Path
    file_format: str = "png"
    max_file_size_mb: int = 10
    enable_caching: bool = True


@dataclass
class AIServiceConfig:
    """AI service configuration settings."""
    groq_api_key: str
    model_name: str = "mixtral-8x7b-32768"
    max_tokens: int = 1000
    temperature: float = 0.3
    enable_shap: bool = True


@dataclass
class SimulationConfig:
    """Portfolio simulation configuration."""
    min_timeframe_years: int = 1
    max_timeframe_years: int = 50
    min_investment_amount: float = 1.0
    max_investment_amount: float = 10_000_000.0
    default_risk_free_rate: float = 0.03
    trading_days_per_month: int = 21
    trading_days_per_year: int = 252


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    max_filename_length: int = 100
    allowed_file_extensions: list[str] = None
    sanitize_inputs: bool = True
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.png', '.jpg', '.jpeg', '.svg']


@dataclass
class AppConfig:
    """Main application configuration container."""
    database: DatabaseConfig
    visualization: VisualizationConfig
    ai_service: AIServiceConfig
    simulation: SimulationConfig
    security: SecurityConfig
    log_level: LogLevel = LogLevel.INFO
    debug_mode: bool = False
    
    @classmethod
    def from_environment(cls) -> 'AppConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            AppConfig: Fully configured application settings
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Required environment variables
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
            
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Optional environment variables with defaults
        viz_dir = Path(os.getenv('VISUALIZATION_DIR', 'static/visualizations'))
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        return cls(
            database=DatabaseConfig(
                url=database_url,
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30'))
            ),
            visualization=VisualizationConfig(
                output_directory=viz_dir,
                file_format=os.getenv('VIZ_FORMAT', 'png'),
                max_file_size_mb=int(os.getenv('VIZ_MAX_SIZE_MB', '10')),
                enable_caching=os.getenv('VIZ_ENABLE_CACHING', 'true').lower() == 'true'
            ),
            ai_service=AIServiceConfig(
                groq_api_key=groq_api_key,
                model_name=os.getenv('AI_MODEL_NAME', 'mixtral-8x7b-32768'),
                max_tokens=int(os.getenv('AI_MAX_TOKENS', '1000')),
                temperature=float(os.getenv('AI_TEMPERATURE', '0.3')),
                enable_shap=os.getenv('AI_ENABLE_SHAP', 'true').lower() == 'true'
            ),
            simulation=SimulationConfig(
                min_timeframe_years=int(os.getenv('SIM_MIN_TIMEFRAME', '1')),
                max_timeframe_years=int(os.getenv('SIM_MAX_TIMEFRAME', '50')),
                min_investment_amount=float(os.getenv('SIM_MIN_INVESTMENT', '1.0')),
                max_investment_amount=float(os.getenv('SIM_MAX_INVESTMENT', '10000000.0')),
                default_risk_free_rate=float(os.getenv('SIM_RISK_FREE_RATE', '0.03')),
                trading_days_per_month=int(os.getenv('SIM_TRADING_DAYS_MONTH', '21')),
                trading_days_per_year=int(os.getenv('SIM_TRADING_DAYS_YEAR', '252'))
            ),
            security=SecurityConfig(
                max_filename_length=int(os.getenv('SEC_MAX_FILENAME_LENGTH', '100')),
                sanitize_inputs=os.getenv('SEC_SANITIZE_INPUTS', 'true').lower() == 'true'
            ),
            log_level=LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        )


# Global configuration instance
# This will be initialized when the application starts
_config: AppConfig = None


def get_config() -> AppConfig:
    """
    Get the global application configuration.
    
    Returns:
        AppConfig: The application configuration instance
        
    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    global _config
    if _config is None:
        raise RuntimeError(
            "Configuration not initialized. Call initialize_config() first."
        )
    return _config


def initialize_config() -> AppConfig:
    """
    Initialize the global application configuration from environment.
    
    Returns:
        AppConfig: The initialized configuration
    """
    global _config
    _config = AppConfig.from_environment()
    return _config


def get_stock_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for supported stocks and ETFs.
    
    Returns:
        Dict mapping ticker symbols to metadata including:
        - name: Human-readable name
        - risk_score: Estimated risk (0-100)
        - expected_return: Estimated annual return percentage
        - category: Asset category (equity, bond, etc.)
        - description: Brief description
    """
    return {
        # Bond ETFs - Low Risk
        "BND": {
            "name": "Vanguard Total Bond Market ETF",
            "risk_score": 5.0,
            "expected_return": 3.0,
            "category": "bond",
            "description": "Broad diversification across U.S. investment-grade bonds"
        },
        "VTEB": {
            "name": "Vanguard Tax-Exempt Bond ETF",
            "risk_score": 4.0,
            "expected_return": 2.5,
            "category": "bond",
            "description": "Municipal bonds with tax advantages"
        },
        "AGG": {
            "name": "iShares Core Aggregate Bond ETF",
            "risk_score": 5.5,
            "expected_return": 3.5,
            "category": "bond",
            "description": "Core U.S. bond market exposure"
        },
        
        # Dividend/Value ETFs - Low-Medium Risk
        "VYM": {
            "name": "Vanguard High Dividend Yield ETF",
            "risk_score": 12.0,
            "expected_return": 7.0,
            "category": "equity_dividend",
            "description": "High-quality dividend-paying stocks"
        },
        "SCHD": {
            "name": "Schwab US Dividend Equity ETF",
            "risk_score": 14.0,
            "expected_return": 8.0,
            "category": "equity_dividend",
            "description": "High-quality dividend stocks with growth potential"
        },
        
        # Broad Market ETFs - Medium Risk
        "VTI": {
            "name": "Vanguard Total Stock Market ETF",
            "risk_score": 16.0,
            "expected_return": 9.0,
            "category": "equity_broad",
            "description": "Total U.S. stock market exposure"
        },
        "VOO": {
            "name": "Vanguard S&P 500 ETF",
            "risk_score": 15.0,
            "expected_return": 9.5,
            "category": "equity_large_cap",
            "description": "S&P 500 large-cap exposure"
        },
        "VEA": {
            "name": "Vanguard FTSE Developed Markets ETF",
            "risk_score": 18.0,
            "expected_return": 7.5,
            "category": "equity_international",
            "description": "International developed market stocks"
        },
        "VWO": {
            "name": "Vanguard Emerging Markets Stock ETF",
            "risk_score": 25.0,
            "expected_return": 8.5,
            "category": "equity_emerging",
            "description": "Emerging market equity exposure"
        },
        "VNQ": {
            "name": "Vanguard Real Estate ETF",
            "risk_score": 20.0,
            "expected_return": 8.5,
            "category": "reit",
            "description": "Real estate investment trusts"
        },
        
        # Growth ETFs - Medium-High Risk
        "VUG": {
            "name": "Vanguard Growth ETF",
            "risk_score": 18.0,
            "expected_return": 11.0,
            "category": "equity_growth",
            "description": "Growth-oriented U.S. stocks"
        },
        "VGT": {
            "name": "Vanguard Information Technology ETF",
            "risk_score": 22.0,
            "expected_return": 12.0,
            "category": "equity_sector",
            "description": "Technology sector concentration"
        },
        "QQQ": {
            "name": "Invesco QQQ Trust",
            "risk_score": 24.0,
            "expected_return": 12.5,
            "category": "equity_tech",
            "description": "NASDAQ-100 technology focus"
        },
        
        # High Risk/High Reward
        "ARKK": {
            "name": "ARK Innovation ETF",
            "risk_score": 35.0,
            "expected_return": 15.0,
            "category": "equity_innovation",
            "description": "Disruptive innovation companies"
        },
        "TQQQ": {
            "name": "ProShares UltraPro QQQ",
            "risk_score": 50.0,
            "expected_return": 20.0,
            "category": "leveraged",
            "description": "3x leveraged NASDAQ-100 exposure (high risk)"
        },
        "SOXL": {
            "name": "Direxion Daily Semiconductor Bull 3X",
            "risk_score": 55.0,
            "expected_return": 22.0,
            "category": "leveraged",
            "description": "3x leveraged semiconductor exposure (very high risk)"
        }
    }


def get_risk_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined risk profiles for portfolio allocation.
    
    Returns:
        Dict mapping risk profile names to their characteristics
    """
    return {
        "conservative": {
            "risk_score_range": (0, 35),
            "description": "Capital preservation with modest growth",
            "max_equity_allocation": 0.4,
            "min_bond_allocation": 0.5,
            "suggested_etfs": ["VTI", "BND", "VEA", "VTEB", "VYM"]
        },
        "moderate": {
            "risk_score_range": (35, 70),
            "description": "Balanced growth with moderate risk",
            "max_equity_allocation": 0.8,
            "min_bond_allocation": 0.2,
            "suggested_etfs": ["VTI", "VEA", "VWO", "VNQ", "BND"]
        },
        "aggressive": {
            "risk_score_range": (70, 100),
            "description": "Maximum growth with high risk tolerance",
            "max_equity_allocation": 1.0,
            "min_bond_allocation": 0.0,
            "suggested_etfs": ["VTI", "VGT", "VUG", "ARKK", "VEA"]
        }
    }