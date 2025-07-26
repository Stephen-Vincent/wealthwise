"""
Data Validation and Stock Filtering Module

This module provides comprehensive data validation and stock filtering utilities
for the WealthWise Enhanced Stock Recommender. It ensures data integrity,
validates stock tickers, and handles data quality issues gracefully.

Key Features:
1. Stock ticker validation and filtering
2. Financial data quality checks
3. Input parameter validation
4. Data cleaning and preprocessing
5. Market data availability verification
6. Error handling and fallback mechanisms
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import re
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Comprehensive Data Validation System
    
    This class provides robust validation for all data inputs and stock information
    used throughout the recommendation system. It ensures data quality and handles
    edge cases gracefully.
    """
    
    def __init__(self):
        """Initialize the data validator with configuration"""
        
        # Known valid stock/ETF patterns
        self.valid_ticker_patterns = [
            r'^[A-Z]{1,5}$',        # Standard US stocks (1-5 letters)
            r'^[A-Z]{1,4}\.[A-Z]{1,2}$',  # International stocks with exchange
            r'^\^[A-Z0-9]+$',       # Indices (^VIX, ^GSPC)
            r'^[A-Z]{2,5}\-[A-Z]$', # Some international formats
        ]
        
        # Common invalid ticker patterns to catch early
        self.invalid_patterns = [
            r'.*\d{4,}.*',          # Tickers with 4+ consecutive digits
            r'.*[^A-Z0-9\.\-\^].*', # Non-alphanumeric characters
            r'^.{10,}$',            # Excessively long tickers
        ]
        
        # Cache for validation results to avoid repeated API calls
        self.validation_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # Minimum data requirements
        self.min_price_points = 50      # Minimum price data points
        self.min_volume_threshold = 1000 # Minimum average daily volume
        self.max_price_change = 0.50     # Maximum 50% single-day change (likely data error)
        
    def validate_stock_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Comprehensive stock ticker validation
        
        Validates whether a stock ticker is legitimate, tradeable, and has
        sufficient data for analysis.
        
        Args:
            ticker: Stock ticker symbol to validate
            
        Returns:
            Dict containing:
            - is_valid: Boolean indicating if ticker is valid
            - ticker: Cleaned ticker symbol
            - reason: Explanation of validation result
            - data_quality: Assessment of available data quality
            - last_price: Most recent price (if available)
            - market_cap: Market capitalization (if available)
        """
        try:
            # Input cleaning and basic validation
            cleaned_ticker = self._clean_ticker(ticker)
            
            if not cleaned_ticker:
                return self._invalid_result(ticker, "Empty or invalid ticker format")
            
            # Check cache first
            cache_key = f"validate_{cleaned_ticker}"
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if self._is_cache_valid(cached_result.get('timestamp')):
                    logger.debug(f"Using cached validation for {cleaned_ticker}")
                    return cached_result['result']
            
            # Pattern validation
            pattern_result = self._validate_ticker_pattern(cleaned_ticker)
            if not pattern_result['is_valid']:
                return pattern_result
            
            # API validation - check if ticker exists and has data
            api_result = self._validate_ticker_api(cleaned_ticker)
            
            # Cache the result
            self.validation_cache[cache_key] = {
                'result': api_result,
                'timestamp': datetime.now()
            }
            
            return api_result
            
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}")
            return self._invalid_result(ticker, f"Validation error: {str(e)}")
    
    def validate_stock_list(self, tickers: List[str], 
                           max_invalid_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Validate a list of stock tickers in batch
        
        Efficiently validates multiple tickers and provides summary statistics
        about the overall data quality of the list.
        
        Args:
            tickers: List of ticker symbols to validate
            max_invalid_ratio: Maximum ratio of invalid tickers allowed
            
        Returns:
            Dict containing:
            - valid_tickers: List of valid ticker symbols
            - invalid_tickers: List of invalid tickers with reasons
            - validation_summary: Summary statistics
            - overall_quality: Overall quality assessment
        """
        try:
            logger.info(f"Validating {len(tickers)} stock tickers...")
            
            valid_tickers = []
            invalid_tickers = []
            validation_details = {}
            
            for ticker in tickers:
                validation_result = self.validate_stock_ticker(ticker)
                
                if validation_result['is_valid']:
                    valid_tickers.append(validation_result['ticker'])
                    validation_details[ticker] = validation_result
                else:
                    invalid_tickers.append({
                        'ticker': ticker,
                        'reason': validation_result['reason']
                    })
                    logger.debug(f"Invalid ticker {ticker}: {validation_result['reason']}")
            
            # Calculate validation statistics
            total_tickers = len(tickers)
            valid_count = len(valid_tickers)
            invalid_count = len(invalid_tickers)
            invalid_ratio = invalid_count / total_tickers if total_tickers > 0 else 0
            
            # Assess overall quality
            if invalid_ratio <= 0.1:
                overall_quality = "excellent"
            elif invalid_ratio <= 0.2:
                overall_quality = "good"
            elif invalid_ratio <= max_invalid_ratio:
                overall_quality = "acceptable"
            else:
                overall_quality = "poor"
                logger.warning(f"High invalid ticker ratio: {invalid_ratio:.1%}")
            
            summary = {
                "total_tickers": total_tickers,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "invalid_ratio": invalid_ratio,
                "overall_quality": overall_quality,
                "meets_threshold": invalid_ratio <= max_invalid_ratio
            }
            
            logger.info(f"✅ Validation complete: {valid_count}/{total_tickers} valid ({invalid_ratio:.1%} invalid)")
            
            return {
                "valid_tickers": valid_tickers,
                "invalid_tickers": invalid_tickers,
                "validation_summary": summary,
                "validation_details": validation_details,
                "overall_quality": overall_quality
            }
            
        except Exception as e:
            logger.error(f"Error validating ticker list: {e}")
            return {
                "valid_tickers": [],
                "invalid_tickers": [],
                "validation_summary": {"error": str(e)},
                "overall_quality": "error"
            }
    
    def validate_financial_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate financial input parameters for goal calculations
        
        Ensures all user inputs are within reasonable bounds and
        mathematically valid for financial calculations.
        
        Args:
            **kwargs: Financial parameters to validate
                - target_value: Financial goal amount
                - current_investment: Starting investment
                - timeframe: Investment period in years
                - monthly_contribution: Regular monthly investments
                - risk_score: Risk tolerance score (0-100)
                
        Returns:
            Dict with validation results and cleaned parameters
        """
        try:
            logger.debug("Validating financial parameters...")
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "cleaned_params": {},
                "validation_details": {}
            }
            
            # Validate target_value
            target_value = kwargs.get('target_value', 0)
            target_result = self._validate_target_value(target_value)
            validation_results["cleaned_params"]["target_value"] = target_result["value"]
            validation_results["validation_details"]["target_value"] = target_result
            if not target_result["is_valid"]:
                validation_results["errors"].append(target_result["message"])
                validation_results["is_valid"] = False
            
            # Validate current_investment
            current_investment = kwargs.get('current_investment', 0)
            current_result = self._validate_current_investment(current_investment)
            validation_results["cleaned_params"]["current_investment"] = current_result["value"]
            validation_results["validation_details"]["current_investment"] = current_result
            if not current_result["is_valid"]:
                validation_results["errors"].append(current_result["message"])
                validation_results["is_valid"] = False
            
            # Validate timeframe
            timeframe = kwargs.get('timeframe', 10)
            timeframe_result = self._validate_timeframe(timeframe)
            validation_results["cleaned_params"]["timeframe"] = timeframe_result["value"]
            validation_results["validation_details"]["timeframe"] = timeframe_result
            if not timeframe_result["is_valid"]:
                validation_results["errors"].append(timeframe_result["message"])
                validation_results["is_valid"] = False
            
            # Validate monthly_contribution
            monthly_contribution = kwargs.get('monthly_contribution', 0)
            contribution_result = self._validate_monthly_contribution(monthly_contribution)
            validation_results["cleaned_params"]["monthly_contribution"] = contribution_result["value"]
            validation_results["validation_details"]["monthly_contribution"] = contribution_result
            if not contribution_result["is_valid"]:
                validation_results["errors"].append(contribution_result["message"])
                validation_results["is_valid"] = False
            
            # Validate risk_score
            risk_score = kwargs.get('risk_score', 50)
            risk_result = self._validate_risk_score(risk_score)
            validation_results["cleaned_params"]["risk_score"] = risk_result["value"]
            validation_results["validation_details"]["risk_score"] = risk_result
            if not risk_result["is_valid"]:
                validation_results["errors"].append(risk_result["message"])
                validation_results["is_valid"] = False
            
            # Cross-validation checks
            cross_validation = self._cross_validate_parameters(validation_results["cleaned_params"])
            validation_results["warnings"].extend(cross_validation["warnings"])
            
            if validation_results["is_valid"]:
                logger.debug("✅ All financial parameters validated successfully")
            else:
                logger.warning(f"❌ Parameter validation failed: {validation_results['errors']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating financial parameters: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "cleaned_params": {},
                "validation_details": {}
            }
    
    def validate_market_data(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Validate quality of market data downloaded from APIs
        
        Checks for data completeness, obvious errors, and sufficient
        history for reliable analysis.
        
        Args:
            data: DataFrame with market data (OHLCV format)
            ticker: Stock ticker for reference
            
        Returns:
            Dict with data quality assessment and cleaned data
        """
        try:
            logger.debug(f"Validating market data for {ticker}")
            
            validation_result = {
                "is_valid": True,
                "quality_score": 0,
                "issues": [],
                "warnings": [],
                "cleaned_data": None,
                "data_stats": {}
            }
            
            if data is None or data.empty:
                validation_result["is_valid"] = False
                validation_result["issues"].append("No data available")
                return validation_result
            
            # Basic data structure validation
            required_columns = ['Close']
            optional_columns = ['Open', 'High', 'Low', 'Volume']
            
            missing_required = [col for col in required_columns if col not in data.columns]
            if missing_required:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Missing required columns: {missing_required}")
                return validation_result
            
            # Data quantity validation
            data_length = len(data)
            validation_result["data_stats"]["total_points"] = data_length
            
            if data_length < self.min_price_points:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Insufficient data: {data_length} points (need {self.min_price_points})")
                return validation_result
            
            # Data quality checks
            quality_issues = []
            
            # Check for missing values
            close_prices = data['Close'].dropna()
            missing_ratio = (data_length - len(close_prices)) / data_length
            if missing_ratio > 0.1:  # More than 10% missing
                quality_issues.append(f"High missing data ratio: {missing_ratio:.1%}")
            
            # Check for extreme price movements (potential data errors)
            if len(close_prices) > 1:
                daily_returns = close_prices.pct_change().dropna()
                extreme_moves = daily_returns[abs(daily_returns) > self.max_price_change]
                if len(extreme_moves) > 0:
                    quality_issues.append(f"Extreme price movements detected: {len(extreme_moves)} days")
            
            # Check for zero or negative prices
            invalid_prices = close_prices[close_prices <= 0]
            if len(invalid_prices) > 0:
                quality_issues.append(f"Invalid prices detected: {len(invalid_prices)} negative/zero prices")
            
            # Volume validation (if available)
            if 'Volume' in data.columns:
                volumes = data['Volume'].dropna()
                avg_volume = volumes.mean() if len(volumes) > 0 else 0
                validation_result["data_stats"]["avg_volume"] = avg_volume
                
                if avg_volume < self.min_volume_threshold:
                    validation_result["warnings"].append(f"Low trading volume: {avg_volume:.0f} average")
            
            # Calculate quality score (0-100)
            quality_score = 100
            quality_score -= len(quality_issues) * 15  # Deduct for each issue
            quality_score -= missing_ratio * 50        # Deduct for missing data
            quality_score = max(0, min(100, quality_score))
            
            validation_result["quality_score"] = quality_score
            validation_result["issues"].extend(quality_issues)
            
            # Clean the data
            cleaned_data = self._clean_market_data(data, ticker)
            validation_result["cleaned_data"] = cleaned_data
            
            # Final validation decision
            if quality_score < 50:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Overall data quality too low")
            
            validation_result["data_stats"].update({
                "date_range": f"{data.index.min()} to {data.index.max()}",
                "latest_price": float(close_prices.iloc[-1]) if len(close_prices) > 0 else None,
                "price_range": {
                    "min": float(close_prices.min()),
                    "max": float(close_prices.max())
                } if len(close_prices) > 0 else None
            })
            
            logger.debug(f"✅ Data validation for {ticker}: {quality_score}/100 quality score")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating market data for {ticker}: {e}")
            return {
                "is_valid": False,
                "quality_score": 0,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "cleaned_data": None,
                "data_stats": {}
            }
    
    def _clean_ticker(self, ticker: str) -> Optional[str]:
        """Clean and standardize ticker symbol"""
        if not isinstance(ticker, str):
            return None
        
        # Basic cleaning
        cleaned = ticker.strip().upper()
        
        # Remove common prefixes/suffixes that might cause issues
        # while preserving valid formats
        if not cleaned:
            return None
        
        return cleaned
    
    def _validate_ticker_pattern(self, ticker: str) -> Dict[str, Any]:
        """Validate ticker against known patterns"""
        # Check against invalid patterns first
        for pattern in self.invalid_patterns:
            if re.match(pattern, ticker):
                return self._invalid_result(ticker, f"Invalid ticker pattern: {pattern}")
        
        # Check against valid patterns
        for pattern in self.valid_ticker_patterns:
            if re.match(pattern, ticker):
                return {"is_valid": True, "ticker": ticker, "reason": "Valid pattern"}
        
        # If no patterns match, it might still be valid (patterns aren't exhaustive)
        return {"is_valid": True, "ticker": ticker, "reason": "Pattern not recognized but may be valid"}
    
    def _validate_ticker_api(self, ticker: str) -> Dict[str, Any]:
        """Validate ticker using API call"""
        try:
            # Download minimal data to check if ticker exists
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid info
            if not info or info.get('regularMarketPrice') is None:
                # Try getting historical data as fallback
                hist = stock.history(period="5d")
                if hist.empty:
                    return self._invalid_result(ticker, "No market data available")
            
            # Extract useful information
            last_price = info.get('regularMarketPrice', info.get('previousClose'))
            market_cap = info.get('marketCap')
            exchange = info.get('exchange', 'Unknown')
            
            # Validate that it's a tradeable security
            if last_price is None or last_price <= 0:
                return self._invalid_result(ticker, "Invalid or missing price data")
            
            return {
                "is_valid": True,
                "ticker": ticker,
                "reason": "Valid tradeable security",
                "data_quality": "good",
                "last_price": last_price,
                "market_cap": market_cap,
                "exchange": exchange,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return self._invalid_result(ticker, f"API validation failed: {str(e)}")
    
    def _invalid_result(self, ticker: str, reason: str) -> Dict[str, Any]:
        """Create standardized invalid result"""
        return {
            "is_valid": False,
            "ticker": ticker,
            "reason": reason,
            "data_quality": "invalid",
            "last_price": None,
            "market_cap": None,
            "timestamp": datetime.now()
        }
    
    def _validate_target_value(self, value: Any) -> Dict[str, Any]:
        """Validate target value parameter"""
        try:
            float_value = float(value)
            
            if float_value <= 0:
                return {"is_valid": False, "value": 0, "message": "Target value must be positive"}
            
            if float_value < 100:
                return {"is_valid": False, "value": float_value, "message": "Target value too small (minimum £100)"}
            
            if float_value > 10_000_000:  # £10 million
                return {"is_valid": False, "value": float_value, "message": "Target value unrealistically large"}
            
            return {"is_valid": True, "value": float_value, "message": "Valid target value"}
            
        except (ValueError, TypeError):
            return {"is_valid": False, "value": 0, "message": "Target value must be a number"}
    
    def _validate_current_investment(self, value: Any) -> Dict[str, Any]:
        """Validate current investment parameter"""
        try:
            float_value = float(value)
            
            if float_value < 0:
                return {"is_valid": False, "value": 0, "message": "Current investment cannot be negative"}
            
            if float_value > 10_000_000:  # £10 million
                return {"is_valid": False, "value": float_value, "message": "Current investment unrealistically large"}
            
            return {"is_valid": True, "value": float_value, "message": "Valid current investment"}
            
        except (ValueError, TypeError):
            return {"is_valid": False, "value": 0, "message": "Current investment must be a number"}
    
    def _validate_timeframe(self, value: Any) -> Dict[str, Any]:
        """Validate timeframe parameter"""
        try:
            int_value = int(value)
            
            if int_value <= 0:
                return {"is_valid": False, "value": 1, "message": "Timeframe must be positive"}
            
            if int_value > 50:
                return {"is_valid": False, "value": int_value, "message": "Timeframe too long (maximum 50 years)"}
            
            return {"is_valid": True, "value": int_value, "message": "Valid timeframe"}
            
        except (ValueError, TypeError):
            return {"is_valid": False, "value": 10, "message": "Timeframe must be a whole number of years"}
    
    def _validate_monthly_contribution(self, value: Any) -> Dict[str, Any]:
        """Validate monthly contribution parameter"""
        try:
            float_value = float(value)
            
            if float_value < 0:
                return {"is_valid": False, "value": 0, "message": "Monthly contribution cannot be negative"}
            
            if float_value > 50_000:  # £50k per month seems unrealistic
                return {"is_valid": False, "value": float_value, "message": "Monthly contribution unrealistically large"}
            
            return {"is_valid": True, "value": float_value, "message": "Valid monthly contribution"}
            
        except (ValueError, TypeError):
            return {"is_valid": False, "value": 0, "message": "Monthly contribution must be a number"}
    
    def _validate_risk_score(self, value: Any) -> Dict[str, Any]:
        """Validate risk score parameter"""
        try:
            float_value = float(value)
            
            if float_value < 0 or float_value > 100:
                return {"is_valid": False, "value": 50, "message": "Risk score must be between 0 and 100"}
            
            return {"is_valid": True, "value": float_value, "message": "Valid risk score"}
            
        except (ValueError, TypeError):
            return {"is_valid": False, "value": 50, "message": "Risk score must be a number"}
    
    def _cross_validate_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Perform cross-validation checks between parameters"""
        warnings = []
        
        target = params.get('target_value', 0)
        current = params.get('current_investment', 0)
        monthly = params.get('monthly_contribution', 0)
        timeframe = params.get('timeframe', 10)
        
        # Check if goal is already achieved
        if current >= target:
            warnings.append("Current investment already meets or exceeds target")
        
        # Check if contributions alone will achieve goal
        total_contributions = current + (monthly * 12 * timeframe)
        if total_contributions >= target:
            warnings.append("Goal achievable through contributions alone - no investment return needed")
        
        # Check for very low contribution relative to goal
        if monthly > 0 and target > 0:
            contribution_ratio = (monthly * 12) / target
            if contribution_ratio < 0.01:  # Less than 1% of goal per year
                warnings.append("Monthly contributions very low relative to target - may need higher returns")
        
        return {"warnings": warnings}
    
    def _clean_market_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and preprocess market data"""
        try:
            cleaned_data = data.copy()
            
            # Remove rows with missing close prices
            cleaned_data = cleaned_data.dropna(subset=['Close'])
            
            # Remove rows with invalid prices
            cleaned_data = cleaned_data[cleaned_data['Close'] > 0]
            
            # Sort by date
            cleaned_data = cleaned_data.sort_index()
            
            # Remove extreme outliers (potential data errors)
            if len(cleaned_data) > 1:
                returns = cleaned_data['Close'].pct_change()
                outlier_threshold = 3 * returns.std()  # 3 standard deviations
                outliers = abs(returns) > outlier_threshold
                
                if outliers.sum() > 0:
                    logger.debug(f"Removing {outliers.sum()} outlier data points for {ticker}")
                    cleaned_data = cleaned_data[~outliers]
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning market data for {ticker}: {e}")
            return data  # Return original data if cleaning fails
    
    def _is_cache_valid(self, timestamp: Optional[datetime]) -> bool:
        """Check if cached result is still valid"""
        if timestamp is None:
            return False
        
        time_elapsed = (datetime.now() - timestamp).total_seconds()
        return time_elapsed < self.cache_timeout


# Utility functions for easy access
def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Quick utility function to validate a list of tickers and return valid ones
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List of valid ticker symbols
    """
    validator = DataValidator()
    result = validator.validate_stock_list(tickers)
    return result['valid_tickers']

def validate_financial_inputs(**kwargs) -> Dict[str, Any]:
    """
    Quick utility function to validate financial parameters
    
    Args:
        **kwargs: Financial parameters to validate
        
    Returns:
        Validation results dictionary
    """
    validator = DataValidator()
    return validator.validate_financial_parameters(**kwargs)

def clean_ticker_list(tickers: List[str]) -> List[str]:
    """
    Clean and standardize a list of ticker symbols
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        List of cleaned ticker symbols
    """
    validator = DataValidator()
    cleaned = []
    
    for ticker in tickers:
        cleaned_ticker = validator._clean_ticker(ticker)
        if cleaned_ticker:
            cleaned.append(cleaned_ticker)
    
    return list(set(cleaned))  # Remove duplicates