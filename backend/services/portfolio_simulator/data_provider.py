"""
Market data provider for the Portfolio Simulator Service.

This module handles downloading and processing historical market data
from various sources with proper error handling and data quality validation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .config import get_config, get_stock_metadata
from .exceptions import DataProviderError, InsufficientDataError
from .validators import InputValidator

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Handles market data retrieval and processing with robust error handling.
    
    This class provides methods to download historical stock data, validate
    data quality, and handle various data provider failures gracefully.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the market data provider.
        
        Args:
            validator: Input validator instance for symbol validation
        """
        self.config = get_config()
        self.validator = validator or InputValidator()
        self.stock_metadata = get_stock_metadata()
        
        # Data quality thresholds
        self.min_data_completeness = 0.7  # 70% of expected data points
        self.max_missing_consecutive_days = 10
        self.min_price_threshold = 0.01  # Minimum valid price
        
        # Rate limiting for API calls
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def download_stock_data(
        self, 
        tickers: List[str], 
        timeframe_years: int,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Download historical stock data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            timeframe_years: Number of years of historical data needed
            end_date: End date for data download (defaults to today)
            
        Returns:
            DataFrame with historical closing prices for all tickers
            
        Raises:
            DataProviderError: If data download fails
            InsufficientDataError: If insufficient data is available
        """
        # Validate input parameters
        validated_tickers = self.validator.validate_ticker_symbols(tickers)
        
        if not validated_tickers:
            raise DataProviderError(
                "No valid ticker symbols provided",
                provider="yfinance",
                symbols=tickers
            )
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        
        # Add buffer for weekends/holidays and ensure sufficient data
        buffer_days = max(90, timeframe_years * 30)  # At least 90 days buffer
        start_date = end_date - timedelta(days=timeframe_years * 365 + buffer_days)
        
        logger.info(
            f"Downloading data for {len(validated_tickers)} tickers "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        try:
            # Download data with retry logic
            data = self._download_with_retry(
                validated_tickers, start_date, end_date
            )
            
            if data is None or data.empty:
                raise DataProviderError(
                    "No data returned from provider",
                    provider="yfinance",
                    symbols=validated_tickers
                )
            
            # Process and validate the downloaded data
            processed_data = self._process_raw_data(data, validated_tickers)
            
            # Validate data quality
            self._validate_data_quality(
                processed_data, timeframe_years, validated_tickers
            )
            
            logger.info(
                f"Successfully downloaded data: {processed_data.shape[0]} days, "
                f"{processed_data.shape[1]} symbols"
            )
            
            return processed_data
            
        except Exception as e:
            if isinstance(e, (DataProviderError, InsufficientDataError)):
                raise
            
            logger.error(f"Unexpected error downloading stock data: {str(e)}")
            raise DataProviderError(
                f"Failed to download stock data: {str(e)}",
                provider="yfinance",
                symbols=validated_tickers
            )
    
    def _download_with_retry(
        self, 
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Download data with retry logic and rate limiting.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            max_retries: Maximum number of retry attempts
            
        Returns:
            Raw DataFrame from yfinance or None if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._enforce_rate_limit()
                
                logger.debug(f"Download attempt {attempt + 1} for tickers: {tickers}")
                
                # Download data using yfinance
                data = yf.download(
                    tickers,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False,
                    threads=True,
                    group_by='ticker' if len(tickers) > 1 else None,
                    auto_adjust=True,  # Adjust for splits and dividends
                    prepost=False,  # Only regular trading hours
                    actions=False   # Don't need dividend/split data
                )
                
                if data is not None and not data.empty:
                    logger.debug(f"Download successful on attempt {attempt + 1}")
                    return data
                
                logger.warning(f"Empty data returned on attempt {attempt + 1}")
                
            except Exception as e:
                logger.warning(
                    f"Download attempt {attempt + 1} failed: {str(e)}"
                )
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 1.0
                    logger.debug(f"Waiting {wait_time} seconds before retry")
                    time.sleep(wait_time)
        
        logger.error(f"All {max_retries} download attempts failed")
        return None
    
    def _process_raw_data(
        self, 
        raw_data: pd.DataFrame, 
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Process raw data from yfinance into clean format.
        
        Args:
            raw_data: Raw DataFrame from yfinance
            tickers: List of expected ticker symbols
            
        Returns:
            Processed DataFrame with closing prices
            
        Raises:
            DataProviderError: If data processing fails
        """
        try:
            # Handle different data structures from yfinance
            if len(tickers) == 1:
                # Single ticker: data is a simple DataFrame
                if 'Close' in raw_data.columns:
                    processed = raw_data[['Close']].copy()
                    processed.columns = tickers
                else:
                    raise DataProviderError(
                        "Close price column not found in single ticker data",
                        provider="yfinance"
                    )
            else:
                # Multiple tickers: data has MultiIndex columns
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # Extract Close prices for all tickers
                    close_data = []
                    for ticker in tickers:
                        if ('Close', ticker) in raw_data.columns:
                            close_data.append(raw_data[('Close', ticker)])
                        elif ticker in raw_data.columns:
                            # Sometimes yfinance returns flattened structure
                            close_data.append(raw_data[ticker])
                    
                    if close_data:
                        processed = pd.concat(close_data, axis=1, keys=tickers)
                    else:
                        raise DataProviderError(
                            "No Close price data found for any ticker",
                            provider="yfinance",
                            symbols=tickers
                        )
                else:
                    # Fallback: assume columns are tickers
                    available_tickers = [t for t in tickers if t in raw_data.columns]
                    if available_tickers:
                        processed = raw_data[available_tickers].copy()
                    else:
                        raise DataProviderError(
                            "No matching ticker columns found",
                            provider="yfinance",
                            symbols=tickers
                        )
            
            # Clean the data
            processed = self._clean_price_data(processed)
            
            # Ensure we have at least some valid data
            if processed.empty:
                raise DataProviderError(
                    "All data was filtered out during cleaning",
                    provider="yfinance",
                    symbols=tickers
                )
            
            return processed
            
        except Exception as e:
            if isinstance(e, DataProviderError):
                raise
            
            logger.error(f"Error processing raw data: {str(e)}")
            raise DataProviderError(
                f"Failed to process market data: {str(e)}",
                provider="yfinance",
                symbols=tickers
            )
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean price data by handling missing values and outliers.
        
        Args:
            data: Raw price data
            
        Returns:
            Cleaned price data
        """
        logger.debug("Cleaning price data")
        
        # Remove rows where all values are NaN
        data = data.dropna(how='all')
        
        # Replace zero and negative prices with NaN
        data = data.where(data > self.min_price_threshold, np.nan)
        
        # Handle extreme outliers (price changes > 50% in one day are suspicious)
        for column in data.columns:
            pct_change = data[column].pct_change().abs()
            extreme_changes = pct_change > 0.5
            
            if extreme_changes.any():
                logger.warning(
                    f"Found {extreme_changes.sum()} extreme price changes for {column}"
                )
                # Keep the data but log the warning
                # In production, you might want more sophisticated outlier detection
        
        # Forward fill missing values (up to max_missing_consecutive_days)
        data = data.ffill(limit=self.max_missing_consecutive_days)
        
        # Remove columns that are still mostly NaN after cleaning
        threshold = len(data) * self.min_data_completeness
        data = data.dropna(axis=1, thresh=int(threshold))
        
        logger.debug(f"Data cleaning complete: {data.shape}")
        
        return data
    
    def _validate_data_quality(
        self, 
        data: pd.DataFrame, 
        timeframe_years: int, 
        requested_tickers: List[str]
    ) -> None:
        """
        Validate that downloaded data meets quality requirements.
        
        Args:
            data: Processed price data
            timeframe_years: Required timeframe in years
            requested_tickers: Originally requested ticker symbols
            
        Raises:
            InsufficientDataError: If data quality is insufficient
        """
        if data.empty:
            raise InsufficientDataError(
                "No valid data available after cleaning",
                required_days=timeframe_years * self.config.simulation.trading_days_per_year,
                available_days=0
            )
        
        # Check if we have enough historical data
        required_days = timeframe_years * self.config.simulation.trading_days_per_year
        available_days = len(data)
        
        if available_days < required_days * 0.8:  # Allow 20% shortage
            raise InsufficientDataError(
                f"Insufficient historical data: need {required_days} days, "
                f"got {available_days} days",
                required_days=required_days,
                available_days=available_days
            )
        
        # Check if we lost too many tickers during processing
        missing_tickers = set(requested_tickers) - set(data.columns)
        if len(missing_tickers) > len(requested_tickers) * 0.5:  # Lost more than 50%
            raise InsufficientDataError(
                f"Too many tickers unavailable: {list(missing_tickers)}",
                symbols=list(missing_tickers)
            )
        
        # Check data completeness for each remaining ticker
        for ticker in data.columns:
            ticker_data = data[ticker]
            completeness = ticker_data.notna().sum() / len(ticker_data)
            
            if completeness < self.min_data_completeness:
                logger.warning(
                    f"Low data completeness for {ticker}: {completeness:.1%}"
                )
        
        logger.info(
            f"Data quality validation passed: {available_days} days, "
            f"{len(data.columns)} tickers, "
            f"{len(missing_tickers)} missing tickers"
        )
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_supported_tickers(self) -> List[str]:
        """
        Get list of supported ticker symbols.
        
        Returns:
            List of supported ticker symbols
        """
        return list(self.stock_metadata.keys())
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """
        Get metadata for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker metadata or None if not found
        """
        return self.stock_metadata.get(ticker.upper())
    
    def estimate_data_availability(
        self, 
        tickers: List[str], 
        timeframe_years: int
    ) -> Dict[str, bool]:
        """
        Estimate data availability without downloading full dataset.
        
        Args:
            tickers: List of ticker symbols
            timeframe_years: Required timeframe in years
            
        Returns:
            Dictionary mapping tickers to availability status
        """
        availability = {}
        
        for ticker in tickers:
            # Check if ticker is in our known list
            if ticker.upper() in self.stock_metadata:
                # For known tickers, assume data is available
                # In production, you might want to check actual availability
                availability[ticker] = True
            else:
                # Unknown tickers are less reliable
                availability[ticker] = False
        
        return availability


class DataQualityAnalyzer:
    """
    Analyzes the quality and characteristics of market data.
    
    This class provides methods to assess data quality, detect anomalies,
    and provide insights about the downloaded data.
    """
    
    def __init__(self):
        """Initialize the data quality analyzer."""
        self.config = get_config()
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            data: Historical price data
            
        Returns:
            Dictionary with data quality metrics
        """
        analysis = {
            'overall_quality': 'good',
            'data_points': len(data),
            'date_range': {
                'start': data.index.min().isoformat() if not data.empty else None,
                'end': data.index.max().isoformat() if not data.empty else None,
                'days': len(data)
            },
            'tickers': list(data.columns),
            'completeness': {},
            'volatility': {},
            'anomalies': [],
            'recommendations': []
        }
        
        if data.empty:
            analysis['overall_quality'] = 'poor'
            analysis['recommendations'].append("No data available")
            return analysis
        
        # Analyze each ticker
        for ticker in data.columns:
            ticker_data = data[ticker].dropna()
            
            if len(ticker_data) == 0:
                analysis['completeness'][ticker] = 0.0
                analysis['anomalies'].append(f"{ticker}: No valid data points")
                continue
            
            # Completeness
            completeness = len(ticker_data) / len(data)
            analysis['completeness'][ticker] = round(completeness, 3)
            
            # Volatility (annualized standard deviation)
            returns = ticker_data.pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std() * np.sqrt(252)  # Annualized
                analysis['volatility'][ticker] = round(volatility, 3)
            
            # Check for anomalies
            if completeness < 0.8:
                analysis['anomalies'].append(
                    f"{ticker}: Low data completeness ({completeness:.1%})"
                )
            
            # Check for extreme volatility
            if len(returns) > 1 and volatility > 1.0:  # >100% annualized volatility
                analysis['anomalies'].append(
                    f"{ticker}: Very high volatility ({volatility:.1%})"
                )
        
        # Overall quality assessment
        avg_completeness = np.mean(list(analysis['completeness'].values()))
        
        if avg_completeness < 0.7:
            analysis['overall_quality'] = 'poor'
            analysis['recommendations'].append(
                "Consider using different tickers with better data availability"
            )
        elif avg_completeness < 0.9:
            analysis['overall_quality'] = 'fair'
            analysis['recommendations'].append(
                "Some data gaps detected - results may be less accurate"
            )
        
        if len(analysis['anomalies']) > len(data.columns) * 0.5:
            analysis['overall_quality'] = 'poor'
            analysis['recommendations'].append(
                "Multiple data quality issues detected"
            )
        
        return analysis