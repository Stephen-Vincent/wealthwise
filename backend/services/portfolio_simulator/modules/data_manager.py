"""
Data Manager Module

This module handles stock data download, validation, and preparation
for portfolio simulations with enhanced error handling and quality checks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class StockDataManager:
    """
    Manages stock data download and preparation for portfolio simulations.
    
    Features:
    - Robust data download with retry logic
    - Comprehensive data quality validation
    - Missing data handling
    - Performance optimization
    - Detailed logging and error reporting
    """
    
    def __init__(self, max_workers: int = 5, max_retries: int = 3):
        """
        Initialize the stock data manager.
        
        Args:
            max_workers: Maximum number of concurrent download threads
            max_retries: Maximum number of retry attempts for failed downloads
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"📊 StockDataManager initialized with {max_workers} workers")
    
    async def download_stock_data(self, tickers: List[str], timeframe: int, 
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download historical stock data for portfolio simulation.
        
        Args:
            tickers: List of stock symbols to download
            timeframe: Investment timeframe in years
            end_date: Optional end date (defaults to today)
            
        Returns:
            DataFrame with historical closing prices
        """
        
        try:
            logger.info(f"📊 Downloading data for {len(tickers)} stocks over {timeframe} years")
            
            # Calculate date range
            start_date, end_date = self._calculate_date_range(timeframe, end_date)
            
            # Download data with retry logic
            raw_data = await self._download_with_retry(tickers, start_date, end_date)
            
            # Validate and clean data
            cleaned_data = self._validate_and_clean_data(raw_data, tickers)
            
            # Perform quality checks
            self._perform_quality_checks(cleaned_data, tickers, timeframe)
            
            logger.info(f"✅ Data download completed - shape: {cleaned_data.shape}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Error downloading stock data: {str(e)}")
            raise ValueError(f"Failed to download stock data: {str(e)}")
    
    def _calculate_date_range(self, timeframe: int, end_date: Optional[str] = None) -> Tuple[str, str]:
        """
        Calculate start and end dates for data download.
        
        Args:
            timeframe: Investment timeframe in years
            end_date: Optional end date
            
        Returns:
            Tuple of (start_date, end_date) as strings
        """
        
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = datetime.today()
        
        # Add buffer to ensure we have enough data
        days_needed = max(timeframe * 365 + 90, 365)  # At least 1 year + buffer
        start_dt = end_dt - timedelta(days=days_needed)
        
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        
        logger.info(f"📅 Data range: {start_date} to {end_date}")
        return start_date, end_date
    
    async def _download_with_retry(self, tickers: List[str], start_date: str, 
                                  end_date: str) -> pd.DataFrame:
        """
        Download data with retry logic and error handling.
        """
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📡 Download attempt {attempt + 1}/{self.max_retries}")
                
                # Use asyncio to run yfinance download in executor
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    self.executor,
                    self._download_yfinance_data,
                    tickers, start_date, end_date
                )
                
                if data is not None and not data.empty:
                    logger.info("✅ Data download successful")
                    return data
                else:
                    logger.warning(f"⚠️ Empty data received on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error("❌ All download attempts failed")
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise ValueError("Failed to download data after all retry attempts")
    
    def _download_yfinance_data(self, tickers: List[str], start_date: str, 
                               end_date: str) -> pd.DataFrame:
        """
        Actual yfinance download function (runs in executor).
        """
        
        try:
            # Download data using yfinance
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False,
                threads=True,
                group_by='ticker' if len(tickers) > 1 else None
            )
            
            # Extract closing prices
            if len(tickers) == 1:
                # Single ticker case
                if 'Close' in data.columns:
                    return data[['Close']].rename(columns={'Close': tickers[0]})
                else:
                    return data.to_frame(name=tickers[0])
            else:
                # Multiple tickers case
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns: extract Close prices
                    close_data = data.xs('Close', level=1, axis=1)
                    return close_data
                else:
                    # Single level columns (fallback)
                    return data
                    
        except Exception as e:
            logger.error(f"❌ yfinance download error: {e}")
            raise
    
    def _validate_and_clean_data(self, data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Validate and clean downloaded data.
        
        Args:
            data: Raw downloaded data
            tickers: List of expected tickers
            
        Returns:
            Cleaned and validated DataFrame
        """
        
        try:
            logger.info("🧹 Validating and cleaning data")
            
            # Ensure we have a DataFrame
            if data is None or data.empty:
                raise ValueError("Downloaded data is empty")
            
            # Convert to DataFrame if Series
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Ensure column names match tickers
            if len(data.columns) != len(tickers):
                logger.warning(f"⚠️ Column count mismatch: got {len(data.columns)}, expected {len(tickers)}")
                
                # Try to map columns to tickers
                if len(data.columns) == len(tickers):
                    data.columns = tickers
                elif len(data.columns) > len(tickers):
                    # Take first N columns
                    data = data.iloc[:, :len(tickers)]
                    data.columns = tickers
                else:
                    # Pad with NaN columns if needed
                    for i, ticker in enumerate(tickers):
                        if i >= len(data.columns):
                            data[ticker] = np.nan
            
            # Ensure correct column names
            data.columns = tickers
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill and backward fill missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Remove columns with too many missing values (>50%)
            missing_threshold = len(data) * 0.5
            data = data.dropna(axis=1, thresh=missing_threshold)
            
            # Ensure we have numeric data
            data = data.apply(pd.to_numeric, errors='coerce')
            
            # Remove any remaining rows with NaN values
            data = data.dropna()
            
            # Validate that we have positive prices
            if (data <= 0).any().any():
                logger.warning("⚠️ Found zero or negative prices, removing affected rows")
                data = data[(data > 0).all(axis=1)]
            
            # Sort by date
            data = data.sort_index()
            
            logger.info(f"✅ Data cleaned - final shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error validating data: {e}")
            raise ValueError(f"Data validation failed: {e}")
    
    def _perform_quality_checks(self, data: pd.DataFrame, tickers: List[str], 
                               timeframe: int):
        """
        Perform comprehensive quality checks on the data.
        """
        
        logger.info("🔍 Performing data quality checks")
        
        # Check minimum data requirements
        min_days_required = timeframe * 252  # Approximate trading days per year
        if len(data) < min_days_required * 0.7:  # Allow 30% flexibility
            logger.warning(f"⚠️ Limited data: {len(data)} days (expected ~{min_days_required})")
        
        # Check for tickers that were completely removed
        missing_tickers = set(tickers) - set(data.columns)
        if missing_tickers:
            logger.warning(f"⚠️ Missing tickers after cleaning: {missing_tickers}")
        
        # Check for suspicious price movements (>50% daily change)
        for ticker in data.columns:
            daily_returns = data[ticker].pct_change()
            extreme_moves = daily_returns.abs() > 0.5
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                logger.warning(f"⚠️ {ticker}: {extreme_count} extreme daily moves (>50%)")
        
        # Check data freshness
        latest_date = data.index.max()
        days_old = (datetime.now() - latest_date).days
        if days_old > 7:
            logger.warning(f"⚠️ Data is {days_old} days old (latest: {latest_date.date()})")
        
        # Check for gaps in data
        expected_days = (data.index.max() - data.index.min()).days
        actual_days = len(data)
        coverage = actual_days / (expected_days / 7 * 5)  # Approximate for weekdays only
        
        if coverage < 0.8:
            logger.warning(f"⚠️ Data coverage is {coverage:.1%} (may have significant gaps)")
        
        logger.info("✅ Quality checks completed")
    
    async def get_stock_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get additional information about stocks.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary with stock information
        """
        
        try:
            logger.info(f"📋 Getting stock info for {len(tickers)} tickers")
            
            stock_info = {}
            
            # Download info in parallel
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, self._get_single_stock_info, ticker)
                for ticker in tickers
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(tickers, results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Failed to get info for {ticker}: {result}")
                    stock_info[ticker] = self._get_fallback_stock_info(ticker)
                else:
                    stock_info[ticker] = result
            
            logger.info("✅ Stock info retrieval completed")
            return stock_info
            
        except Exception as e:
            logger.error(f"❌ Error getting stock info: {e}")
            return {ticker: self._get_fallback_stock_info(ticker) for ticker in tickers}
    
    def _get_single_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get information for a single stock.
        """
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "pe_ratio": info.get("trailingPE"),
                "description": info.get("longBusinessSummary", "")[:200] + "..." if info.get("longBusinessSummary") else ""
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error getting info for {ticker}: {e}")
            return self._get_fallback_stock_info(ticker)
    
    def _get_fallback_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get fallback stock information when API fails.
        """
        
        # Basic ETF and common stock information
        fallback_info = {
            "VTI": {"name": "Vanguard Total Stock Market ETF", "sector": "Equity", "industry": "Index Fund"},
            "BND": {"name": "Vanguard Total Bond Market ETF", "sector": "Fixed Income", "industry": "Index Fund"},
            "VEA": {"name": "Vanguard FTSE Developed Markets ETF", "sector": "International Equity", "industry": "Index Fund"},
            "VTEB": {"name": "Vanguard Tax-Exempt Bond ETF", "sector": "Fixed Income", "industry": "Index Fund"},
            "VWO": {"name": "Vanguard Emerging Markets ETF", "sector": "Emerging Markets", "industry": "Index Fund"},
            "VNQ": {"name": "Vanguard Real Estate ETF", "sector": "Real Estate", "industry": "REIT Fund"},
            "VGT": {"name": "Vanguard Information Technology ETF", "sector": "Technology", "industry": "Sector Fund"},
            "VUG": {"name": "Vanguard Growth ETF", "sector": "Equity", "industry": "Growth Fund"},
            "ARKK": {"name": "ARK Innovation ETF", "sector": "Technology", "industry": "Active Fund"}
        }
        
        if ticker in fallback_info:
            return {
                **fallback_info[ticker],
                "market_cap": None,
                "dividend_yield": None,
                "beta": None,
                "pe_ratio": None,
                "description": f"Exchange-traded fund tracking {fallback_info[ticker]['industry'].lower()} investments"
            }
        
        return {
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": None,
            "dividend_yield": None,
            "beta": None,
            "pe_ratio": None,
            "description": f"Information not available for {ticker}"
        }
    
    def calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics about the data.
        
        Args:
            data: Stock price data
            
        Returns:
            Dictionary with data statistics
        """
        
        try:
            logger.info("📊 Calculating data statistics")
            
            # Basic statistics
            stats = {
                "data_range": {
                    "start_date": data.index.min().strftime("%Y-%m-%d"),
                    "end_date": data.index.max().strftime("%Y-%m-%d"),
                    "total_days": len(data),
                    "trading_years": len(data) / 252
                },
                "tickers": {
                    "count": len(data.columns),
                    "symbols": list(data.columns)
                },
                "data_quality": {
                    "missing_values": data.isnull().sum().sum(),
                    "complete_rows": len(data.dropna()),
                    "completeness_ratio": len(data.dropna()) / len(data)
                }
            }
            
            # Calculate returns and volatility for each ticker
            returns_data = data.pct_change().dropna()
            
            ticker_stats = {}
            for ticker in data.columns:
                ticker_returns = returns_data[ticker]
                
                ticker_stats[ticker] = {
                    "total_return": (data[ticker].iloc[-1] / data[ticker].iloc[0] - 1) * 100,
                    "annualized_return": ((data[ticker].iloc[-1] / data[ticker].iloc[0]) ** (252 / len(data)) - 1) * 100,
                    "volatility": ticker_returns.std() * np.sqrt(252) * 100,
                    "max_drawdown": self._calculate_max_drawdown(data[ticker]) * 100,
                    "sharpe_ratio": self._calculate_sharpe_ratio(ticker_returns),
                    "best_day": ticker_returns.max() * 100,
                    "worst_day": ticker_returns.min() * 100
                }
            
            stats["ticker_performance"] = ticker_stats
            
            # Portfolio correlation matrix
            correlation_matrix = returns_data.corr()
            stats["correlations"] = {
                "average_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                "max_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                "min_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
            }
            
            logger.info("✅ Data statistics calculated")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating statistics: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a returns series."""
        
        try:
            excess_returns = returns.mean() * 252 - risk_free_rate
            volatility = returns.std() * np.sqrt(252)
            
            if volatility == 0:
                return 0.0
            
            return excess_returns / volatility
            
        except Exception:
            return 0.0
    
    async def validate_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Validate that tickers exist and can be downloaded.
        
        Args:
            tickers: List of stock symbols to validate
            
        Returns:
            Validation results
        """
        
        try:
            logger.info(f"✅ Validating {len(tickers)} tickers")
            
            # Try to download a small sample of recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            validation_results = {
                "valid_tickers": [],
                "invalid_tickers": [],
                "warnings": []
            }
            
            for ticker in tickers:
                try:
                    # Test download for each ticker individually
                    loop = asyncio.get_event_loop()
                    test_data = await loop.run_in_executor(
                        self.executor,
                        self._test_ticker_download,
                        ticker, start_date, end_date
                    )
                    
                    if test_data is not None and not test_data.empty:
                        validation_results["valid_tickers"].append(ticker)
                        
                        # Check for data quality issues
                        if len(test_data) < 15:  # Less than 15 days of data
                            validation_results["warnings"].append(f"{ticker}: Limited recent data")
                    else:
                        validation_results["invalid_tickers"].append(ticker)
                        logger.warning(f"⚠️ Invalid ticker: {ticker}")
                        
                except Exception as e:
                    validation_results["invalid_tickers"].append(ticker)
                    logger.warning(f"⚠️ Ticker validation failed for {ticker}: {e}")
            
            # Summary
            validation_results["summary"] = {
                "total_tickers": len(tickers),
                "valid_count": len(validation_results["valid_tickers"]),
                "invalid_count": len(validation_results["invalid_tickers"]),
                "validation_rate": len(validation_results["valid_tickers"]) / len(tickers)
            }
            
            logger.info(f"✅ Validation complete: {validation_results['summary']['valid_count']}/{len(tickers)} valid")
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating tickers: {e}")
            return {
                "valid_tickers": tickers,  # Assume all valid as fallback
                "invalid_tickers": [],
                "warnings": [f"Validation failed: {e}"],
                "summary": {"validation_rate": 1.0}
            }
    
    def _test_ticker_download(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Test download for a single ticker."""
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            return data
        except Exception:
            return None
    
    def get_alternative_tickers(self, failed_tickers: List[str]) -> Dict[str, str]:
        """
        Suggest alternative tickers for failed downloads.
        
        Args:
            failed_tickers: List of tickers that failed to download
            
        Returns:
            Dictionary mapping failed tickers to suggested alternatives
        """
        
        # Common alternatives for popular tickers/ETFs
        alternatives = {
            # If VTI fails, suggest other total market funds
            "VTI": "ITOT",  # iShares Core S&P Total Market
            "VTSAX": "VTI",  # ETF version of mutual fund
            
            # Bond alternatives
            "BND": "AGG",   # iShares Core Aggregate Bond
            "VBTLX": "BND", # ETF version
            
            # International alternatives
            "VEA": "IEFA",  # iShares Core MSCI EAFE
            "VTIAX": "VEA", # ETF version
            
            # Emerging markets
            "VWO": "IEMG",  # iShares Core MSCI Emerging Markets
            "VEMAX": "VWO", # ETF version
            
            # Technology
            "VGT": "XLK",   # Technology Select Sector
            "QQQ": "VGT",   # Alternative tech exposure
            
            # Real estate
            "VNQ": "IYR",   # iShares Real Estate
            "VGSLX": "VNQ", # ETF version
        }
        
        suggestions = {}
        for ticker in failed_tickers:
            if ticker in alternatives:
                suggestions[ticker] = alternatives[ticker]
                logger.info(f"💡 Suggesting {alternatives[ticker]} as alternative to {ticker}")
        
        return suggestions
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("🧹 Data manager resources cleaned up")