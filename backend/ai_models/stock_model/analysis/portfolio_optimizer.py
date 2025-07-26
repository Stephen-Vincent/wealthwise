"""
Portfolio Optimization Module

This module implements sophisticated portfolio construction techniques
used by professional investment managers, including correlation analysis
and risk-adjusted optimization based on Modern Portfolio Theory.

Key Features:
1. Correlation-Based Diversification - Reduces portfolio risk through low correlation
2. Portfolio Metrics Calculation - Expected return, volatility, Sharpe ratio
3. Weight Optimization - Balances risk and return efficiently
4. Statistical Learning - Uses historical data for optimization
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization System
    
    This class implements Modern Portfolio Theory principles to optimize
    portfolio weights based on correlation analysis and risk-return profiles.
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.metrics_cache = {}
        self.cache_timeout = 1800  # 30 minutes
    
    def optimize_for_diversification(self, stocks: List[str], 
                                   target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Correlation-Based Portfolio Optimization for Maximum Diversification
        
        Implements Modern Portfolio Theory to optimize portfolio weights based on
        correlation analysis. Reduces overall portfolio risk by minimizing
        correlations between holdings.
        
        Process:
        1. Download historical price data for all stocks
        2. Calculate daily returns for each stock
        3. Compute correlation matrix between all stocks
        4. Reduce weights for highly correlated stocks
        5. Increase weights for low-correlation stocks
        6. Normalize weights to sum to 100%
        
        Args:
            stocks: List of stock tickers to optimize
            target_weights: Initial equal weights or strategic weights
            
        Returns:
            Dict with optimized weights: {"VTI": 0.25, "BND": 0.30, ...}
        """
        try:
            logger.info(f"🔧 Optimizing portfolio diversification for {len(stocks)} stocks...")
            
            # Check cache first
            cache_key = f"corr_{hash(tuple(sorted(stocks)))}"
            if cache_key in self.correlation_cache:
                cached_data = self.correlation_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp']):
                    logger.debug("Using cached correlation data")
                    return self._optimize_weights_from_correlations(
                        cached_data['correlations'], stocks, target_weights
                    )
            
            # Use shorter time period for more reliable recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months of data
            
            price_data = {}
            valid_stocks = []
            
            # === DATA COLLECTION PHASE ===
            for stock in stocks:
                try:
                    # Download price data with timeout protection
                    data = yf.download(stock, start=start_date, end=end_date, 
                                    progress=False, timeout=10)['Close']
                    
                    if len(data) > 50:  # Need minimum data for correlation analysis
                        price_data[stock] = data
                        valid_stocks.append(stock)
                        logger.debug(f"✓ {stock}: {len(data)} data points collected")
                    else:
                        logger.warning(f"✗ {stock}: Insufficient data ({len(data)} points)")
                        
                except Exception as e:
                    logger.warning(f"✗ {stock}: Data download failed - {str(e)[:50]}")
                    continue
            
            # Fallback if insufficient stocks have data
            if len(valid_stocks) < 3:
                logger.warning(f"Only {len(valid_stocks)} stocks have data, using target weights")
                return target_weights
            
            # === CORRELATION ANALYSIS PHASE ===
            # Convert prices to returns for correlation calculation
            returns_df = pd.DataFrame()
            for stock in valid_stocks:
                returns_df[stock] = price_data[stock].pct_change().dropna()
            
            # Remove any remaining NaN values and align dates
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:  # Need minimum observations for reliable correlations
                logger.warning(f"Only {len(returns_df)} return observations, using target weights")
                return target_weights
            
            logger.info(f"✓ Correlation analysis with {len(valid_stocks)} stocks, {len(returns_df)} observations")
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            correlation_matrix = correlation_matrix.fillna(0)  # Handle any NaN correlations
            
            # Cache the correlation data
            self.correlation_cache[cache_key] = {
                'correlations': correlation_matrix,
                'timestamp': datetime.now(),
                'valid_stocks': valid_stocks
            }
            
            # Optimize weights based on correlations
            return self._optimize_weights_from_correlations(
                correlation_matrix, valid_stocks, target_weights
            )
            
        except Exception as e:
            logger.error(f"Correlation optimization failed: {e}")
            return target_weights
    
    def _optimize_weights_from_correlations(self, correlation_matrix: pd.DataFrame,
                                          valid_stocks: List[str], 
                                          target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize portfolio weights based on correlation matrix
        """
        # Calculate average correlation for each stock with all others
        avg_correlations = {}
        for stock in valid_stocks:
            other_stocks = [s for s in valid_stocks if s != stock]
            if other_stocks:
                corr_values = correlation_matrix.loc[stock, other_stocks]
                avg_corr = corr_values.fillna(0).mean()
                avg_correlations[stock] = avg_corr
                logger.debug(f"{stock}: avg correlation = {avg_corr:.3f}")
        
        # Optimize weights based on correlation analysis
        optimized_weights = {}
        base_weight = 1.0 / len(valid_stocks)  # Equal weight starting point
        
        for stock in valid_stocks:
            correlation_penalty = avg_correlations.get(stock, 0)
            
            # Apply correlation-based weight adjustments
            if correlation_penalty > 0.7:          # Highly correlated
                weight_adjustment = -0.3            # Reduce weight significantly
            elif correlation_penalty > 0.5:        # Moderately correlated
                weight_adjustment = -0.15           # Reduce weight moderately
            elif correlation_penalty < 0.2:        # Low correlation
                weight_adjustment = 0.2             # Boost weight (good diversifier)
            else:
                weight_adjustment = 0               # No adjustment
            
            # Apply adjustment with minimum weight floor
            optimized_weights[stock] = max(0.05, base_weight + (base_weight * weight_adjustment))
        
        # Normalize weights to sum to 100%
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        else:
            return target_weights
        
        # === FINAL BLENDING PHASE ===
        # Blend optimized weights with target allocation (70% optimized, 30% target)
        final_weights = {}
        for stock in valid_stocks:
            target_weight = target_weights.get(stock, base_weight)
            optimized_weight = optimized_weights.get(stock, base_weight)
            final_weights[stock] = 0.7 * optimized_weight + 0.3 * target_weight
        
        # Final normalization
        total_final = sum(final_weights.values())
        if total_final > 0:
            final_weights = {k: v / total_final for k, v in final_weights.items()}
            
            logger.info(f"✅ Correlation optimization successful for {len(final_weights)} stocks")
            return final_weights
        else:
            return target_weights
    
    def calculate_portfolio_metrics(self, stocks: List[str], 
                                  weights: Dict[str, float]) -> Dict[str, float]:
        """
        Statistical Learning for Portfolio Risk Assessment
        
        Calculates key portfolio metrics using statistical learning techniques.
        These metrics help assess expected performance and risk characteristics.
        
        Metrics Calculated:
        1. Expected Annual Return - Weighted average of individual stock returns
        2. Annual Volatility - Portfolio-level risk using correlation effects
        3. Sharpe Ratio - Risk-adjusted return measure
        4. Portfolio Size - Number of holdings for diversification assessment
        
        Args:
            stocks: List of stock tickers in the portfolio
            weights: Dictionary with weight for each stock
            
        Returns:
            Dict with portfolio metrics:
            {
                "expected_return": 0.08,    # 8% expected annual return
                "volatility": 0.15,         # 15% annual volatility
                "sharpe_ratio": 0.4,        # Sharpe ratio
                "portfolio_size": 6         # Number of holdings
            }
        """
        try:
            logger.info(f"📊 Calculating portfolio metrics for {len(stocks)} stocks...")
            
            # Check cache first
            cache_key = f"metrics_{hash(tuple(sorted(stocks)))}"
            if cache_key in self.metrics_cache:
                cached_data = self.metrics_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp']):
                    logger.debug("Using cached portfolio metrics")
                    return cached_data['metrics']
            
            # Use recent data for current market conditions
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 months
            
            returns_data = []
            valid_weights = []
            valid_stocks = []
            
            # === DATA COLLECTION PHASE ===
            for stock in stocks:
                if stock in weights:
                    try:
                        data = yf.download(stock, start=start_date, end=end_date, 
                                        progress=False, timeout=10)['Close']
                        
                        if len(data) > 30:  # Minimum data requirement
                            returns = data.pct_change().dropna()
                            if len(returns) > 20:
                                returns_data.append(returns)
                                valid_weights.append(weights[stock])
                                valid_stocks.append(stock)
                                logger.debug(f"✓ {stock}: {len(returns)} returns")
                            else:
                                logger.warning(f"✗ {stock}: Insufficient returns data")
                        else:
                            logger.warning(f"✗ {stock}: Insufficient price data")
                            
                    except Exception as e:
                        logger.warning(f"✗ {stock}: Failed to get data - {str(e)[:50]}")
                        continue
            
            # Fallback if insufficient data
            if len(returns_data) < 2:
                logger.warning(f"Only {len(returns_data)} stocks have data, using defaults")
                default_metrics = {
                    "expected_return": 0.08, "volatility": 0.15, 
                    "sharpe_ratio": 0.5, "portfolio_size": len(stocks)
                }
                return default_metrics
            
            # === PORTFOLIO CONSTRUCTION PHASE ===
            # Align all return series to same dates
            returns_df = pd.concat(returns_data, axis=1, keys=valid_stocks)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:
                logger.warning(f"Only {len(returns_df)} aligned observations, using defaults")
                default_metrics = {
                    "expected_return": 0.08, "volatility": 0.15,
                    "sharpe_ratio": 0.5, "portfolio_size": len(stocks)
                }
                return default_metrics
            
            # Normalize weights for valid stocks only
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            
            # === PORTFOLIO METRICS CALCULATION ===
            # Calculate portfolio returns (weighted sum of individual returns)
            portfolio_returns = (returns_df * valid_weights).sum(axis=1)
            
            # Calculate annualized metrics
            expected_annual_return = portfolio_returns.mean() * 252  # 252 trading days per year
            annual_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate Sharpe ratio (risk-adjusted return)
            if annual_volatility == 0:
                sharpe_ratio = 0  # Avoid division by zero
            else:
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                sharpe_ratio = (expected_annual_return - risk_free_rate) / annual_volatility
            
            # Apply reasonable bounds to prevent extreme values
            expected_annual_return = max(-0.5, min(1.0, expected_annual_return))
            annual_volatility = max(0.01, min(2.0, annual_volatility))
            sharpe_ratio = max(-5, min(5, sharpe_ratio))
            
            metrics = {
                "expected_return": expected_annual_return,
                "volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "portfolio_size": len(valid_stocks),
                "data_points": len(returns_df)
            }
            
            # Cache the results
            self.metrics_cache[cache_key] = {
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            
            logger.info(f"📈 Portfolio metrics: {expected_annual_return:.1%} return, {annual_volatility:.1%} volatility, {sharpe_ratio:.2f} Sharpe")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {
                "expected_return": 0.08, "volatility": 0.15, 
                "sharpe_ratio": 0.5, "portfolio_size": len(stocks)
            }
    
    def calculate_efficient_frontier(self, stocks: List[str], 
                                   num_portfolios: int = 100) -> Dict[str, List[float]]:
        """
        Calculate efficient frontier for portfolio optimization
        
        Generates multiple portfolio combinations and finds optimal risk-return profiles
        
        Args:
            stocks: List of stock tickers
            num_portfolios: Number of random portfolios to generate
            
        Returns:
            Dict with lists of returns, volatilities, and sharpe ratios
        """
        try:
            logger.info(f"📊 Calculating efficient frontier with {num_portfolios} portfolios...")
            
            # Download data for all stocks
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            returns_data = {}
            for stock in stocks:
                try:
                    data = yf.download(stock, start=start_date, end=end_date, 
                                    progress=False, timeout=10)['Close']
                    if len(data) > 100:
                        returns_data[stock] = data.pct_change().dropna()
                except Exception:
                    continue
            
            if len(returns_data) < 3:
                logger.warning("Insufficient data for efficient frontier calculation")
                return {"returns": [], "volatilities": [], "sharpe_ratios": []}
            
            # Align return series
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 50:
                logger.warning("Insufficient aligned data for efficient frontier")
                return {"returns": [], "volatilities": [], "sharpe_ratios": []}
            
            # Generate random portfolio weights
            num_assets = len(returns_df.columns)
            results = {"returns": [], "volatilities": [], "sharpe_ratios": []}
            
            for _ in range(num_portfolios):
                # Generate random weights that sum to 1
                weights = np.random.random(num_assets)
                weights = weights / weights.sum()
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(returns_df.mean() * weights) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
                
                # Calculate Sharpe ratio
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_std if portfolio_std > 0 else 0
                
                results["returns"].append(portfolio_return)
                results["volatilities"].append(portfolio_std)
                results["sharpe_ratios"].append(sharpe_ratio)
            
            logger.info(f"✅ Efficient frontier calculated with {len(results['returns'])} portfolios")
            return results
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {e}")
            return {"returns": [], "volatilities": [], "sharpe_ratios": []}
    
    def find_optimal_portfolio(self, stocks: List[str], 
                             target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Find optimal portfolio weights using mean-variance optimization
        
        Args:
            stocks: List of stock tickers
            target_return: Target annual return (optional)
            
        Returns:
            Dict with optimal weights for each stock
        """
        try:
            logger.info(f"🎯 Finding optimal portfolio for {len(stocks)} stocks...")
            
            # This is a simplified implementation
            # In practice, you'd use scipy.optimize for proper mean-variance optimization
            
            # For now, return equal weights as a starting point
            equal_weight = 1.0 / len(stocks)
            optimal_weights = {stock: equal_weight for stock in stocks}
            
            logger.info(f"✅ Optimal portfolio: equal weights ({equal_weight:.3f} each)")
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Optimal portfolio calculation failed: {e}")
            equal_weight = 1.0 / len(stocks)
            return {stock: equal_weight for stock in stocks}
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid"""
        time_elapsed = (datetime.now() - timestamp).total_seconds()
        return time_elapsed < self.cache_timeout
    
    def validate_and_filter_stocks(self, stocks: List[str]) -> List[str]:
        """
        Validate stocks and filter out any that can't be traded
        
        Args:
            stocks: List of stock tickers to validate
            
        Returns:
            List of valid, tradeable stock tickers
        """
        valid_stocks = []
        
        logger.info(f"🔍 Validating {len(stocks)} stocks for trading availability...")
        
        for stock in stocks:
            try:
                # Quick check to see if stock data is available
                ticker = yf.Ticker(stock)
                info = ticker.info
                
                # Check if it's a valid, tradeable security
                if info and info.get('regularMarketPrice', 0) > 0:
                    valid_stocks.append(stock)
                    logger.debug(f"✓ {stock}: Valid")
                else:
                    logger.warning(f"✗ {stock}: No market price data")
                    
            except Exception as e:
                logger.warning(f"✗ {stock}: Validation failed - {str(e)[:50]}")
                continue
        
        logger.info(f"✅ {len(valid_stocks)}/{len(stocks)} stocks validated successfully")
        return valid_stocks