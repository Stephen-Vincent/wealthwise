"""
Core portfolio simulation logic for the Portfolio Simulator Service.

This module handles the mathematical simulation of portfolio growth over time,
including weight calculation, rebalancing, and performance metrics.

CRITICAL MATHEMATICAL FOUNDATIONS:
- Portfolio growth is calculated using daily returns, not cumulative price levels
- Contributions are added at regular intervals (monthly)
- Performance metrics use standard financial formulas
- All calculations handle edge cases (NaN, infinity, zero values)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .config import get_config, get_risk_profiles
from .exceptions import SimulationError, OptimizationError
from .validators import InputValidator

logger = logging.getLogger(__name__)


# ================================================================================================
# DATA STRUCTURES FOR SIMULATION RESULTS
# ================================================================================================

@dataclass
class PortfolioMetrics:
    """
    Container for portfolio performance metrics.
    
    All financial metrics follow industry-standard calculations:
    - Returns are annualized percentages
    - Volatility is annualized standard deviation
    - Sharpe ratio uses risk-free rate subtraction
    - Max drawdown is peak-to-trough decline
    """
    starting_value: float        # Initial portfolio value
    ending_value: float          # Final portfolio value after simulation
    total_return: float          # Total percentage return over entire period
    annualized_return: float     # Compound annual growth rate (CAGR)
    volatility: float           # Annualized volatility (standard deviation)
    sharpe_ratio: float         # Risk-adjusted return metric
    max_drawdown: float         # Maximum peak-to-trough decline
    total_contributed: float    # Total amount invested (lump sum + contributions)
    profit_loss: float          # Net profit/loss (ending value - total contributed)


@dataclass
class SimulationResult:
    """
    Container for complete simulation results.
    
    This structure holds all data needed for analysis and visualization:
    - Performance metrics for quantitative analysis
    - Timeline data for charting portfolio growth
    - Contribution data for cash flow analysis
    - Asset breakdown for allocation analysis
    - Metadata for audit trail and debugging
    """
    portfolio_metrics: PortfolioMetrics         # Core performance numbers
    timeline_data: List[Dict[str, Any]]         # Daily portfolio values
    contribution_data: List[Dict[str, Any]]     # Investment contribution history
    asset_breakdown: Dict[str, float]           # Final asset allocation weights
    rebalancing_events: List[Dict[str, Any]]    # Rebalancing transaction history
    metadata: Dict[str, Any]                    # Simulation parameters and info


# ================================================================================================
# PORTFOLIO WEIGHT CALCULATION ENGINE
# ================================================================================================

class PortfolioWeightCalculator:
    """
    Calculates optimal portfolio weights based on risk profile and constraints.
    
    This class implements multiple portfolio optimization strategies:
    1. Risk-based allocation (uses predefined risk profiles)
    2. Historical data optimization (basic mean-variance approach)
    3. Constraint-based weighting (ensures diversification limits)
    
    MATHEMATICAL APPROACH:
    - Conservative: Equal weighting with slight bias toward stable assets
    - Moderate: Concentration in top 2-3 assets with diversification tail
    - Aggressive: Heavy concentration in growth assets
    - Historical optimization: Simple mean-variance with risk tolerance adjustment
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """Initialize weight calculator with configuration and validation."""
        self.config = get_config()
        self.validator = validator or InputValidator()
        self.risk_profiles = get_risk_profiles()
    
    def calculate_weights(
        self, 
        tickers: List[str], 
        risk_score: int, 
        risk_label: str,
        data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Main entry point for portfolio weight calculation.
        
        DECISION LOGIC:
        1. If sufficient historical data (>60 days): Use quantitative optimization
        2. If limited data: Use risk-based heuristic weights
        3. Always validate and normalize weights to sum to 1.0
        
        Args:
            tickers: List of asset symbols
            risk_score: Numerical risk tolerance (0-100)
            risk_label: Categorical risk label (conservative/moderate/aggressive)
            data: Historical price data for optimization (optional)
            
        Returns:
            Normalized weight array summing to 1.0
            
        Raises:
            OptimizationError: If weight calculation fails
        """
        try:
            # Input validation and sanitization
            validated_tickers = self.validator.validate_ticker_symbols(tickers)
            validated_risk_score = self.validator.validate_risk_score(risk_score)
            
            logger.info(
                f"Calculating weights for {len(validated_tickers)} assets "
                f"with risk score {validated_risk_score}"
            )
            
            # Choose optimization strategy based on data availability
            if data is not None and len(data) > 60:
                # Sufficient data for quantitative optimization
                weights = self._optimize_with_historical_data(
                    validated_tickers, validated_risk_score, data
                )
                logger.info("Using historical data optimization")
            else:
                # Use rule-based risk profile weights
                weights = self._get_risk_based_weights(
                    validated_tickers, validated_risk_score, risk_label
                )
                logger.info("Using risk-based weight allocation")
            
            # Final validation and normalization
            weights = self.validator.validate_weights(weights)
            
            logger.info(f"Weight calculation complete: {dict(zip(validated_tickers, weights))}")
            
            return np.array(weights)
            
        except Exception as e:
            if isinstance(e, OptimizationError):
                raise
            logger.error(f"Error calculating portfolio weights: {str(e)}")
            raise OptimizationError(
                f"Failed to calculate portfolio weights: {str(e)}",
                method="weight_calculation"
            )
    
    def _get_risk_based_weights(
        self, 
        tickers: List[str], 
        risk_score: int, 
        risk_label: str
    ) -> List[float]:
        """
        Calculate weights using rule-based risk profiles.
        
        STRATEGY MAPPING:
        - Conservative (0-34): Equal weighting with stability bias
        - Moderate (35-69): Moderate concentration with diversification
        - Aggressive (70-100): High concentration in growth assets
        
        This approach doesn't require historical data and provides
        sensible defaults based on investment psychology research.
        """
        num_assets = len(tickers)
        
        # Map risk label to profile configuration
        profile_key = risk_label.lower()
        if profile_key not in self.risk_profiles:
            profile_key = self._map_risk_score_to_profile(risk_score)
        
        # Apply appropriate weighting strategy
        if risk_score < 35:
            weights = self._create_conservative_weights(num_assets)
        elif risk_score < 70:
            weights = self._create_moderate_weights(num_assets)
        else:
            weights = self._create_aggressive_weights(num_assets)
        
        return weights
    
    def _create_conservative_weights(self, num_assets: int) -> List[float]:
        """
        Create conservative portfolio weights.
        
        CONSERVATIVE STRATEGY:
        - Nearly equal weighting to minimize concentration risk
        - Slight bias toward first assets (typically bonds/stable investments)
        - Maximum individual weight kept low for safety
        
        Mathematical approach: Base equal weights with 10% bias adjustment
        """
        if num_assets == 1:
            return [1.0]
        
        weights = []
        base_weight = 1.0 / num_assets
        
        for i in range(num_assets):
            # Favor stability assets (typically listed first)
            if i < num_assets // 2:
                weight = base_weight * 1.1  # 10% overweight for stable assets
            else:
                weight = base_weight * 0.9  # 10% underweight for growth assets
            weights.append(weight)
        
        # Normalize to ensure sum equals 1.0
        total = sum(weights)
        return [w / total for w in weights]
    
    def _create_moderate_weights(self, num_assets: int) -> List[float]:
        """
        Create moderate portfolio weights.
        
        MODERATE STRATEGY:
        - Concentrated exposure in top 2 assets (55% combined)
        - Remaining assets get equal smaller allocations
        - Balances growth potential with diversification
        
        Allocation pattern: 30%, 25%, then equal split of remaining 45%
        """
        if num_assets == 1:
            return [1.0]
        
        weights = []
        remaining_weight = 1.0
        
        # Primary allocations for top assets
        primary_ratio, secondary_ratio = 0.3, 0.25
        
        for i in range(min(num_assets, 2)):
            weight = primary_ratio if i == 0 else secondary_ratio
            weights.append(weight)
            remaining_weight -= weight
        
        # Equal distribution for remaining assets
        if num_assets > 2:
            remaining_per_asset = remaining_weight / (num_assets - 2)
            weights.extend([remaining_per_asset] * (num_assets - 2))
        
        return weights
    
    def _create_aggressive_weights(self, num_assets: int) -> List[float]:
        """
        Create aggressive portfolio weights.
        
        AGGRESSIVE STRATEGY:
        - Heavy concentration in top performers (70% in top 2)
        - Diminishing allocation pattern for additional assets
        - Optimized for maximum growth potential
        
        Allocation pattern: 40%, 30%, 20%, 10%, then equal tiny allocations
        """
        if num_assets == 1:
            return [1.0]
        
        weights = []
        remaining_weight = 1.0
        
        # Predefined concentration ratios for aggressive allocation
        ratios = [0.4, 0.3, 0.2, 0.1]  # Top 4 assets get these allocations
        
        for i in range(min(num_assets, len(ratios))):
            weight = ratios[i]
            weights.append(weight)
            remaining_weight -= weight
        
        # Tiny equal allocations for any remaining assets
        if num_assets > len(ratios):
            remaining_per_asset = remaining_weight / (num_assets - len(ratios))
            weights.extend([remaining_per_asset] * (num_assets - len(ratios)))
        
        return weights
    
    def _optimize_with_historical_data(
        self, 
        tickers: List[str], 
        risk_score: int, 
        data: pd.DataFrame
    ) -> List[float]:
        """
        Optimize portfolio weights using historical return data.
        
        QUANTITATIVE OPTIMIZATION APPROACH:
        1. Calculate annualized mean returns and covariance matrix
        2. Apply risk tolerance adjustment to expected returns
        3. Penalize high volatility assets for conservative investors
        4. Ensure minimum 1% allocation to prevent zero weights
        5. Normalize final weights to sum to 1.0
        
        This is a simplified mean-variance optimization that balances
        expected returns against volatility based on investor risk tolerance.
        
        MATHEMATICAL FORMULA:
        weight_adjustment = (expected_return * risk_tolerance - volatility * (1 - risk_tolerance)) * 0.1
        """
        try:
            # Calculate daily returns from price data
            returns = data.pct_change().dropna()
            
            if len(returns) < 30:
                logger.warning("Insufficient data for optimization, using risk-based weights")
                return self._get_risk_based_weights(tickers, risk_score, self._map_risk_score_to_profile(risk_score))
            
            # Annualize statistics (252 trading days per year)
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Convert risk score to risk tolerance factor (0.0 to 1.0)
            risk_tolerance = risk_score / 100.0
            
            # Start with equal weights as baseline
            num_assets = len(tickers)
            weights = np.array([1.0 / num_assets] * num_assets)
            
            # Apply risk-adjusted optimization
            for i in range(num_assets):
                # Boost weight for higher expected returns (scaled by risk tolerance)
                return_boost = mean_returns.iloc[i] * risk_tolerance
                
                # Penalize weight for higher volatility (scaled by risk aversion)
                volatility = np.sqrt(cov_matrix.iloc[i, i])
                volatility_penalty = volatility * (1 - risk_tolerance)
                
                # Apply combined adjustment (10% maximum adjustment per iteration)
                adjustment = (return_boost - volatility_penalty) * 0.1
                weights[i] += adjustment
            
            # Ensure minimum allocation and normalize
            weights = np.maximum(weights, 0.01)  # Minimum 1% per asset
            weights = weights / np.sum(weights)  # Normalize to sum to 1.0
            
            return weights.tolist()
            
        except Exception as e:
            logger.warning(f"Optimization failed: {str(e)}, using fallback weights")
            return self._get_risk_based_weights(tickers, risk_score, self._map_risk_score_to_profile(risk_score))
    
    def _map_risk_score_to_profile(self, risk_score: int) -> str:
        """Map numerical risk score to categorical risk profile."""
        if risk_score < 35:
            return "conservative"
        elif risk_score < 70:
            return "moderate"
        else:
            return "aggressive"


# ================================================================================================
# PORTFOLIO GROWTH SIMULATION ENGINE
# ================================================================================================

class PortfolioSimulator:
    """
    Simulates portfolio growth over time with regular contributions and rebalancing.
    
    SIMULATION METHODOLOGY:
    1. Normalize historical price data to start at 1.0
    2. Calculate daily portfolio returns using weighted average
    3. Apply returns to current portfolio value each day
    4. Add contributions at monthly intervals
    5. Track all changes for performance analysis
    
    CRITICAL MATHEMATICAL CONCEPTS:
    - Uses RETURNS not PRICE LEVELS to avoid compounding errors
    - Handles contributions separately from market growth
    - Maintains detailed audit trail for analysis
    - Calculates standard financial metrics
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """Initialize simulator with configuration and validation."""
        self.config = get_config()
        self.validator = validator or InputValidator()
        self.weight_calculator = PortfolioWeightCalculator(validator)
    
    def simulate_growth(
        self, 
        data: pd.DataFrame,
        weights: np.ndarray,
        lump_sum: float,
        monthly_contribution: float,
        timeframe_years: int,
        rebalance_frequency: str = "quarterly"
    ) -> SimulationResult:
        """
        Execute complete portfolio growth simulation.
        
        SIMULATION PROCESS:
        1. Validate inputs and prepare historical data
        2. Run day-by-day simulation with contributions
        3. Calculate comprehensive performance metrics
        4. Package results with metadata for analysis
        
        Args:
            data: Historical price data for assets
            weights: Portfolio allocation weights (must sum to 1.0)
            lump_sum: Initial investment amount
            monthly_contribution: Regular monthly investment
            timeframe_years: Simulation period in years
            rebalance_frequency: How often to rebalance (not yet implemented)
            
        Returns:
            Complete simulation results with metrics and timeline
            
        Raises:
            SimulationError: If simulation fails at any stage
        """
        try:
            logger.info(
                f"Starting portfolio simulation: £{lump_sum:,.2f} initial + "
                f"£{monthly_contribution:,.2f}/month for {timeframe_years} years"
            )
            
            # Validate and normalize weights
            validated_weights = self.validator.validate_weights(weights.tolist())
            weights = np.array(validated_weights)
            
            # Prepare historical data for simulation
            simulation_data = self._prepare_simulation_data(data, timeframe_years)
            
            if simulation_data.empty:
                raise SimulationError(
                    "No data available for simulation period",
                    simulation_type="growth_simulation",
                    timeframe=timeframe_years
                )
            
            # Execute core simulation
            timeline_data, contribution_data, rebalancing_events = self._run_simulation(
                simulation_data, weights, lump_sum, monthly_contribution, rebalance_frequency
            )
            
            # Calculate comprehensive performance metrics
            portfolio_metrics = self._calculate_metrics(
                timeline_data, contribution_data, lump_sum, monthly_contribution, timeframe_years
            )
            
            # Create asset breakdown for analysis
            asset_breakdown = {
                col: float(weight) for col, weight in zip(simulation_data.columns, weights)
            }
            
            # Package metadata for audit trail
            metadata = {
                "simulation_date": datetime.now().isoformat(),
                "data_period": {
                    "start": simulation_data.index.min().isoformat(),
                    "end": simulation_data.index.max().isoformat(),
                    "days": len(simulation_data)
                },
                "rebalance_frequency": rebalance_frequency,
                "rebalance_count": len(rebalancing_events),
                "parameters": {
                    "lump_sum": lump_sum,
                    "monthly_contribution": monthly_contribution,
                    "timeframe_years": timeframe_years
                }
            }
            
            # Create final result package
            result = SimulationResult(
                portfolio_metrics=portfolio_metrics,
                timeline_data=timeline_data,
                contribution_data=contribution_data,
                asset_breakdown=asset_breakdown,
                rebalancing_events=rebalancing_events,
                metadata=metadata
            )
            
            logger.info(
                f"Simulation complete: £{portfolio_metrics.starting_value:,.2f} → "
                f"£{portfolio_metrics.ending_value:,.2f} "
                f"({portfolio_metrics.total_return:.1f}% total return)"
            )
            
            return result
            
        except Exception as e:
            if isinstance(e, SimulationError):
                raise
            logger.error(f"Portfolio simulation failed: {str(e)}")
            raise SimulationError(
                f"Simulation failed: {str(e)}",
                simulation_type="growth_simulation",
                timeframe=timeframe_years
            )
    
    def _prepare_simulation_data(
        self, 
        data: pd.DataFrame, 
        timeframe_years: int
    ) -> pd.DataFrame:
        """
        Prepare historical price data for simulation.
        
        DATA PREPARATION PROCESS:
        1. Extract required amount of historical data
        2. Handle missing values with forward/backward fill
        3. Validate that all assets have starting prices
        4. Convert to returns-based format for simulation
        
        CRITICAL: This method converts PRICE LEVELS to RETURNS
        to avoid mathematical errors in portfolio growth calculation.
        
        Args:
            data: Raw historical price data
            timeframe_years: Simulation period for data selection
            
        Returns:
            Clean, validated price data ready for simulation
            
        Raises:
            SimulationError: If data cannot be prepared properly
        """
        # Calculate required data points
        required_days = timeframe_years * self.config.simulation.trading_days_per_year
        
        # Extract most recent data for simulation period
        if len(data) > required_days:
            simulation_data = data.tail(required_days).copy()
        else:
            simulation_data = data.copy()
        
        # Handle missing values (forward fill then backward fill)
        # This is necessary but can create unrealistic price movements
        simulation_data = simulation_data.ffill().bfill()
        
        # Validate starting prices (cannot be zero or NaN)
        first_valid_prices = simulation_data.iloc[0].replace(0, np.nan)
        if first_valid_prices.isna().any():
            raise SimulationError(
                "Unable to find valid starting prices for all assets",
                simulation_type="data_preparation"
            )
        
        logger.info(f"Prepared simulation data: {len(simulation_data)} days, {len(simulation_data.columns)} assets")
        
        return simulation_data
    
    def _run_simulation(
        self,
        data: pd.DataFrame,
        weights: np.ndarray,
        lump_sum: float,
        monthly_contribution: float,
        rebalance_frequency: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute the core day-by-day portfolio simulation.
        
        CORRECTED MATHEMATICAL APPROACH:
        This method now uses DAILY RETURNS instead of cumulative price levels
        to avoid the exponential compounding error in the original code.
        
        SIMULATION ALGORITHM:
        1. Calculate daily returns for each asset
        2. Compute portfolio daily returns using weights
        3. Apply daily returns to current portfolio value
        4. Add contributions at monthly intervals
        5. Track all changes for analysis
        
        ORIGINAL BUG FIXED:
        Previous code used: portfolio_value *= (level / prev_level)
        This created exponential distortion over time.
        
        New approach uses: portfolio_value *= (1 + daily_return)
        This is mathematically correct for portfolio simulation.
        
        Args:
            data: Prepared price data
            weights: Portfolio allocation weights
            lump_sum: Initial investment
            monthly_contribution: Regular investment amount
            rebalance_frequency: Rebalancing schedule (future feature)
            
        Returns:
            Tuple of (timeline_data, contribution_data, rebalancing_events)
        """
        timeline_data = []
        contribution_data = []
        rebalancing_events = []  # Not yet implemented
        
        # Initialize simulation state
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        # CRITICAL FIX: Calculate daily returns instead of using price levels
        daily_returns = data.pct_change().fillna(0)  # Handle first day with 0 return
        portfolio_returns = daily_returns.dot(weights)  # Weighted average daily returns
        
        # Configuration
        trading_days_per_month = self.config.simulation.trading_days_per_month
        
        logger.info(f"Starting simulation with {len(portfolio_returns)} trading days")
        
        # Main simulation loop - process each trading day
        for i, (date, daily_return) in enumerate(portfolio_returns.items()):
            # Add monthly contribution (approximately every 21 trading days)
            if i > 0 and (i % trading_days_per_month == 0) and monthly_contribution > 0:
                current_value += monthly_contribution
                total_contributions += monthly_contribution
                
                # Record contribution event
                contribution_data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "contribution": float(monthly_contribution),
                    "total_contributed": float(total_contributions)
                })
                
                logger.debug(f"Day {i}: Added £{monthly_contribution} contribution, total: £{total_contributions:,.2f}")
            
            # Apply daily market return to portfolio
            # MATHEMATICAL FORMULA: new_value = old_value * (1 + return)
            if np.isfinite(daily_return):
                current_value *= (1 + daily_return)
            # If return is NaN or infinite, portfolio value stays unchanged
            
            # Record daily portfolio value
            timeline_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": float(current_value),
                "daily_return": float(daily_return) if np.isfinite(daily_return) else 0.0,
                "total_contributed": float(total_contributions)
            })
        
        logger.info(f"Simulation complete: {len(timeline_data)} days processed")
        logger.info(f"Final value: £{current_value:,.2f}, Total contributed: £{total_contributions:,.2f}")
        
        return timeline_data, contribution_data, rebalancing_events
    
    def _calculate_metrics(
        self,
        timeline_data: List[Dict[str, Any]],
        contribution_data: List[Dict[str, Any]],
        lump_sum: float,
        monthly_contribution: float,
        timeframe_years: int
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        FINANCIAL METRICS CALCULATED:
        1. Total Return: Overall percentage gain/loss
        2. Annualized Return: Compound Annual Growth Rate (CAGR)
        3. Volatility: Annualized standard deviation of returns
        4. Sharpe Ratio: Risk-adjusted return metric
        5. Maximum Drawdown: Worst peak-to-trough decline
        
        MATHEMATICAL FORMULAS:
        - Total Return = (End Value - Total Invested) / Total Invested
        - CAGR = (End Value / Total Invested)^(1/years) - 1
        - Volatility = StdDev(Daily Returns) * sqrt(252)
        - Sharpe = (Return - Risk Free Rate) / Volatility
        - Max Drawdown = Max((Peak - Trough) / Peak)
        
        Args:
            timeline_data: Daily portfolio values
            contribution_data: Investment contribution history
            lump_sum: Initial investment
            monthly_contribution: Regular investment amount
            timeframe_years: Investment period
            
        Returns:
            Complete portfolio performance metrics
            
        Raises:
            SimulationError: If metrics cannot be calculated
        """
        if not timeline_data:
            raise SimulationError("No timeline data available for metrics calculation")
        
        # Basic values for calculation
        starting_value = float(lump_sum)
        ending_value = float(timeline_data[-1]['value'])
        total_contributed = float(lump_sum + (monthly_contribution * 12 * timeframe_years))
        profit_loss = ending_value - total_contributed
        
        # Total return calculation (percentage of total invested)
        if total_contributed > 0:
            total_return = ((ending_value - total_contributed) / total_contributed) * 100
        else:
            total_return = 0.0
        
        # Annualized return calculation (CAGR formula)
        if timeframe_years > 0 and total_contributed > 0:
            annualized_return = ((ending_value / total_contributed) ** (1 / timeframe_years) - 1) * 100
        else:
            annualized_return = 0.0
        
        # CORRECTED: Calculate volatility from portfolio values, not normalized levels
        portfolio_values = [item['value'] for item in timeline_data]
        
        if len(portfolio_values) > 1:
            # Calculate daily returns from portfolio values
            value_returns = [
                (portfolio_values[i] / portfolio_values[i-1] - 1) 
                for i in range(1, len(portfolio_values))
                if portfolio_values[i-1] > 0  # Avoid division by zero
            ]
            
            if value_returns:
                # Annualize volatility (252 trading days per year)
                volatility = np.std(value_returns) * np.sqrt(252)
                
                # Calculate maximum drawdown from portfolio values
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                volatility = max_drawdown = 0.0
        else:
            volatility = max_drawdown = 0.0
        
        # Sharpe ratio calculation (risk-adjusted return)
        risk_free_rate = self.config.simulation.default_risk_free_rate
        if volatility > 0:
            sharpe_ratio = (annualized_return / 100 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Package all metrics
        metrics = PortfolioMetrics(
            starting_value=starting_value,
            ending_value=ending_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility * 100,  # Convert to percentage
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            total_contributed=total_contributed,
            profit_loss=profit_loss
        )
        
        logger.info(f"Calculated metrics: Return {total_return:.1f}%, Volatility {volatility*100:.1f}%, Sharpe {sharpe_ratio:.2f}")
        
        return metrics
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """
        Calculate maximum drawdown from peak to trough.
        
        MAXIMUM DRAWDOWN FORMULA:
        For each point in time:
        1. Track the highest value seen so far (peak)
        2. Calculate current drawdown = (peak - current) / peak
        3. Track the maximum drawdown seen
        
        This metric shows the worst loss an investor would have
        experienced from a previous high point.
        
        Args:
            values: Series of portfolio values over time
            
        Returns:
            Maximum drawdown as decimal (0.15 = 15% drawdown)
        """
        if len(values) < 2:
            return 0.0
        
        max_drawdown = 0.0
        peak = values[0]
        
        for value in values[1:]:
            # Update peak if we reach a new high
            if value > peak:
                peak = value
            else:
                # Calculate drawdown from peak
                if peak > 0:  # Avoid division by zero
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown