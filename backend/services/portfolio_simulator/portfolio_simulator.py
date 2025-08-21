"""
Core portfolio simulation logic for the Portfolio Simulator Service.

This module handles the mathematical simulation of portfolio growth over time,
including weight calculation, rebalancing, and performance metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .config import get_config, get_risk_profiles
from .exceptions import SimulationError, OptimizationError
from .validators import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    starting_value: float
    ending_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_contributed: float
    profit_loss: float


@dataclass
class SimulationResult:
    """Container for complete simulation results."""
    portfolio_metrics: PortfolioMetrics
    timeline_data: List[Dict[str, Any]]
    contribution_data: List[Dict[str, Any]]
    asset_breakdown: Dict[str, float]
    rebalancing_events: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PortfolioWeightCalculator:
    """
    Calculates optimal portfolio weights based on risk profile and constraints.
    
    This class implements various portfolio optimization strategies including
    equal weighting, risk-based allocation, and modern portfolio theory approaches.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the weight calculator.
        
        Args:
            validator: Input validator instance
        """
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
        Calculate portfolio weights based on risk profile and historical data.
        
        Args:
            tickers: List of ticker symbols
            risk_score: Risk tolerance score (0-100)
            risk_label: Risk profile label
            data: Historical price data for optimization (optional)
            
        Returns:
            Array of portfolio weights summing to 1.0
            
        Raises:
            OptimizationError: If weight calculation fails
        """
        try:
            # Validate inputs
            validated_tickers = self.validator.validate_ticker_symbols(tickers)
            validated_risk_score = self.validator.validate_risk_score(risk_score)
            
            logger.info(
                f"Calculating weights for {len(validated_tickers)} assets "
                f"with risk score {validated_risk_score}"
            )
            
            # Choose optimization method based on available data and risk profile
            if data is not None and len(data) > 60:  # Need sufficient data for optimization
                weights = self._optimize_with_historical_data(
                    validated_tickers, validated_risk_score, data
                )
            else:
                weights = self._get_risk_based_weights(
                    validated_tickers, validated_risk_score, risk_label
                )
            
            # Validate and normalize weights
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
        Calculate weights based on risk profile without historical data.
        
        Args:
            tickers: List of ticker symbols
            risk_score: Risk tolerance score
            risk_label: Risk profile label
            
        Returns:
            List of portfolio weights
        """
        num_assets = len(tickers)
        
        # Get risk profile information
        profile_key = risk_label.lower()
        if profile_key not in self.risk_profiles:
            profile_key = self._map_risk_score_to_profile(risk_score)
        
        profile = self.risk_profiles[profile_key]
        
        if risk_score < 35:  # Conservative
            # Equal weighting with slight bias toward first assets (typically safer)
            weights = self._create_conservative_weights(num_assets)
            
        elif risk_score < 70:  # Moderate
            # Moderate concentration with diversification
            weights = self._create_moderate_weights(num_assets)
            
        else:  # Aggressive
            # Higher concentration in growth assets
            weights = self._create_aggressive_weights(num_assets)
        
        return weights
    
    def _create_conservative_weights(self, num_assets: int) -> List[float]:
        """Create conservative portfolio weights with equal-ish distribution."""
        if num_assets == 1:
            return [1.0]
        
        # Slightly favor first assets (bonds/stable investments)
        weights = []
        base_weight = 1.0 / num_assets
        
        for i in range(num_assets):
            # First few assets get slightly higher weights
            if i < num_assets // 2:
                weight = base_weight * 1.1
            else:
                weight = base_weight * 0.9
            weights.append(weight)
        
        # Normalize to sum to 1
        total = sum(weights)
        return [w / total for w in weights]
    
    def _create_moderate_weights(self, num_assets: int) -> List[float]:
        """Create moderate portfolio weights with some concentration."""
        if num_assets == 1:
            return [1.0]
        
        # Moderate concentration: larger weights for first few assets
        weights = []
        remaining_weight = 1.0
        
        # Define concentration ratios
        primary_ratio = 0.3
        secondary_ratio = 0.25
        
        for i in range(min(num_assets, 2)):
            if i == 0:
                weight = primary_ratio
            else:
                weight = secondary_ratio
            
            weights.append(weight)
            remaining_weight -= weight
        
        # Distribute remaining weight equally among other assets
        if num_assets > 2:
            remaining_per_asset = remaining_weight / (num_assets - 2)
            weights.extend([remaining_per_asset] * (num_assets - 2))
        
        return weights
    
    def _create_aggressive_weights(self, num_assets: int) -> List[float]:
        """Create aggressive portfolio weights with higher concentration."""
        if num_assets == 1:
            return [1.0]
        
        # High concentration in first few assets
        weights = []
        remaining_weight = 1.0
        
        # Define concentration ratios for aggressive portfolio
        ratios = [0.4, 0.3, 0.2, 0.1]
        
        for i in range(min(num_assets, len(ratios))):
            weight = ratios[i]
            weights.append(weight)
            remaining_weight -= weight
        
        # If more assets than predefined ratios, distribute remaining equally
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
        Optimize portfolio weights using historical data.
        
        Args:
            tickers: List of ticker symbols
            risk_score: Risk tolerance score
            data: Historical price data
            
        Returns:
            Optimized portfolio weights
        """
        try:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            if len(returns) < 30:  # Need sufficient data for optimization
                logger.warning("Insufficient data for optimization, using risk-based weights")
                return self._get_risk_based_weights(tickers, risk_score, "moderate")
            
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean() * 252  # Annualized
            cov_matrix = returns.cov() * 252     # Annualized
            
            # Risk tolerance factor (0 = very conservative, 1 = very aggressive)
            risk_tolerance = risk_score / 100.0
            
            # Simple mean-variance optimization with risk penalty
            num_assets = len(tickers)
            
            # Equal weights as starting point
            weights = np.array([1.0 / num_assets] * num_assets)
            
            # Adjust weights based on risk-return characteristics
            for i in range(num_assets):
                # Favor higher expected returns for higher risk tolerance
                return_boost = mean_returns.iloc[i] * risk_tolerance
                
                # Penalize higher volatility for lower risk tolerance
                volatility = np.sqrt(cov_matrix.iloc[i, i])
                volatility_penalty = volatility * (1 - risk_tolerance)
                
                # Adjust weight (simple heuristic)
                adjustment = (return_boost - volatility_penalty) * 0.1
                weights[i] += adjustment
            
            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 0.01)  # Minimum 1% allocation
            weights = weights / np.sum(weights)
            
            return weights.tolist()
            
        except Exception as e:
            logger.warning(f"Optimization failed: {str(e)}, using fallback weights")
            return self._get_risk_based_weights(tickers, risk_score, "moderate")
    
    def _map_risk_score_to_profile(self, risk_score: int) -> str:
        """Map numerical risk score to risk profile name."""
        if risk_score < 35:
            return "conservative"
        elif risk_score < 70:
            return "moderate"
        else:
            return "aggressive"


class PortfolioSimulator:
    """
    Simulates portfolio growth over time with regular contributions and rebalancing.
    
    This class implements the core simulation logic including compound growth,
    regular contributions, and optional rebalancing strategies.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
        """
        Initialize the portfolio simulator.
        
        Args:
            validator: Input validator instance
        """
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
        Simulate portfolio growth over the specified timeframe.
        
        Args:
            data: Historical price data
            weights: Portfolio weights for each asset
            lump_sum: Initial lump sum investment
            monthly_contribution: Monthly contribution amount
            timeframe_years: Investment timeframe in years
            rebalance_frequency: How often to rebalance ("never", "monthly", "quarterly", "annually")
            
        Returns:
            Complete simulation results
            
        Raises:
            SimulationError: If simulation fails
        """
        try:
            logger.info(
                f"Starting portfolio simulation: £{lump_sum:,.2f} initial + "
                f"£{monthly_contribution:,.2f}/month for {timeframe_years} years"
            )
            
            # Validate inputs
            validated_weights = self.validator.validate_weights(weights.tolist())
            weights = np.array(validated_weights)
            
            # Prepare data for simulation
            simulation_data = self._prepare_simulation_data(data, timeframe_years)
            
            if simulation_data.empty:
                raise SimulationError(
                    "No data available for simulation period",
                    simulation_type="growth_simulation",
                    timeframe=timeframe_years
                )
            
            # Run the simulation
            timeline_data, contribution_data, rebalancing_events = self._run_simulation(
                simulation_data, weights, lump_sum, monthly_contribution, rebalance_frequency
            )
            
            # Calculate performance metrics
            portfolio_metrics = self._calculate_metrics(
                timeline_data, contribution_data, lump_sum, monthly_contribution, timeframe_years
            )
            
            # Create asset breakdown
            asset_breakdown = {
                col: float(weight) for col, weight in zip(data.columns, weights)
            }
            
            # Prepare metadata
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
        Prepare historical data for simulation by normalizing and validating.
        
        Args:
            data: Historical price data
            timeframe_years: Simulation timeframe
            
        Returns:
            Prepared simulation data
        """
        # Use the most recent data up to the required timeframe
        required_days = timeframe_years * self.config.simulation.trading_days_per_year
        
        if len(data) > required_days:
            # Use the most recent data
            simulation_data = data.tail(required_days).copy()
        else:
            # Use all available data
            simulation_data = data.copy()
        
        # Normalize prices to start at 1.0 for growth calculation
        first_valid_prices = simulation_data.iloc[0]
        
        # Handle any zero or NaN starting prices
        first_valid_prices = first_valid_prices.replace(0, np.nan)
        first_valid_prices = first_valid_prices.fillna(method='bfill')
        
        if first_valid_prices.isna().any():
            raise SimulationError(
                "Unable to find valid starting prices for all assets",
                simulation_type="data_preparation"
            )
        
        # Normalize data
        normalized_data = simulation_data.div(first_valid_prices)
        
        # Forward fill any remaining NaN values
        normalized_data = normalized_data.ffill()
        
        return normalized_data
    
    def _run_simulation(
        self,
        data: pd.DataFrame,
        weights: np.ndarray,
        lump_sum: float,
        monthly_contribution: float,
        rebalance_frequency: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run the core simulation logic.
        
        Args:
            data: Normalized price data
            weights: Portfolio weights
            lump_sum: Initial investment
            monthly_contribution: Monthly contribution
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Tuple of (timeline_data, contribution_data, rebalancing_events)
        """
        timeline_data = []
        contribution_data = []
        rebalancing_events = []
        
        # Initialize portfolio
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        # Calculate portfolio values for each day
        portfolio_growth = data.dot(weights)
        
        # Determine contribution schedule
        trading_days_per_month = self.config.simulation.trading_days_per_month
        
        for i, (date, growth_factor) in enumerate(portfolio_growth.items()):
            # Add monthly contribution (approximately every 21 trading days)
            if i > 0 and i % trading_days_per_month == 0 and monthly_contribution > 0:
                current_value += monthly_contribution
                total_contributions += monthly_contribution
                
                # Record contribution event
                contribution_data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "contribution": float(monthly_contribution),
                    "total_contributed": float(total_contributions)
                })
            
            # Apply market growth
            current_value = current_value * growth_factor
            
            # Record daily portfolio value
            timeline_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": float(current_value),
                "growth_factor": float(growth_factor)
            })
            
            # TODO: Implement rebalancing logic if needed
            # This would require tracking individual asset values and rebalancing
            # according to the specified frequency
        
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
        
        Args:
            timeline_data: Daily portfolio values
            contribution_data: Contribution events
            lump_sum: Initial investment
            monthly_contribution: Monthly contribution
            timeframe_years: Investment timeframe
            
        Returns:
            Portfolio performance metrics
        """
        if not timeline_data:
            raise SimulationError("No timeline data available for metrics calculation")
        
        # Basic values
        starting_value = float(lump_sum)
        ending_value = float(timeline_data[-1]['value'])
        total_contributed = float(lump_sum + (monthly_contribution * 12 * timeframe_years))
        profit_loss = ending_value - total_contributed
        
        # Total return calculation
        if total_contributed > 0:
            total_return = ((ending_value - total_contributed) / total_contributed) * 100
        else:
            total_return = 0.0
        
        # Annualized return calculation
        if timeframe_years > 0:
            annualized_return = ((ending_value / total_contributed) ** (1 / timeframe_years) - 1) * 100
        else:
            annualized_return = 0.0
        
        # Calculate volatility from daily returns
        values = [item['value'] for item in timeline_data]
        if len(values) > 1:
            daily_returns = [
                (values[i] / values[i-1] - 1) for i in range(1, len(values))
            ]
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0.0
        
        # Sharpe ratio (assuming risk-free rate from config)
        risk_free_rate = self.config.simulation.default_risk_free_rate
        if volatility > 0:
            sharpe_ratio = (annualized_return / 100 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown calculation
        max_drawdown = self._calculate_max_drawdown(values)
        
        return PortfolioMetrics(
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
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """
        Calculate maximum drawdown from peak to trough.
        
        Args:
            values: List of portfolio values
            
        Returns:
            Maximum drawdown as a decimal (e.g., 0.15 for 15%)
        """
        if len(values) < 2:
            return 0.0
        
        max_drawdown = 0.0
        peak = values[0]
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown