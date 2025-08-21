"""
Core portfolio simulation logic for the Portfolio Simulator Service.

This module handles the mathematical simulation of portfolio growth over time,
including weight calculation, rebalancing, and performance metrics.
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
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
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
        try:
            validated_tickers = self.validator.validate_ticker_symbols(tickers)
            validated_risk_score = self.validator.validate_risk_score(risk_score)
            
            logger.info(
                f"Calculating weights for {len(validated_tickers)} assets "
                f"with risk score {validated_risk_score}"
            )
            
            if data is not None and len(data) > 60:
                weights = self._optimize_with_historical_data(
                    validated_tickers, validated_risk_score, data
                )
            else:
                weights = self._get_risk_based_weights(
                    validated_tickers, validated_risk_score, risk_label
                )
            
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
        num_assets = len(tickers)
        
        profile_key = risk_label.lower()
        if profile_key not in self.risk_profiles:
            profile_key = self._map_risk_score_to_profile(risk_score)
        
        if risk_score < 35:
            weights = self._create_conservative_weights(num_assets)
        elif risk_score < 70:
            weights = self._create_moderate_weights(num_assets)
        else:
            weights = self._create_aggressive_weights(num_assets)
        
        return weights
    
    def _create_conservative_weights(self, num_assets: int) -> List[float]:
        if num_assets == 1:
            return [1.0]
        weights = []
        base_weight = 1.0 / num_assets
        for i in range(num_assets):
            if i < num_assets // 2:
                weight = base_weight * 1.1
            else:
                weight = base_weight * 0.9
            weights.append(weight)
        total = sum(weights)
        return [w / total for w in weights]
    
    def _create_moderate_weights(self, num_assets: int) -> List[float]:
        if num_assets == 1:
            return [1.0]
        weights = []
        remaining_weight = 1.0
        primary_ratio, secondary_ratio = 0.3, 0.25
        for i in range(min(num_assets, 2)):
            weight = primary_ratio if i == 0 else secondary_ratio
            weights.append(weight)
            remaining_weight -= weight
        if num_assets > 2:
            remaining_per_asset = remaining_weight / (num_assets - 2)
            weights.extend([remaining_per_asset] * (num_assets - 2))
        return weights
    
    def _create_aggressive_weights(self, num_assets: int) -> List[float]:
        if num_assets == 1:
            return [1.0]
        weights = []
        remaining_weight = 1.0
        ratios = [0.4, 0.3, 0.2, 0.1]
        for i in range(min(num_assets, len(ratios))):
            weight = ratios[i]
            weights.append(weight)
            remaining_weight -= weight
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
        try:
            returns = data.pct_change().dropna()
            if len(returns) < 30:
                logger.warning("Insufficient data for optimization, using risk-based weights")
                return self._get_risk_based_weights(tickers, risk_score, self._map_risk_score_to_profile(risk_score))
            
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            risk_tolerance = risk_score / 100.0
            num_assets = len(tickers)
            weights = np.array([1.0 / num_assets] * num_assets)
            
            for i in range(num_assets):
                return_boost = mean_returns.iloc[i] * risk_tolerance
                volatility = np.sqrt(cov_matrix.iloc[i, i])
                volatility_penalty = volatility * (1 - risk_tolerance)
                adjustment = (return_boost - volatility_penalty) * 0.1
                weights[i] += adjustment
            
            weights = np.maximum(weights, 0.01)
            weights = weights / np.sum(weights)
            return weights.tolist()
            
        except Exception as e:
            logger.warning(f"Optimization failed: {str(e)}, using fallback weights")
            return self._get_risk_based_weights(tickers, risk_score, self._map_risk_score_to_profile(risk_score))
    
    def _map_risk_score_to_profile(self, risk_score: int) -> str:
        if risk_score < 35:
            return "conservative"
        elif risk_score < 70:
            return "moderate"
        else:
            return "aggressive"


class PortfolioSimulator:
    """
    Simulates portfolio growth over time with regular contributions and rebalancing.
    """
    
    def __init__(self, validator: Optional[InputValidator] = None):
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
        try:
            logger.info(
                f"Starting portfolio simulation: £{lump_sum:,.2f} initial + "
                f"£{monthly_contribution:,.2f}/month for {timeframe_years} years"
            )
            
            validated_weights = self.validator.validate_weights(weights.tolist())
            weights = np.array(validated_weights)
            simulation_data = self._prepare_simulation_data(data, timeframe_years)
            
            if simulation_data.empty:
                raise SimulationError(
                    "No data available for simulation period",
                    simulation_type="growth_simulation",
                    timeframe=timeframe_years
                )
            
            timeline_data, contribution_data, rebalancing_events = self._run_simulation(
                simulation_data, weights, lump_sum, monthly_contribution, rebalance_frequency
            )
            
            portfolio_metrics = self._calculate_metrics(
                timeline_data, contribution_data, lump_sum, monthly_contribution, timeframe_years
            )
            
            asset_breakdown = { col: float(w) for col, w in zip(simulation_data.columns, weights) }
            
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
        required_days = timeframe_years * self.config.simulation.trading_days_per_year
        simulation_data = data.tail(required_days).copy() if len(data) > required_days else data.copy()
        simulation_data = simulation_data.ffill().bfill()
        
        first_valid = simulation_data.iloc[0].replace(0, np.nan)
        if first_valid.isna().any():
            raise SimulationError(
                "Unable to find valid starting prices for all assets",
                simulation_type="data_preparation"
            )
        
        normalized_data = simulation_data.divide(first_valid).ffill().bfill()
        return normalized_data
    
    def _run_simulation(
        self,
        data: pd.DataFrame,
        weights: np.ndarray,
        lump_sum: float,
        monthly_contribution: float,
        rebalance_frequency: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        timeline_data, contribution_data, rebalancing_events = [], [], []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        portfolio_growth = data.dot(weights)  # normalized cumulative level series
        trading_days_per_month = self.config.simulation.trading_days_per_month
        prev_level = None
        
        for i, (date, level) in enumerate(portfolio_growth.items()):
            if i > 0 and (i % trading_days_per_month == 0) and monthly_contribution > 0:
                current_value += monthly_contribution
                total_contributions += monthly_contribution
                contribution_data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "contribution": float(monthly_contribution),
                    "total_contributed": float(total_contributions),
                })
            
            if prev_level is None:
                daily_factor = 1.0
            else:
                if not np.isfinite(prev_level) or not np.isfinite(level) or prev_level == 0:
                    daily_factor = 1.0
                else:
                    daily_factor = float(level) / float(prev_level)
            
            current_value *= daily_factor
            prev_level = level
            
            timeline_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": float(current_value),
                "growth_factor": float(daily_factor),
                "level": float(level),  # keep raw normalized level
            })
        
        return timeline_data, contribution_data, rebalancing_events
    
    def _calculate_metrics(
        self,
        timeline_data: List[Dict[str, Any]],
        contribution_data: List[Dict[str, Any]],
        lump_sum: float,
        monthly_contribution: float,
        timeframe_years: int
    ) -> PortfolioMetrics:
        if not timeline_data:
            raise SimulationError("No timeline data available for metrics calculation")
        
        starting_value = float(lump_sum)
        ending_value = float(timeline_data[-1]['value'])
        total_contributed = float(lump_sum + (monthly_contribution * 12 * timeframe_years))
        profit_loss = ending_value - total_contributed
        
        total_return = ((ending_value - total_contributed) / total_contributed) * 100 if total_contributed > 0 else 0.0
        if timeframe_years > 0 and total_contributed > 0:
            annualized_return = ((ending_value / total_contributed) ** (1 / timeframe_years) - 1) * 100
        else:
            annualized_return = 0.0
        
        levels = [item.get('level') for item in timeline_data if 'level' in item]
        if levels and len(levels) > 1 and all(np.isfinite(x) for x in levels):
            daily_returns = [levels[i] / levels[i-1] - 1 for i in range(1, len(levels))]
            volatility = np.std(daily_returns) * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(levels)
        else:
            volatility, max_drawdown = 0.0, 0.0
        
        risk_free_rate = self.config.simulation.default_risk_free_rate
        sharpe_ratio = (annualized_return / 100 - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        return PortfolioMetrics(
            starting_value=starting_value,
            ending_value=ending_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility * 100,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,
            total_contributed=total_contributed,
            profit_loss=profit_loss
        )
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        max_drawdown, peak = 0.0, values[0]
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown