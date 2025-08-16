"""
Simulation Engine Module

This module handles the core portfolio growth simulation with enhanced debugging,
error handling, and fallback mechanisms.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimulationEngine:
    """
    Core simulation engine for portfolio growth calculations.
    
    Features:
    - Enhanced debugging and error detection
    - Robust fallback mechanisms
    - Comprehensive validation
    - Detailed progress tracking
    """
    
    def __init__(self):
        """Initialize the simulation engine."""
        logger.info("📈 SimulationEngine initialized")
    
    def simulate_portfolio_growth(self, data: pd.DataFrame, weights: np.ndarray, 
                                lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
        """
        Enhanced portfolio growth simulation with comprehensive debugging and error handling.
        
        Args:
            data: Historical stock price data
            weights: Portfolio allocation weights
            lump_sum: Initial investment amount
            monthly: Monthly contribution amount
            timeframe: Investment timeframe in years
            
        Returns:
            Simulation results with timeline data
        """
        try:
            logger.info(f"📈 Starting simulation: £{lump_sum:,.2f} initial + £{monthly:,.2f}/month for {timeframe} years")
            
            # Step 1: Validate inputs
            self._validate_simulation_inputs(data, weights, lump_sum, monthly, timeframe)
            
            # Step 2: Prepare and clean data
            cleaned_data = self._prepare_simulation_data(data)
            
            # Step 3: Calculate portfolio performance
            portfolio_performance = self._calculate_portfolio_performance(cleaned_data, weights)
            
            # Step 4: Run the simulation
            simulation_results = self._execute_simulation(
                portfolio_performance, lump_sum, monthly, timeframe
            )
            
            # Step 5: Validate results
            self._validate_simulation_results(simulation_results)
            
            logger.info(f"✅ Simulation completed successfully")
            return simulation_results
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio simulation: {str(e)}")
            return self._run_fallback_simulation(lump_sum, monthly, timeframe, str(e))
    
    def _validate_simulation_inputs(self, data: pd.DataFrame, weights: np.ndarray, 
                                  lump_sum: float, monthly: float, timeframe: int):
        """Validate all simulation inputs."""
        
        # Data validation
        if data.empty:
            raise ValueError("Stock data is empty")
        
        if data.isnull().all().any():
            raise ValueError("Stock data contains columns with all NaN values")
        
        # Weights validation
        if len(weights) != len(data.columns):
            raise ValueError(f"Weights length ({len(weights)}) doesn't match data columns ({len(data.columns)})")
        
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            raise ValueError(f"Weights don't sum to 1.0 (sum: {np.sum(weights)})")
        
        if np.any(weights < 0):
            raise ValueError("Negative weights detected")
        
        # Investment validation
        if lump_sum < 0 or monthly < 0:
            raise ValueError("Investment amounts cannot be negative")
        
        if lump_sum == 0 and monthly == 0:
            raise ValueError("Either lump sum or monthly investment must be greater than 0")
        
        if timeframe <= 0:
            raise ValueError("Timeframe must be greater than 0")
        
        logger.info("✅ All simulation inputs validated")
    
    def _prepare_simulation_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean data for simulation."""
        
        logger.info(f"📊 Preparing data - shape: {data.shape}")
        
        # Handle missing values
        if data.isnull().any().any():
            logger.warning("⚠️ Found NaN values in stock data")
            data = data.fillna(method='ffill').fillna(method='bfill')
            logger.info("✅ NaN values filled using forward/backward fill")
        
        # Check for zero or near-zero values in first row
        first_row = data.iloc[0]
        problematic_values = first_row[(first_row == 0) | (np.abs(first_row) < 1e-10)]
        
        if not problematic_values.empty:
            logger.error(f"❌ Found zero or near-zero values in first row: {problematic_values}")
            raise ValueError("Invalid starting prices in stock data")
        
        logger.info("✅ Data preparation completed")
        return data
    
    def _calculate_portfolio_performance(self, data: pd.DataFrame, weights: np.ndarray) -> pd.Series:
        """Calculate weighted portfolio performance."""
        
        logger.info("📊 Calculating portfolio performance")
        
        # Normalize data to starting values
        first_row = data.iloc[0]
        normalized = data.div(first_row)
        
        logger.info(f"📊 Normalized data range: {normalized.min().min():.4f} to {normalized.max().max():.4f}")
        
        # Check for issues in normalized data
        if normalized.isnull().any().any():
            raise ValueError("Normalization created NaN values")
        
        # Apply portfolio weights
        weighted_performance = normalized.dot(weights)
        
        # Validate weighted performance
        if weighted_performance.isnull().any():
            raise ValueError("Weighted portfolio contains NaN values")
        
        if (weighted_performance <= 0).any():
            zero_count = (weighted_performance <= 0).sum()
            logger.warning(f"⚠️ Found {zero_count} zero or negative values in weighted portfolio")
        
        logger.info(f"📊 Portfolio performance calculated - range: {weighted_performance.min():.4f} to {weighted_performance.max():.4f}")
        return weighted_performance
    
    def _execute_simulation(self, portfolio_performance: pd.Series, 
                          lump_sum: float, monthly: float, timeframe: int) -> Dict[str, Any]:
        """Execute the main simulation loop."""
        
        logger.info("🚀 Executing simulation loop")
        
        # Initialize tracking variables
        portfolio_values = []
        contributions = []
        current_value = float(lump_sum)
        total_contributions = float(lump_sum)
        
        logger.info(f"💰 Starting values - Portfolio: £{current_value}, Contributions: £{total_contributions}")
        
        # Simulation loop
        for i, (date, growth_factor) in enumerate(portfolio_performance.items()):
            try:
                # Add monthly contribution (every ~21 trading days)
                if i > 0 and i % 21 == 0:
                    current_value += monthly
                    total_contributions += monthly
                    
                    if i < 100:  # Log first few contributions
                        logger.debug(f"📅 Month {i//21}: Added £{monthly}, Total contributions: £{total_contributions}")
                
                # Apply growth (but not on first day)
                if i > 0:
                    previous_factor = portfolio_performance.iloc[i - 1]
                    
                    # Check for division by zero
                    if abs(previous_factor) < 1e-10:
                        logger.error(f"❌ Division by zero at index {i}: previous_factor={previous_factor}")
                        raise ValueError(f"Division by zero in growth calculation at index {i}")
                    
                    growth_rate = growth_factor / previous_factor
                    
                    # Check for invalid growth rate
                    if not np.isfinite(growth_rate):
                        logger.error(f"❌ Invalid growth rate at index {i}: {growth_rate}")
                        raise ValueError(f"Invalid growth rate: {growth_rate}")
                    
                    # Apply growth
                    old_value = current_value
                    current_value *= growth_rate
                    
                    # Log significant changes
                    if i < 10 or abs(growth_rate - 1) > 0.1:
                        logger.debug(f"📈 Day {i}: Growth {growth_rate:.4f}, Value £{old_value:.2f} -> £{current_value:.2f}")
                    
                    # Check if portfolio value became zero or negative
                    if current_value <= 0:
                        logger.error(f"❌ Portfolio value became {current_value} at index {i}")
                        raise ValueError(f"Portfolio value became {current_value}")
                
                # Record values
                contributions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": round(float(total_contributions), 2)
                })
                portfolio_values.append({
                    "date": date.strftime("%Y-%m-%d"), 
                    "value": round(float(current_value), 2)
                })
                
                # Periodic progress logging
                if i % 252 == 0 and i > 0:  # Yearly
                    logger.info(f"📅 Year {i//252}: Portfolio £{current_value:,.2f}, Contributions £{total_contributions:,.2f}")
                
            except Exception as loop_error:
                logger.error(f"❌ Error in simulation loop at index {i}: {loop_error}")
                raise ValueError(f"Simulation failed at day {i}: {loop_error}")
        
        # Calculate final results
        end_value = float(current_value)
        starting_value = float(total_contributions)
        portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0
        
        logger.info(f"📊 Final results: £{starting_value:,.2f} invested -> £{end_value:,.2f} final ({portfolio_return:.1%} return)")
        
        return {
            "starting_value": round(starting_value, 2),
            "end_value": round(end_value, 2),
            "portfolio_return": round(portfolio_return, 4),
            "timeline": {
                "contributions": contributions,
                "portfolio": portfolio_values
            },
            "simulation_metadata": {
                "total_days": len(portfolio_performance),
                "total_contributions": round(total_contributions, 2),
                "growth_contribution": round(end_value - total_contributions, 2)
            }
        }
    
    def _validate_simulation_results(self, results: Dict[str, Any]):
        """Validate simulation results."""
        
        end_value = results.get("end_value", 0)
        starting_value = results.get("starting_value", 0)
        portfolio_values = results.get("timeline", {}).get("portfolio", [])
        
        # Check final values
        if end_value <= 0:
            raise ValueError(f"Invalid final portfolio value: {end_value}")
        
        if starting_value <= 0:
            raise ValueError(f"Invalid starting value: {starting_value}")
        
        # Check timeline data
        if len(portfolio_values) == 0:
            raise ValueError("No portfolio timeline data generated")
        
        # Check for reasonable values
        if end_value > starting_value * 1000:  # 100,000% return seems unrealistic
            logger.warning(f"⚠️ Unusually high return detected: {(end_value/starting_value - 1)*100:.1f}%")
        
        logger.info("✅ Simulation results validated")
    
    def _run_fallback_simulation(self, lump_sum: float, monthly: float, 
                                timeframe: int, original_error: str) -> Dict[str, Any]:
        """
        Run fallback simulation with conservative assumptions.
        """
        
        logger.warning("🔄 Running fallback simulation with conservative assumptions")
        
        try:
            # Calculate total invested
            starting_value = lump_sum + monthly * 12 * timeframe
            
            # Use moderate 6% annual growth as fallback
            annual_growth = 1.06
            end_value = lump_sum * (annual_growth ** timeframe)
            
            # Add future value of monthly contributions (annuity formula)
            if monthly > 0:
                months = timeframe * 12
                monthly_growth = annual_growth ** (1/12)
                if monthly_growth != 1:
                    fv_annuity = monthly * (((monthly_growth ** months) - 1) / (monthly_growth - 1))
                else:
                    fv_annuity = monthly * months
                end_value += fv_annuity
            
            portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0
            
            logger.info(f"🔄 Fallback results: £{starting_value:,.2f} -> £{end_value:,.2f} ({portfolio_return:.1%})")
            
            # Create simple timeline
            start_date = datetime.today()
            end_date = start_date + timedelta(days=timeframe * 365)
            
            # Create monthly timeline for better visualization
            timeline_contributions = []
            timeline_portfolio = []
            
            for year in range(timeframe + 1):
                current_date = start_date + timedelta(days=year * 365)
                
                # Calculate values at this point
                contrib_value = lump_sum + (monthly * 12 * year)
                if year == 0:
                    portfolio_value = lump_sum
                else:
                    portfolio_value = lump_sum * (annual_growth ** year)
                    if monthly > 0:
                        months_elapsed = year * 12
                        monthly_growth = annual_growth ** (1/12)
                        if monthly_growth != 1:
                            fv_annuity = monthly * (((monthly_growth ** months_elapsed) - 1) / (monthly_growth - 1))
                        else:
                            fv_annuity = monthly * months_elapsed
                        portfolio_value += fv_annuity
                
                timeline_contributions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "value": round(contrib_value, 2)
                })
                timeline_portfolio.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "value": round(portfolio_value, 2)
                })
            
            return {
                "starting_value": round(starting_value, 2),
                "end_value": round(end_value, 2),
                "portfolio_return": round(portfolio_return, 4),
                "timeline": {
                    "contributions": timeline_contributions,
                    "portfolio": timeline_portfolio
                },
                "simulation_metadata": {
                    "fallback_used": True,
                    "original_error": original_error,
                    "fallback_assumptions": {
                        "annual_growth_rate": "6%",
                        "methodology": "Simple compound interest with annuity calculation"
                    }
                }
            }
            
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback simulation failed: {fallback_error}")
            raise ValueError(f"Both main and fallback simulations failed: {original_error}, {fallback_error}")
    
    def get_simulation_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional statistics about the simulation.
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            Dictionary with additional statistics
        """
        
        try:
            portfolio_timeline = results.get("timeline", {}).get("portfolio", [])
            contribution_timeline = results.get("timeline", {}).get("contributions", [])
            
            if not portfolio_timeline:
                return {"error": "No portfolio timeline data available"}
            
            # Extract values
            portfolio_values = [item["value"] for item in portfolio_timeline]
            contribution_values = [item["value"] for item in contribution_timeline]
            
            # Calculate statistics
            max_value = max(portfolio_values)
            min_value = min(portfolio_values)
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            volatility = self._calculate_volatility(portfolio_values)
            sharpe_ratio = self._calculate_simple_sharpe_ratio(portfolio_values, contribution_values)
            
            return {
                "max_portfolio_value": round(max_value, 2),
                "min_portfolio_value": round(min_value, 2),
                "max_drawdown_percent": round(max_drawdown * 100, 2),
                "volatility_percent": round(volatility * 100, 2),
                "simple_sharpe_ratio": round(sharpe_ratio, 2),
                "total_growth_amount": round(portfolio_values[-1] - contribution_values[-1], 2),
                "average_annual_growth": round(((portfolio_values[-1] / contribution_values[0]) ** (1/len(portfolio_values)) - 1) * 252 * 100, 2) if len(portfolio_values) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating simulation statistics: {e}")
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from peak."""
        
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
        
        return max_dd
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate simple volatility measure."""
        
        if len(values) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                returns.append((values[i] - values[i-1]) / values[i-1])
        
        if not returns:
            return 0.0
        
        # Return standard deviation of returns
        return np.std(returns) if len(returns) > 1 else 0.0
    
    def _calculate_simple_sharpe_ratio(self, portfolio_values: List[float], 
                                     contribution_values: List[float]) -> float:
        """Calculate a simplified Sharpe ratio."""
        
        if len(portfolio_values) < 2:
            return 0.0
        
        try:
            # Calculate total return
            total_return = (portfolio_values[-1] - contribution_values[-1]) / contribution_values[0]
            
            # Calculate volatility
            volatility = self._calculate_volatility(portfolio_values)
            
            # Simple Sharpe ratio (excess return / volatility)
            # Assuming 2% risk-free rate
            risk_free_rate = 0.02
            excess_return = total_return - risk_free_rate
            
            return excess_return / volatility if volatility > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def compare_scenarios(self, base_results: Dict[str, Any], 
                         alternative_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple simulation scenarios.
        
        Args:
            base_results: Base scenario results
            alternative_results: List of alternative scenario results
            
        Returns:
            Comparison analysis
        """
        
        try:
            comparisons = []
            
            base_final_value = base_results.get("end_value", 0)
            base_return = base_results.get("portfolio_return", 0)
            
            for i, alt_result in enumerate(alternative_results):
                alt_final_value = alt_result.get("end_value", 0)
                alt_return = alt_result.get("portfolio_return", 0)
                
                comparison = {
                    "scenario_name": f"Alternative {i + 1}",
                    "final_value": alt_final_value,
                    "return_percent": alt_return * 100,
                    "vs_base_value_diff": alt_final_value - base_final_value,
                    "vs_base_return_diff": (alt_return - base_return) * 100,
                    "better_than_base": alt_final_value > base_final_value
                }
                
                comparisons.append(comparison)
            
            # Find best scenario
            best_scenario = max(comparisons, key=lambda x: x["final_value"])
            
            return {
                "base_scenario": {
                    "final_value": base_final_value,
                    "return_percent": base_return * 100
                },
                "comparisons": comparisons,
                "best_alternative": best_scenario,
                "summary": {
                    "scenarios_analyzed": len(alternative_results) + 1,
                    "best_improvement": best_scenario["vs_base_value_diff"] if best_scenario["vs_base_value_diff"] > 0 else 0,
                    "scenarios_better_than_base": len([c for c in comparisons if c["better_than_base"]])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error comparing scenarios: {e}")
            return {"error": str(e)}