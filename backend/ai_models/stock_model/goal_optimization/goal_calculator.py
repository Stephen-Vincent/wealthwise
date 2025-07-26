"""
Goal-Oriented Financial Calculations Module

This module implements the core innovation of goal-oriented portfolio optimization.
Instead of just matching portfolios to risk tolerance, it calculates exactly
what return is needed to reach the user's specific financial goal and designs
the portfolio around that requirement.

Key Features:
1. Required Return Calculation - What annual return is needed to reach goals
2. Goal Timeline Analysis - How different timeframes affect requirements
3. Contribution Impact Analysis - How monthly savings affect outcomes
4. Scenario Planning - What-if analysis for different parameters
5. Goal Adjustment Recommendations - Suggestions for unrealistic goals
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class GoalCalculator:
    """
    Goal-Oriented Financial Calculator
    
    This class implements sophisticated financial calculations that form the
    foundation of goal-oriented portfolio optimization. It answers the key
    question: "What return do I need to reach my financial goal?"
    
    Core Innovation: Traditional robo-advisors ask "What's your risk tolerance?"
    and give you a generic portfolio. This system asks "What do you want to achieve?"
    and designs a portfolio specifically to reach that goal.
    """
    
    def __init__(self):
        """Initialize the goal calculator with default parameters"""
        # Default assumptions for calculations
        self.default_inflation_rate = 0.025  # 2.5% annual inflation
        self.default_tax_rate = 0.20         # 20% capital gains tax
        self.min_annual_return = 0.0         # 0% minimum return
        self.max_annual_return = 0.25        # 25% maximum realistic return
        
        # Cache for expensive calculations
        self._calculation_cache = {}
        self._cache_timeout = 3600  # 1 hour cache
    
    def calculate_required_return(self, target_value: float, current_investment: float,
                                timeframe: int, monthly_contribution: float = 0,
                                include_inflation: bool = True) -> Dict[str, float]:
        """
        Calculate the annual return required to reach a financial goal
        
        This is the CORE INNOVATION: Goal-oriented optimization
        Instead of generic portfolios, we calculate exactly what return is needed
        and design the investment strategy around that specific requirement.
        
        Formula: CAGR = (Ending Value / Beginning Value)^(1/years) - 1
        Where Beginning Value includes all future contributions
        
        Args:
            target_value: The user's financial goal (e.g., £50,000)
            current_investment: Money they're starting with (e.g., £5,000)
            timeframe: Years to reach the goal (e.g., 10)
            monthly_contribution: Regular monthly investments (e.g., £300)
            include_inflation: Whether to adjust for inflation
            
        Returns:
            Dict containing:
            - required_return: Annual return needed (decimal)
            - required_return_percent: Annual return needed (percentage)
            - total_contributions: Total money invested over timeframe
            - inflation_adjusted_target: Target adjusted for inflation
            - feasibility_rating: How realistic the goal is (1-5 scale)
            
        Example:
            Goal: £50,000 in 10 years
            Starting: £5,000 + £200/month = £29,000 total invested
            Required return: (50,000 / 29,000)^(1/10) - 1 = 5.6% annually
        """
        try:
            logger.debug(f"Calculating required return: £{target_value:,.0f} in {timeframe} years")
            
            # Input validation
            if target_value <= 0:
                raise ValueError("Target value must be positive")
            if timeframe <= 0:
                raise ValueError("Timeframe must be positive")
            if current_investment < 0:
                raise ValueError("Current investment cannot be negative")
            if monthly_contribution < 0:
                raise ValueError("Monthly contribution cannot be negative")
            
            # Calculate total contributions over the timeframe
            total_monthly_contributions = monthly_contribution * 12 * timeframe
            total_contributions = current_investment + total_monthly_contributions
            
            # Handle edge case where no money is invested
            if total_contributions <= 0:
                logger.warning("No money invested, using default 10% return assumption")
                return {
                    "required_return": 0.10,
                    "required_return_percent": 10.0,
                    "total_contributions": 0,
                    "inflation_adjusted_target": target_value,
                    "feasibility_rating": 1,
                    "calculation_method": "default_no_investment"
                }
            
            # Adjust target for inflation if requested
            inflation_adjusted_target = target_value
            if include_inflation and timeframe > 1:
                inflation_multiplier = (1 + self.default_inflation_rate) ** timeframe
                inflation_adjusted_target = target_value * inflation_multiplier
                logger.debug(f"Inflation-adjusted target: £{inflation_adjusted_target:,.0f}")
            
            # Calculate required compound annual growth rate (CAGR)
            # This is the magic number: what return do we need to hit the goal?
            required_multiplier = inflation_adjusted_target / total_contributions
            
            if required_multiplier <= 1.0:
                # Goal is already achievable with contributions alone
                required_annual_return = 0.0
                logger.info("Goal achievable with contributions alone, no investment return needed")
            else:
                required_annual_return = (required_multiplier ** (1/timeframe)) - 1
            
            # Apply reasonable bounds
            required_annual_return = max(self.min_annual_return, 
                                       min(self.max_annual_return, required_annual_return))
            
            # Calculate feasibility rating (1-5 scale)
            feasibility_rating = self._calculate_feasibility_rating(required_annual_return)
            
            result = {
                "required_return": required_annual_return,
                "required_return_percent": required_annual_return * 100,
                "total_contributions": total_contributions,
                "inflation_adjusted_target": inflation_adjusted_target,
                "feasibility_rating": feasibility_rating,
                "calculation_method": "standard_cagr",
                "monthly_contribution_total": total_monthly_contributions,
                "starting_investment": current_investment,
                "target_multiplier": required_multiplier
            }
            
            logger.info(f"💡 Required return: {required_annual_return:.1%} annually for £{target_value:,.0f} goal")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating required return: {e}")
            # Return safe default values
            return {
                "required_return": 0.10,
                "required_return_percent": 10.0,
                "total_contributions": current_investment,
                "inflation_adjusted_target": target_value,
                "feasibility_rating": 3,
                "calculation_method": "error_fallback",
                "error": str(e)
            }
    
    def calculate_future_value(self, present_value: float, annual_return: float,
                             timeframe: int, monthly_contribution: float = 0) -> Dict[str, float]:
        """
        Calculate future value given return assumptions
        
        This is the inverse of required return calculation. Given an expected
        return, what will the portfolio be worth in the future?
        
        Args:
            present_value: Current investment amount
            annual_return: Expected annual return (decimal)
            timeframe: Investment period in years
            monthly_contribution: Regular monthly investments
            
        Returns:
            Dict with future value projections and breakdown
        """
        try:
            logger.debug(f"Calculating future value: £{present_value:,.0f} at {annual_return:.1%}")
            
            # Validate inputs
            if timeframe <= 0:
                raise ValueError("Timeframe must be positive")
            
            # Monthly return for contribution calculations
            monthly_return = annual_return / 12
            total_months = timeframe * 12
            
            # Future value of current investment (lump sum)
            future_value_lump_sum = present_value * ((1 + annual_return) ** timeframe)
            
            # Future value of monthly contributions (annuity)
            future_value_contributions = 0
            if monthly_contribution > 0 and monthly_return != 0:
                # Standard annuity formula
                future_value_contributions = monthly_contribution * (
                    ((1 + monthly_return) ** total_months - 1) / monthly_return
                )
            elif monthly_contribution > 0:
                # If return is 0, it's just sum of contributions
                future_value_contributions = monthly_contribution * total_months
            
            # Total future value
            total_future_value = future_value_lump_sum + future_value_contributions
            
            # Calculate total contributions
            total_contributions = present_value + (monthly_contribution * total_months)
            
            # Calculate total gains
            total_gains = total_future_value - total_contributions
            
            result = {
                "total_future_value": total_future_value,
                "future_value_lump_sum": future_value_lump_sum,
                "future_value_contributions": future_value_contributions,
                "total_contributions": total_contributions,
                "total_gains": total_gains,
                "effective_return": (total_future_value / total_contributions - 1) if total_contributions > 0 else 0,
                "annual_return_used": annual_return
            }
            
            logger.debug(f"Future value: £{total_future_value:,.0f} (gains: £{total_gains:,.0f})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating future value: {e}")
            return {
                "total_future_value": present_value,
                "future_value_lump_sum": present_value,
                "future_value_contributions": 0,
                "total_contributions": present_value,
                "total_gains": 0,
                "effective_return": 0,
                "annual_return_used": annual_return,
                "error": str(e)
            }
    
    def analyze_contribution_impact(self, target_value: float, current_investment: float,
                                  timeframe: int, contribution_range: Tuple[float, float] = (0, 1000),
                                  num_scenarios: int = 20) -> pd.DataFrame:
        """
        Analyze how different monthly contribution levels affect goal achievement
        
        This helps users understand the trade-off between monthly savings
        and required investment returns.
        
        Args:
            target_value: Financial goal
            current_investment: Starting amount
            timeframe: Years to goal
            contribution_range: (min, max) monthly contribution to analyze
            num_scenarios: Number of scenarios to calculate
            
        Returns:
            DataFrame with contribution scenarios and required returns
        """
        try:
            logger.info(f"Analyzing contribution impact for £{target_value:,.0f} goal")
            
            # Generate contribution scenarios
            min_contrib, max_contrib = contribution_range
            contribution_levels = np.linspace(min_contrib, max_contrib, num_scenarios)
            
            scenarios = []
            
            for monthly_contrib in contribution_levels:
                # Calculate required return for this contribution level
                result = self.calculate_required_return(
                    target_value, current_investment, timeframe, monthly_contrib
                )
                
                # Calculate total invested
                total_invested = current_investment + (monthly_contrib * 12 * timeframe)
                
                scenarios.append({
                    "monthly_contribution": monthly_contrib,
                    "total_invested": total_invested,
                    "required_return_percent": result["required_return_percent"],
                    "feasibility_rating": result["feasibility_rating"],
                    "annual_savings": monthly_contrib * 12
                })
            
            df = pd.DataFrame(scenarios)
            
            # Add insights
            df["return_category"] = df["required_return_percent"].apply(self._categorize_return)
            df["investment_ratio"] = df["total_invested"] / target_value
            
            logger.info(f"✅ Generated {len(df)} contribution scenarios")
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing contribution impact: {e}")
            return pd.DataFrame()
    
    def analyze_timeframe_impact(self, target_value: float, current_investment: float,
                               monthly_contribution: float, timeframe_range: Tuple[int, int] = (1, 30),
                               num_scenarios: int = 15) -> pd.DataFrame:
        """
        Analyze how different timeframes affect required returns
        
        Helps users understand the power of time in investing and how
        longer timeframes can make goals more achievable.
        
        Args:
            target_value: Financial goal
            current_investment: Starting amount
            monthly_contribution: Regular monthly investment
            timeframe_range: (min, max) years to analyze
            num_scenarios: Number of timeframe scenarios
            
        Returns:
            DataFrame with timeframe scenarios and required returns
        """
        try:
            logger.info(f"Analyzing timeframe impact for £{target_value:,.0f} goal")
            
            min_years, max_years = timeframe_range
            timeframes = np.linspace(min_years, max_years, num_scenarios, dtype=int)
            
            scenarios = []
            
            for years in timeframes:
                if years < 1:
                    continue
                    
                # Calculate required return for this timeframe
                result = self.calculate_required_return(
                    target_value, current_investment, years, monthly_contribution
                )
                
                scenarios.append({
                    "timeframe_years": years,
                    "required_return_percent": result["required_return_percent"],
                    "total_contributions": result["total_contributions"],
                    "feasibility_rating": result["feasibility_rating"],
                    "inflation_adjusted_target": result["inflation_adjusted_target"]
                })
            
            df = pd.DataFrame(scenarios)
            
            # Add insights
            df["return_category"] = df["required_return_percent"].apply(self._categorize_return)
            df["time_benefit"] = df["required_return_percent"].iloc[0] - df["required_return_percent"]
            
            logger.info(f"✅ Generated {len(df)} timeframe scenarios")
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe impact: {e}")
            return pd.DataFrame()
    
    def create_goal_scenarios(self, base_target: float, current_investment: float,
                            timeframe: int, monthly_contribution: float) -> Dict[str, Dict]:
        """
        Create multiple goal scenarios for comparison
        
        Generates conservative, moderate, and aggressive goal scenarios
        to help users understand different outcome possibilities.
        
        Args:
            base_target: User's stated goal
            current_investment: Starting investment
            timeframe: Investment timeframe
            monthly_contribution: Monthly savings
            
        Returns:
            Dict with different goal scenarios and their requirements
        """
        try:
            logger.info(f"Creating goal scenarios for £{base_target:,.0f} base target")
            
            # Define scenario variations
            scenarios = {
                "conservative": {
                    "target_multiplier": 0.75,
                    "description": "75% of original goal - more achievable"
                },
                "base_goal": {
                    "target_multiplier": 1.0,
                    "description": "Your original goal"
                },
                "stretch_goal": {
                    "target_multiplier": 1.25,
                    "description": "25% higher - ambitious but possible"
                },
                "aggressive": {
                    "target_multiplier": 1.5,
                    "description": "50% higher - very ambitious"
                }
            }
            
            results = {}
            
            for scenario_name, scenario_config in scenarios.items():
                target = base_target * scenario_config["target_multiplier"]
                
                # Calculate requirements for this scenario
                calculation = self.calculate_required_return(
                    target, current_investment, timeframe, monthly_contribution
                )
                
                # Add scenario-specific information
                results[scenario_name] = {
                    "target_value": target,
                    "description": scenario_config["description"],
                    "required_return_percent": calculation["required_return_percent"],
                    "feasibility_rating": calculation["feasibility_rating"],
                    "total_contributions": calculation["total_contributions"],
                    "recommendation": self._get_scenario_recommendation(calculation["feasibility_rating"])
                }
            
            logger.info(f"✅ Created {len(results)} goal scenarios")
            return results
            
        except Exception as e:
            logger.error(f"Error creating goal scenarios: {e}")
            return {}
    
    def calculate_retirement_needs(self, current_age: int, retirement_age: int,
                                 desired_annual_income: float, 
                                 current_savings: float = 0,
                                 monthly_contribution: float = 0) -> Dict[str, float]:
        """
        Calculate retirement planning requirements
        
        Specialized calculation for retirement goals using the 4% withdrawal rule
        and life expectancy assumptions.
        
        Args:
            current_age: Current age
            retirement_age: Planned retirement age
            desired_annual_income: Desired income in retirement
            current_savings: Current retirement savings
            monthly_contribution: Monthly retirement contributions
            
        Returns:
            Dict with retirement planning analysis
        """
        try:
            logger.info(f"Calculating retirement needs: £{desired_annual_income:,.0f} annual income")
            
            # Validate inputs
            if current_age >= retirement_age:
                raise ValueError("Current age must be less than retirement age")
            
            years_to_retirement = retirement_age - current_age
            
            # Calculate required retirement fund using 4% rule
            # Assumes you can withdraw 4% annually without depleting principal
            required_retirement_fund = desired_annual_income / 0.04
            
            # Adjust for inflation over the accumulation period
            inflation_adjusted_fund = required_retirement_fund * (
                (1 + self.default_inflation_rate) ** years_to_retirement
            )
            
            # Calculate required return
            return_calculation = self.calculate_required_return(
                inflation_adjusted_fund, current_savings, 
                years_to_retirement, monthly_contribution
            )
            
            # Calculate additional retirement-specific metrics
            total_contributions = return_calculation["total_contributions"]
            
            result = {
                "required_retirement_fund": required_retirement_fund,
                "inflation_adjusted_fund": inflation_adjusted_fund,
                "years_to_retirement": years_to_retirement,
                "required_annual_return": return_calculation["required_return"],
                "required_return_percent": return_calculation["required_return_percent"],
                "total_contributions_needed": total_contributions,
                "annual_contribution_needed": monthly_contribution * 12,
                "savings_gap": max(0, inflation_adjusted_fund - total_contributions),
                "feasibility_rating": return_calculation["feasibility_rating"],
                "withdrawal_rule_used": "4% rule"
            }
            
            logger.info(f"Retirement analysis: £{inflation_adjusted_fund:,.0f} needed, {return_calculation['required_return_percent']:.1f}% return required")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating retirement needs: {e}")
            return {"error": str(e)}
    
    def _calculate_feasibility_rating(self, required_return: float) -> int:
        """
        Calculate feasibility rating (1-5 scale) based on required return
        
        1 = Very difficult (>20% annual return needed)
        2 = Difficult (15-20% annual return needed)
        3 = Challenging (10-15% annual return needed)
        4 = Achievable (5-10% annual return needed)
        5 = Highly achievable (<5% annual return needed)
        """
        if required_return > 0.20:
            return 1  # Very difficult
        elif required_return > 0.15:
            return 2  # Difficult
        elif required_return > 0.10:
            return 3  # Challenging
        elif required_return > 0.05:
            return 4  # Achievable
        else:
            return 5  # Highly achievable
    
    def _categorize_return(self, return_percent: float) -> str:
        """Categorize required return for easy understanding"""
        if return_percent > 20:
            return "Very High Risk"
        elif return_percent > 15:
            return "High Risk"
        elif return_percent > 10:
            return "Moderate-High Risk"
        elif return_percent > 5:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def _get_scenario_recommendation(self, feasibility_rating: int) -> str:
        """Get recommendation based on feasibility rating"""
        recommendations = {
            1: "Consider extending timeframe or increasing contributions",
            2: "Challenging but possible with aggressive investing",
            3: "Achievable with balanced growth strategy",
            4: "Realistic goal with moderate risk approach",
            5: "Highly achievable with conservative approach"
        }
        return recommendations.get(feasibility_rating, "Assess risk tolerance")