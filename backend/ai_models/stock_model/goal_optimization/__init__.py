"""
Goal Optimization Module for WealthWise Enhanced Stock Recommender

This module implements the core innovation of goal-oriented portfolio optimization.
Instead of generic risk-based portfolios, it calculates exactly what return is
needed to reach specific financial goals and designs investment strategies accordingly.

Core Philosophy:
Traditional robo-advisors ask: "What's your risk tolerance?" → Generic portfolio
This system asks: "What do you want to achieve?" → Goal-specific strategy

Key Components:
1. GoalCalculator - Required return calculations and scenario analysis
2. FeasibilityAssessor - Reality checks and honest feedback about goal achievability

Key Features:
- Required return calculation for any financial goal
- Inflation-adjusted projections
- Contribution impact analysis
- Timeframe optimization
- Monte Carlo probability analysis
- Honest feasibility assessment
- Specific improvement recommendations
- Alternative scenario generation

Example Usage:
    from ai_models.stock_model.goal_optimization import GoalCalculator, FeasibilityAssessor
    
    # Calculate what return is needed
    calculator = GoalCalculator()
    goal_analysis = calculator.calculate_required_return(
        target_value=50000,      # £50k goal
        current_investment=5000,  # £5k starting amount
        timeframe=10,            # 10 years
        monthly_contribution=300  # £300/month
    )
    
    # Assess if goal is realistic
    assessor = FeasibilityAssessor()
    feasibility = assessor.assess_goal_feasibility(
        required_return=goal_analysis["required_return"],
        risk_score=65,           # User's risk tolerance
        timeframe=10
    )
    
    print(f"Need {goal_analysis['required_return_percent']:.1f}% annual return")
    print(f"Feasibility: {feasibility['feasibility_score']:.0f}%")
    print(f"Recommendation: {feasibility['recommendations']['primary']}")

Mathematical Foundation:
The core calculation uses the compound annual growth rate (CAGR) formula:
    CAGR = (Ending Value / Beginning Value)^(1/years) - 1

Where Beginning Value includes all contributions made over the investment period.
This tells us exactly what annual return is needed to reach any goal.

Educational Value:
This module helps users understand:
- The relationship between time, risk, and returns
- How monthly contributions dramatically affect required returns
- Why longer timeframes make goals more achievable
- The trade-offs between different goal scenarios
- Realistic expectations based on historical market performance
"""

from .goal_calculator import GoalCalculator
from .feasibility_assessor import FeasibilityAssessor
from typing import Dict, List, Optional, Any

__all__ = [
    'GoalCalculator',
    'FeasibilityAssessor'
]

# Version and metadata
__version__ = '1.0.0'
__author__ = 'WealthWise Team'
__description__ = 'Goal-oriented financial planning and portfolio optimization'

# Key mathematical constants used throughout the module
FINANCIAL_CONSTANTS = {
    'DEFAULT_INFLATION_RATE': 0.025,      # 2.5% annual inflation assumption
    'DEFAULT_TAX_RATE': 0.20,             # 20% capital gains tax assumption
    'WITHDRAWAL_RATE': 0.04,              # 4% safe withdrawal rate for retirement
    'MIN_FEASIBLE_RETURN': 0.0,           # 0% minimum realistic return
    'MAX_FEASIBLE_RETURN': 0.25,          # 25% maximum realistic return
    'MARKET_VOLATILITY': 0.16             # 16% typical stock market volatility
}

# Risk tolerance to expected return mapping
# Based on historical market performance and academic research
RISK_RETURN_PROFILES = {
    'ultra_conservative': {'return': 0.05, 'volatility': 0.08},   # Bonds focus
    'conservative': {'return': 0.07, 'volatility': 0.10},         # Balanced conservative
    'moderate': {'return': 0.09, 'volatility': 0.12},             # Balanced growth
    'moderate_aggressive': {'return': 0.11, 'volatility': 0.15},  # Growth focus
    'aggressive': {'return': 0.13, 'volatility': 0.18},           # High growth
    'ultra_aggressive': {'return': 0.16, 'volatility': 0.22}      # Maximum growth
}

def quick_goal_analysis(target_value: float, current_investment: float, 
                       timeframe: int, monthly_contribution: float = 0,
                       risk_score: float = 50) -> Dict[str, Any]:
    """
    Quick goal analysis combining calculation and feasibility assessment
    
    Convenience function that provides a complete goal analysis in one call.
    Perfect for initial goal evaluation and quick feasibility checks.
    
    Args:
        target_value: Financial goal amount
        current_investment: Starting investment
        timeframe: Years to reach goal
        monthly_contribution: Regular monthly savings
        risk_score: Risk tolerance (0-100)
        
    Returns:
        Dict with comprehensive goal analysis including:
        - Required return calculation
        - Feasibility assessment
        - Recommendations
        - Alternative scenarios
    """
    try:
        # Calculate required return
        calculator = GoalCalculator()
        goal_calc = calculator.calculate_required_return(
            target_value, current_investment, timeframe, monthly_contribution
        )
        
        # Assess feasibility
        assessor = FeasibilityAssessor()
        feasibility = assessor.assess_goal_feasibility(
            required_return=goal_calc["required_return"],
            risk_score=risk_score,
            timeframe=timeframe,
            current_investment=current_investment,
            monthly_contribution=monthly_contribution
        )
        
        # Combine results
        return {
            "goal_calculation": goal_calc,
            "feasibility_assessment": feasibility,
            "summary": {
                "required_return_percent": goal_calc["required_return_percent"],
                "feasibility_score": feasibility["feasibility_score"],
                "primary_recommendation": feasibility["recommendations"]["primary"],
                "is_achievable": feasibility["feasibility_score"] >= 60
            }
        }
        
    except Exception as e:
        return {
            "error": f"Quick goal analysis failed: {str(e)}",
            "summary": {
                "required_return_percent": 10.0,
                "feasibility_score": 50,
                "primary_recommendation": "Unable to analyze goal",
                "is_achievable": False
            }
        }

def calculate_retirement_goal(current_age: int, retirement_age: int,
                            desired_annual_income: float,
                            current_savings: float = 0,
                            monthly_contribution: float = 0,
                            risk_score: float = 50) -> Dict[str, any]:
    """
    Specialized retirement goal analysis
    
    Calculates retirement needs using the 4% withdrawal rule and provides
    comprehensive analysis of retirement feasibility.
    
    Args:
        current_age: Current age
        retirement_age: Planned retirement age
        desired_annual_income: Desired retirement income
        current_savings: Current retirement savings
        monthly_contribution: Monthly retirement contributions
        risk_score: Risk tolerance
        
    Returns:
        Dict with retirement-specific analysis
    """
    try:
        calculator = GoalCalculator()
        
        # Calculate retirement needs
        retirement_analysis = calculator.calculate_retirement_needs(
            current_age, retirement_age, desired_annual_income,
            current_savings, monthly_contribution
        )
        
        # Assess feasibility of retirement plan
        assessor = FeasibilityAssessor()
        feasibility = assessor.assess_goal_feasibility(
            required_return=retirement_analysis["required_annual_return"],
            risk_score=risk_score,
            timeframe=retirement_analysis["years_to_retirement"],
            current_investment=current_savings,
            monthly_contribution=monthly_contribution
        )
        
        return {
            "retirement_calculation": retirement_analysis,
            "feasibility_assessment": feasibility,
            "retirement_summary": {
                "fund_needed": retirement_analysis["inflation_adjusted_fund"],
                "years_to_save": retirement_analysis["years_to_retirement"],
                "required_return": retirement_analysis["required_return_percent"],
                "feasibility_score": feasibility["feasibility_score"],
                "is_on_track": feasibility["feasibility_score"] >= 60
            }
        }
        
    except Exception as e:
        return {"error": f"Retirement analysis failed: {str(e)}"}

def compare_goal_scenarios(base_target: float, current_investment: float,
                          base_timeframe: int, monthly_contribution: float,
                          risk_score: float) -> Dict[str, any]:
    """
    Compare multiple goal scenarios for decision making
    
    Generates and compares different goal scenarios to help users
    understand the impact of various adjustments.
    
    Returns:
        Dict with scenario comparisons and recommendations
    """
    try:
        calculator = GoalCalculator()
        
        # Generate multiple scenarios
        scenarios = calculator.create_goal_scenarios(
            base_target, current_investment, base_timeframe, monthly_contribution
        )
        
        # Assess feasibility for each scenario
        assessor = FeasibilityAssessor()
        scenario_assessments = {}
        
        for scenario_name, scenario_data in scenarios.items():
            assessment = assessor.assess_goal_feasibility(
                required_return=scenario_data["required_return_percent"] / 100,
                risk_score=risk_score,
                timeframe=base_timeframe
            )
            
            scenario_assessments[scenario_name] = {
                "scenario_data": scenario_data,
                "feasibility": assessment["feasibility_score"],
                "recommendation": assessment["recommendations"]["primary"]
            }
        
        # Find best scenario
        best_scenario = max(scenario_assessments.items(), 
                           key=lambda x: x[1]["feasibility"])
        
        return {
            "scenarios": scenario_assessments,
            "best_scenario": {
                "name": best_scenario[0],
                "data": best_scenario[1]
            },
            "comparison_summary": f"Best option: {best_scenario[0]} scenario with {best_scenario[1]['feasibility']:.0f}% feasibility"
        }
        
    except Exception as e:
        return {"error": f"Scenario comparison failed: {str(e)}"}

# Import required dependencies for type hints
from typing import Dict, Any