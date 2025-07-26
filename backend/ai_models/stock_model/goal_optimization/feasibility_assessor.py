"""
Goal Feasibility Assessment Module

This module provides honest, realistic assessment of whether users' financial
goals are achievable given their risk tolerance and market realities. It helps
prevent unrealistic expectations and provides actionable recommendations.

Key Features:
1. Reality Check Analysis - Compare goals against market realities
2. Risk-Return Alignment - Match required returns with risk tolerance
3. Adjustment Recommendations - Specific suggestions for improvement
4. Probability Analysis - Monte Carlo simulations for success probability
5. Alternative Scenarios - Modified goals that are more achievable
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import scipy.stats as stats

logger = logging.getLogger(__name__)


class FeasibilityAssessor:
    """
    Goal Feasibility Assessment System
    
    This class provides realistic assessment of whether users' financial goals
    are achievable. It compares what return the user NEEDS vs what return their
    risk tolerance can typically PROVIDE, then gives honest feedback and suggestions.
    
    Core Innovation: Many financial apps let users set unrealistic goals
    (like wanting 15% returns with conservative risk). This system provides
    honest reality checks and specific suggestions for improvement.
    """
    
    def __init__(self):
        """Initialize the feasibility assessor with market assumptions"""
        
        # Historical market return assumptions by asset class
        # Based on long-term historical data and academic research
        self.asset_class_returns = {
            "cash": {"return": 0.02, "volatility": 0.01},           # 2% return, 1% volatility
            "bonds": {"return": 0.05, "volatility": 0.08},          # 5% return, 8% volatility
            "balanced": {"return": 0.08, "volatility": 0.12},       # 8% return, 12% volatility
            "stocks": {"return": 0.10, "volatility": 0.16},         # 10% return, 16% volatility
            "growth": {"return": 0.12, "volatility": 0.20},         # 12% return, 20% volatility
            "aggressive": {"return": 0.15, "volatility": 0.25}      # 15% return, 25% volatility
        }
        
        # Risk tolerance to asset class mapping
        # Maps user risk scores to expected portfolio characteristics
        self.risk_to_portfolio = {
            "ultra_conservative": {"primary_class": "bonds", "expected_return": 0.05, "volatility": 0.08},
            "conservative": {"primary_class": "balanced", "expected_return": 0.07, "volatility": 0.10},
            "moderate": {"primary_class": "balanced", "expected_return": 0.09, "volatility": 0.12},
            "moderate_aggressive": {"primary_class": "stocks", "expected_return": 0.11, "volatility": 0.15},
            "aggressive": {"primary_class": "growth", "expected_return": 0.13, "volatility": 0.18},
            "ultra_aggressive": {"primary_class": "aggressive", "expected_return": 0.16, "volatility": 0.22}
        }
        
        # Success probability thresholds
        self.probability_thresholds = {
            "very_high": 0.85,      # 85%+ chance of success
            "high": 0.70,           # 70-85% chance
            "moderate": 0.50,       # 50-70% chance
            "low": 0.30,            # 30-50% chance
            "very_low": 0.0         # <30% chance
        }
    
    def assess_goal_feasibility(self, required_return: float, risk_score: float,
                              timeframe: int, current_investment: float = 0,
                              monthly_contribution: float = 0) -> Dict[str, Any]:
        """
        Comprehensive goal feasibility assessment
        
        This is the core function that provides honest feedback about whether
        a user's goal is realistic given their risk tolerance and the realities
        of financial markets.
        
        Args:
            required_return: Annual return needed to reach goal (from GoalCalculator)
            risk_score: User's risk tolerance (0-100 scale)
            timeframe: Investment horizon in years
            current_investment: Starting amount
            monthly_contribution: Regular monthly investments
            
        Returns:
            Dict containing:
            - feasibility_score: 0-100% likelihood of achieving goal
            - risk_alignment: How well goal matches risk tolerance
            - expected_return: What user can typically achieve
            - return_gap: Difference between need and realistic expectation
            - recommendation: Specific advice for improvement
            - alternative_scenarios: Modified goals that are more achievable
            - probability_analysis: Monte Carlo simulation results
        """
        try:
            logger.info(f"Assessing goal feasibility: {required_return:.1%} return needed, risk score {risk_score}")
            
            # Convert risk score to risk category and get portfolio expectations
            risk_category = self._risk_score_to_category(risk_score)
            portfolio_profile = self.risk_to_portfolio[risk_category]
            
            expected_return = portfolio_profile["expected_return"]
            expected_volatility = portfolio_profile["volatility"]
            
            # Calculate return gap (how much more return is needed than expected)
            return_gap = required_return - expected_return
            
            # Calculate basic feasibility score
            feasibility_score = self._calculate_feasibility_score(
                required_return, expected_return, timeframe
            )
            
            # Perform Monte Carlo probability analysis
            probability_analysis = self._monte_carlo_analysis(
                expected_return, expected_volatility, required_return, 
                timeframe, current_investment, monthly_contribution
            )
            
            # Generate specific recommendations
            recommendations = self._generate_recommendations(
                return_gap, feasibility_score, timeframe, risk_score
            )
            
            # Create alternative scenarios
            alternative_scenarios = self._create_alternative_scenarios(
                required_return, expected_return, timeframe
            )
            
            # Assess risk alignment
            risk_alignment = self._assess_risk_alignment(required_return, risk_score)
            
            assessment = {
                "feasibility_score": feasibility_score,
                "risk_alignment": risk_alignment,
                "expected_return": expected_return,
                "required_return": required_return,
                "return_gap": return_gap,
                "return_gap_percent": return_gap * 100,
                "risk_category": risk_category,
                "expected_volatility": expected_volatility,
                "probability_analysis": probability_analysis,
                "recommendations": recommendations,
                "alternative_scenarios": alternative_scenarios,
                "timeframe_impact": self._analyze_timeframe_impact(return_gap, timeframe)
            }
            
            logger.info(f"💡 Feasibility assessment: {feasibility_score:.0f}% - {recommendations['primary']}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in feasibility assessment: {e}")
            return self._default_assessment(required_return, risk_score)
    
    def _calculate_feasibility_score(self, required_return: float, 
                                   expected_return: float, timeframe: int) -> float:
        """
        Calculate feasibility score based on return gap and timeframe
        
        The score represents the likelihood of achieving the goal given
        historical market performance and the user's risk tolerance.
        """
        return_gap = required_return - expected_return
        
        # Base score calculation
        if return_gap <= 0:
            # Goal is achievable - user needs less return than their risk tolerance provides
            base_score = min(100, 90 + abs(return_gap) * 200)
        else:
            # Goal is challenging - user needs more return than typical for their risk level
            # Penalty increases exponentially with larger gaps
            base_score = max(10, 80 - (return_gap * 300))
        
        # Timeframe adjustments
        if timeframe >= 15:
            # Long timeframe helps - more time for compounding and recovery
            timeframe_bonus = 10
        elif timeframe >= 7:
            # Medium timeframe - some benefit
            timeframe_bonus = 5
        elif timeframe <= 3:
            # Short timeframe penalty - less time for markets to work
            timeframe_bonus = -15
        else:
            timeframe_bonus = 0
        
        # Market volatility adjustment
        # Longer timeframes can better handle market volatility
        volatility_adjustment = min(5, timeframe * 0.5) if return_gap > 0.03 else 0
        
        final_score = base_score + timeframe_bonus + volatility_adjustment
        return max(0, min(100, final_score))
    
    def _monte_carlo_analysis(self, expected_return: float, volatility: float,
                            required_return: float, timeframe: int,
                            current_investment: float, monthly_contribution: float,
                            num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Monte Carlo simulation to estimate probability of success
        
        Runs thousands of simulated market scenarios to estimate the
        probability of achieving the goal given market uncertainty.
        """
        try:
            logger.debug(f"Running Monte Carlo analysis with {num_simulations} simulations")
            
            successes = 0
            final_values = []
            
            # Target value calculation
            total_contributions = current_investment + (monthly_contribution * 12 * timeframe)
            target_value = total_contributions * ((1 + required_return) ** timeframe)
            
            for _ in range(num_simulations):
                # Simulate random annual returns based on normal distribution
                portfolio_value = current_investment
                
                for year in range(timeframe):
                    # Generate random annual return
                    annual_return = np.random.normal(expected_return, volatility)
                    annual_return = max(-0.50, min(1.00, annual_return))  # Bound returns
                    
                    # Apply return to current portfolio
                    portfolio_value *= (1 + annual_return)
                    
                    # Add monthly contributions throughout the year
                    for month in range(12):
                        portfolio_value += monthly_contribution
                        if monthly_contribution > 0:
                            # Apply monthly return to new contribution
                            monthly_return = annual_return / 12
                            remaining_months = 12 - month
                            contribution_growth = monthly_contribution * (
                                (1 + monthly_return) ** remaining_months - 1
                            ) / monthly_return if monthly_return != 0 else 0
                            portfolio_value += contribution_growth
                
                final_values.append(portfolio_value)
                
                # Check if this simulation achieved the goal
                if portfolio_value >= target_value:
                    successes += 1
            
            success_probability = successes / num_simulations
            
            # Calculate percentile outcomes
            final_values = np.array(final_values)
            percentiles = np.percentile(final_values, [10, 25, 50, 75, 90])
            
            return {
                "success_probability": success_probability,
                "num_simulations": num_simulations,
                "target_value": target_value,
                "median_outcome": percentiles[2],
                "pessimistic_outcome": percentiles[0],  # 10th percentile
                "optimistic_outcome": percentiles[4],   # 90th percentile
                "expected_shortfall": max(0, target_value - percentiles[2]),
                "probability_category": self._categorize_probability(success_probability)
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo analysis failed: {e}")
            return {
                "success_probability": 0.5,
                "num_simulations": 0,
                "target_value": 0,
                "median_outcome": 0,
                "probability_category": "uncertain",
                "error": str(e)
            }
    
    def _generate_recommendations(self, return_gap: float, feasibility_score: float,
                                timeframe: int, risk_score: float) -> Dict[str, str]:
        """
        Generate specific, actionable recommendations for improving goal feasibility
        
        Provides concrete steps users can take to make their goals more achievable.
        """
        recommendations = {
            "primary": "",
            "secondary": [],
            "specific_actions": []
        }
        
        if feasibility_score >= 80:
            # Highly feasible goal
            recommendations["primary"] = "Your goal is highly achievable with your current plan"
            recommendations["secondary"] = [
                "Consider diversifying across asset classes",
                "Review and rebalance portfolio annually",
                "Stay consistent with your investment plan"
            ]
        
        elif feasibility_score >= 60:
            # Moderately feasible
            recommendations["primary"] = "Your goal is achievable but may require market outperformance"
            recommendations["secondary"] = [
                "Consider slightly increasing risk tolerance if appropriate",
                "Look for ways to increase monthly contributions",
                "Monitor progress and adjust strategy as needed"
            ]
        
        elif feasibility_score >= 40:
            # Challenging goal
            recommendations["primary"] = "Your goal is challenging - consider making adjustments"
            
            if return_gap > 0.05:  # Need 5%+ more return
                recommendations["secondary"].append("Required return is significantly higher than typical for your risk level")
            
            if timeframe < 5:
                recommendations["secondary"].append("Consider extending your timeframe to allow more time for growth")
            
            recommendations["secondary"].extend([
                "Increase monthly contributions if possible",
                "Consider slightly higher risk tolerance",
                "Break goal into smaller, interim targets"
            ])
        
        else:
            # Very challenging goal
            recommendations["primary"] = "Your goal requires significant changes to be achievable"
            recommendations["secondary"] = [
                "Consider reducing target amount or extending timeframe",
                "Substantially increase monthly contributions",
                "Consider higher risk tolerance if appropriate for your situation",
                "Break into multiple smaller goals over time"
            ]
        
        # Add specific numerical recommendations
        if return_gap > 0.03:  # Need 3%+ more return
            risk_increase_needed = self._calculate_risk_increase_needed(return_gap, risk_score)
            if risk_increase_needed:
                recommendations["specific_actions"].append(
                    f"Consider increasing risk tolerance to {risk_increase_needed} level"
                )
        
        if timeframe < 10 and feasibility_score < 60:
            recommended_timeframe = self._calculate_better_timeframe(return_gap, timeframe)
            recommendations["specific_actions"].append(
                f"Extending timeframe to {recommended_timeframe} years would improve feasibility"
            )
        
        return recommendations
    
    def _create_alternative_scenarios(self, required_return: float, 
                                    expected_return: float, timeframe: int) -> Dict[str, Dict]:
        """
        Create alternative goal scenarios that are more achievable
        
        Shows users what goals would be more realistic given their constraints.
        """
        scenarios = {}
        
        # Scenario 1: Extend timeframe
        if timeframe < 20:
            extended_timeframe = min(30, timeframe + 5)
            scenarios["extended_time"] = {
                "description": f"Extend timeframe to {extended_timeframe} years",
                "new_timeframe": extended_timeframe,
                "feasibility_improvement": "Significantly improved",
                "trade_off": "Longer wait time for goal achievement"
            }
        
        # Scenario 2: Reduce target (if required return is much higher than expected)
        if required_return > expected_return + 0.03:
            reduction_factor = expected_return / required_return
            scenarios["reduced_target"] = {
                "description": f"Reduce target by {(1-reduction_factor)*100:.0f}%",
                "target_multiplier": reduction_factor,
                "feasibility_improvement": "Highly improved",
                "trade_off": "Lower target amount"
            }
        
        # Scenario 3: Increase contributions (if reasonable)
        scenarios["increased_contributions"] = {
            "description": "Increase monthly contributions by 50%",
            "contribution_multiplier": 1.5,
            "feasibility_improvement": "Moderately improved",
            "trade_off": "Higher monthly financial commitment"
        }
        
        # Scenario 4: Balanced approach
        scenarios["balanced_adjustment"] = {
            "description": "Modest increases in time, contributions, and risk",
            "timeframe_increase": 2,
            "contribution_increase": 0.25,
            "risk_increase": "one level",
            "feasibility_improvement": "Significantly improved",
            "trade_off": "Multiple small adjustments needed"
        }
        
        return scenarios
    
    def _assess_risk_alignment(self, required_return: float, risk_score: float) -> Dict[str, Any]:
        """
        Assess how well the required return aligns with the user's risk tolerance
        """
        risk_category = self._risk_score_to_category(risk_score)
        expected_return = self.risk_to_portfolio[risk_category]["expected_return"]
        
        return_gap = required_return - expected_return
        
        if abs(return_gap) <= 0.02:  # Within 2%
            alignment = "excellent"
            message = "Required return matches well with your risk tolerance"
        elif return_gap <= 0.05:  # Need up to 5% more
            alignment = "good"
            message = "Required return is slightly higher than typical for your risk level"
        elif return_gap <= 0.08:  # Need up to 8% more
            alignment = "moderate"
            message = "Required return is significantly higher than your risk tolerance suggests"
        else:
            alignment = "poor"
            message = "Required return is much higher than realistic for your risk tolerance"
        
        return {
            "alignment_rating": alignment,
            "message": message,
            "return_gap": return_gap,
            "risk_category": risk_category,
            "expected_return": expected_return
        }
    
    def _analyze_timeframe_impact(self, return_gap: float, timeframe: int) -> Dict[str, Any]:
        """
        Analyze how timeframe affects goal feasibility
        """
        impact_analysis = {
            "current_timeframe": timeframe,
            "timeframe_rating": "",
            "improvement_potential": "",
            "recommended_timeframe": timeframe
        }
        
        if timeframe >= 15:
            impact_analysis["timeframe_rating"] = "excellent"
            impact_analysis["improvement_potential"] = "Minimal - already have long-term advantage"
        elif timeframe >= 10:
            impact_analysis["timeframe_rating"] = "good"
            impact_analysis["improvement_potential"] = "Some benefit from extending further"
            impact_analysis["recommended_timeframe"] = timeframe + 3
        elif timeframe >= 5:
            impact_analysis["timeframe_rating"] = "moderate"
            impact_analysis["improvement_potential"] = "Significant benefit from extending timeframe"
            impact_analysis["recommended_timeframe"] = timeframe + 5
        else:
            impact_analysis["timeframe_rating"] = "challenging"
            impact_analysis["improvement_potential"] = "Major benefit from extending timeframe"
            impact_analysis["recommended_timeframe"] = max(10, timeframe * 2)
        
        # Calculate benefit of extending timeframe
        if return_gap > 0.03:  # If we need significantly more return
            years_needed = self._calculate_years_for_feasibility(return_gap)
            if years_needed > timeframe:
                impact_analysis["years_for_feasibility"] = years_needed
                impact_analysis["extension_benefit"] = f"Extending to {years_needed} years would make goal highly feasible"
        
        return impact_analysis
    
    def create_improvement_plan(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a specific improvement plan based on feasibility assessment
        
        Provides a step-by-step plan for making goals more achievable.
        """
        try:
            feasibility_score = assessment.get("feasibility_score", 50)
            return_gap = assessment.get("return_gap", 0)
            
            plan = {
                "current_status": self._get_status_message(feasibility_score),
                "priority_actions": [],
                "optional_actions": [],
                "timeline": "Implement over next 3-6 months",
                "expected_improvement": ""
            }
            
            # Priority actions based on feasibility score
            if feasibility_score < 40:
                plan["priority_actions"] = [
                    "Re-evaluate goal amount - consider reducing by 20-30%",
                    "Extend timeframe by at least 3-5 years if possible",
                    "Increase monthly contributions significantly",
                    "Consider increasing risk tolerance if appropriate"
                ]
                plan["expected_improvement"] = "These changes could improve feasibility by 30-50 percentage points"
            
            elif feasibility_score < 60:
                plan["priority_actions"] = [
                    "Increase monthly contributions by 25-50%",
                    "Consider extending timeframe by 2-3 years",
                    "Review risk tolerance - slight increase may help"
                ]
                plan["expected_improvement"] = "These changes could improve feasibility by 15-25 percentage points"
            
            else:
                plan["priority_actions"] = [
                    "Continue current strategy - it's working well",
                    "Review progress annually and adjust as needed",
                    "Consider tax-advantaged accounts if not already using"
                ]
                plan["expected_improvement"] = "Minor optimizations could add 5-10 percentage points"
            
            # Optional actions for all scenarios
            plan["optional_actions"] = [
                "Consider automatic investment increases (annual raises)",
                "Explore tax-efficient investment strategies",
                "Set up automatic rebalancing",
                "Plan for periodic strategy reviews"
            ]
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating improvement plan: {e}")
            return {"error": str(e)}
    
    def _risk_score_to_category(self, risk_score: float) -> str:
        """Convert numerical risk score to category"""
        if risk_score < 15:
            return "ultra_conservative"
        elif risk_score < 30:
            return "conservative"
        elif risk_score < 50:
            return "moderate"
        elif risk_score < 70:
            return "moderate_aggressive"
        elif risk_score < 85:
            return "aggressive"
        else:
            return "ultra_aggressive"
    
    def _categorize_probability(self, probability: float) -> str:
        """Categorize success probability"""
        if probability >= 0.85:
            return "very_high"
        elif probability >= 0.70:
            return "high"
        elif probability >= 0.50:
            return "moderate"
        elif probability >= 0.30:
            return "low"
        else:
            return "very_low"
    
    def _calculate_risk_increase_needed(self, return_gap: float, current_risk_score: float) -> Optional[str]:
        """Calculate what risk level would be needed to close return gap"""
        for category, profile in self.risk_to_portfolio.items():
            if profile["expected_return"] >= return_gap + 0.02:  # 2% buffer
                return category
        return None
    
    def _calculate_better_timeframe(self, return_gap: float, current_timeframe: int) -> int:
        """Calculate timeframe that would make goal more feasible"""
        if return_gap <= 0.02:
            return current_timeframe + 1
        elif return_gap <= 0.05:
            return current_timeframe + 3
        else:
            return current_timeframe + 5
    
    def _calculate_years_for_feasibility(self, return_gap: float) -> int:
        """Calculate years needed to make goal highly feasible"""
        # Rough heuristic: each additional year reduces required return
        additional_years = max(3, int(return_gap * 20))
        return additional_years
    
    def _get_status_message(self, feasibility_score: float) -> str:
        """Get status message based on feasibility score"""
        if feasibility_score >= 80:
            return "Goal is highly achievable with current plan"
        elif feasibility_score >= 60:
            return "Goal is achievable but may need minor adjustments"
        elif feasibility_score >= 40:
            return "Goal is challenging and would benefit from modifications"
        else:
            return "Goal requires significant changes to be realistic"
    
    def _default_assessment(self, required_return: float, risk_score: float) -> Dict[str, Any]:
        """Return default assessment when calculation fails"""
        return {
            "feasibility_score": 50,
            "risk_alignment": {"alignment_rating": "unknown", "message": "Unable to assess"},
            "expected_return": 0.08,
            "required_return": required_return,
            "return_gap": required_return - 0.08,
            "recommendations": {"primary": "Unable to generate recommendations due to calculation error"},
            "alternative_scenarios": {},
            "error": "Assessment calculation failed"
        }
                    #