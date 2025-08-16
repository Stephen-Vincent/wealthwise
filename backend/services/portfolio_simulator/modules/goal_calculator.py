"""
Smart Goal Calculator Module

This module handles the intelligent calculation of required returns for investment goals.
It fixes the 0.0% required return issue by properly handling cases where contributions
alone would reach the target.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SmartGoalCalculator:
    """
    Smart goal calculator that handles edge cases and provides realistic return targets.
    
    Key features:
    - Detects when contributions alone reach target
    - Sets minimum growth rate to beat inflation
    - Provides user-friendly messaging
    - Uses proper compound interest calculations
    """
    
    def __init__(self, minimum_return_percent: float = 4.0):
        """
        Initialize the smart goal calculator.
        
        Args:
            minimum_return_percent: Minimum return target when contributions alone suffice
        """
        self.minimum_return_percent = minimum_return_percent
        logger.info(f"🎯 SmartGoalCalculator initialized with {minimum_return_percent}% minimum return")
    
    def calculate_smart_required_return(self, target_value: float, current_investment: float, 
                                      timeframe: int, monthly_contribution: float) -> Dict[str, Any]:
        """
        Calculate required return with smart handling for when contributions alone reach the target.
        
        This fixes the 0.0% required return issue by setting a minimum growth rate
        when monthly contributions alone would reach the target.
        
        Args:
            target_value: Target investment value
            current_investment: Initial lump sum investment
            timeframe: Investment timeframe in years
            monthly_contribution: Monthly contribution amount
            
        Returns:
            Dict containing required return analysis and user messaging
        """
        
        try:
            # Calculate total contributions over the timeframe
            total_contributions = current_investment + (monthly_contribution * 12 * timeframe)
            
            logger.info(f"💰 Goal analysis: Target £{target_value:,.2f}, Contributions £{total_contributions:,.2f}")
            
            if total_contributions >= target_value:
                # Contributions alone will reach target - set minimum growth rate
                logger.info(f"✅ Contributions alone reach target! Setting minimum return: {self.minimum_return_percent}%")
                
                return self._create_contributions_sufficient_result(
                    target_value, total_contributions, self.minimum_return_percent
                )
            
            else:
                # Need growth to reach target - calculate required return
                return self._calculate_required_growth_return(
                    target_value, current_investment, timeframe, monthly_contribution
                )
                
        except Exception as e:
            logger.error(f"❌ Error calculating smart required return: {e}")
            return self._create_fallback_result(e)
    
    def _create_contributions_sufficient_result(self, target_value: float, 
                                              total_contributions: float, 
                                              minimum_return: float) -> Dict[str, Any]:
        """Create result when contributions alone reach the target."""
        
        excess_amount = total_contributions - target_value
        
        return {
            "required_return": minimum_return / 100,  # Convert to decimal
            "required_return_percent": minimum_return,
            "can_reach_with_contributions": True,
            "contributions_total": total_contributions,
            "excess_amount": excess_amount,
            "message": f"Good news! Your contributions alone will reach your goal. We're targeting {minimum_return}% growth to beat inflation and give you extra security.",
            "feasibility_rating": 5.0,  # High feasibility since contributions alone work
            "calculation_type": "contributions_sufficient"
        }
    
    def _calculate_required_growth_return(self, target_value: float, current_investment: float,
                                        timeframe: int, monthly_contribution: float) -> Dict[str, Any]:
        """Calculate required return when growth is needed to reach target."""
        
        # Using compound interest formula: FV = PV(1+r)^t + PMT[((1+r)^t - 1)/r]
        years = timeframe
        pv = current_investment
        pmt_annual = monthly_contribution * 12
        fv = target_value
        
        # Solve for r using iterative method (binary search)
        required_return = self._solve_for_required_return(pv, pmt_annual, fv, years)
        required_return_percent = required_return * 100
        
        logger.info(f"📊 Calculated required return: {required_return_percent:.1f}%")
        
        # Assess feasibility
        feasibility_info = self._assess_feasibility(required_return_percent)
        
        return {
            "required_return": required_return,  # Decimal format
            "required_return_percent": required_return_percent,
            "can_reach_with_contributions": False,
            "feasibility_rating": feasibility_info["rating"],
            "message": feasibility_info["message"],
            "calculation_type": "growth_required",
            "annual_contribution": pmt_annual,
            "total_needed_growth": target_value - (pv + pmt_annual * years)
        }
    
    def _solve_for_required_return(self, pv: float, pmt_annual: float, 
                                  fv: float, years: int) -> float:
        """
        Solve for required return using binary search method.
        
        This implements the compound interest formula solving for the interest rate.
        """
        
        low_rate = 0.001  # 0.1%
        high_rate = 0.30   # 30%
        tolerance = 0.0001
        max_iterations = 100
        
        for iteration in range(max_iterations):
            mid_rate = (low_rate + high_rate) / 2
            
            # Calculate future value with this rate
            fv_calculated = pv * ((1 + mid_rate) ** years)
            
            if pmt_annual > 0 and mid_rate > 0:
                # Add future value of annuity
                fv_calculated += pmt_annual * (((1 + mid_rate) ** years - 1) / mid_rate)
            elif pmt_annual > 0:
                # Handle zero interest rate case
                fv_calculated += pmt_annual * years
            
            # Check convergence
            if abs(fv_calculated - fv) < tolerance:
                logger.info(f"📈 Converged after {iteration + 1} iterations")
                break
                
            # Adjust search bounds
            if fv_calculated < fv:
                low_rate = mid_rate
            else:
                high_rate = mid_rate
        
        return mid_rate
    
    def _assess_feasibility(self, required_return_percent: float) -> Dict[str, Any]:
        """Assess the feasibility of achieving the required return."""
        
        if required_return_percent > 15:
            return {
                "rating": 2.0,
                "message": f"Challenging: You need {required_return_percent:.1f}% annual growth. Consider increasing contributions or extending timeframe.",
                "risk_level": "Very High"
            }
        elif required_return_percent > 10:
            return {
                "rating": 3.0,
                "message": f"Ambitious: You need {required_return_percent:.1f}% annual growth. This requires growth-focused investments.",
                "risk_level": "High"
            }
        elif required_return_percent > 7:
            return {
                "rating": 4.0,
                "message": f"Achievable: You need {required_return_percent:.1f}% annual growth. A balanced approach should work.",
                "risk_level": "Moderate"
            }
        else:
            return {
                "rating": 5.0,
                "message": f"Very achievable: You need {required_return_percent:.1f}% annual growth. Conservative investments may suffice.",
                "risk_level": "Low"
            }
    
    def _create_fallback_result(self, error: Exception) -> Dict[str, Any]:
        """Create fallback result when calculation fails."""
        
        default_return = 6.0  # 6% default
        
        return {
            "required_return": default_return / 100,
            "required_return_percent": default_return,
            "can_reach_with_contributions": False,
            "feasibility_rating": 3.0,
            "message": f"Using default {default_return}% growth target due to calculation error.",
            "calculation_type": "fallback_error",
            "error": str(error)
        }
    
    def get_goal_insights(self, goal_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate educational insights about the goal analysis.
        
        Args:
            goal_analysis: Result from calculate_smart_required_return
            
        Returns:
            List of educational insights
        """
        
        insights = []
        
        calculation_type = goal_analysis.get("calculation_type", "unknown")
        
        if calculation_type == "contributions_sufficient":
            insights.extend([
                "🎉 Your disciplined saving approach means you're already on track to reach your goal!",
                f"💡 The {goal_analysis['required_return_percent']:.1f}% growth target will help you beat inflation and provide extra security.",
                "📈 Even modest investment growth can significantly boost your final outcome.",
                f"🎯 You'll have an extra £{goal_analysis['excess_amount']:,.0f} cushion above your target."
            ])
        
        elif calculation_type == "growth_required":
            required_return = goal_analysis["required_return_percent"]
            feasibility_rating = goal_analysis["feasibility_rating"]
            
            if feasibility_rating >= 4.0:
                insights.extend([
                    f"✅ Your {required_return:.1f}% return target is very achievable with proper portfolio allocation.",
                    "📊 Historical market returns suggest this is realistic for long-term investors.",
                    "💪 Stay consistent with your contributions and time will work in your favor."
                ])
            elif feasibility_rating >= 3.0:
                insights.extend([
                    f"⚖️ Your {required_return:.1f}% return target requires a growth-focused strategy.",
                    "📈 Consider higher allocation to stocks and growth investments.",
                    "⏳ Market volatility is normal - staying invested long-term is key."
                ])
            else:
                insights.extend([
                    f"🎯 Your {required_return:.1f}% return target is ambitious but not impossible.",
                    "💡 Consider increasing monthly contributions or extending your timeframe.",
                    "🔄 Breaking your goal into smaller milestones can help track progress."
                ])
        
        # General insights
        insights.extend([
            "📚 Remember: Past performance doesn't guarantee future results, but diversification helps manage risk.",
            "⏰ Starting early and staying consistent are the most powerful wealth-building strategies."
        ])
        
        return insights
    
    def compare_scenarios(self, base_scenario: Dict[str, Any], 
                         alternative_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare different goal scenarios to help users understand their options.
        
        Args:
            base_scenario: The user's current goal parameters
            alternative_scenarios: List of alternative parameter sets to compare
            
        Returns:
            Comparison analysis with recommendations
        """
        
        comparisons = []
        
        for i, scenario in enumerate(alternative_scenarios):
            alt_analysis = self.calculate_smart_required_return(
                scenario.get("target_value", base_scenario.get("target_value", 50000)),
                scenario.get("current_investment", base_scenario.get("current_investment", 0)),
                scenario.get("timeframe", base_scenario.get("timeframe", 10)),
                scenario.get("monthly_contribution", base_scenario.get("monthly_contribution", 0))
            )
            
            comparisons.append({
                "scenario_name": scenario.get("name", f"Alternative {i + 1}"),
                "parameters": scenario,
                "analysis": alt_analysis,
                "vs_base_return_diff": alt_analysis["required_return_percent"] - base_scenario.get("required_return_percent", 0)
            })
        
        # Find the most feasible scenario
        best_scenario = max(comparisons, key=lambda x: x["analysis"]["feasibility_rating"])
        
        return {
            "base_scenario": base_scenario,
            "comparisons": comparisons,
            "recommendation": {
                "best_scenario": best_scenario,
                "improvement_suggestions": self._generate_improvement_suggestions(comparisons)
            }
        }
    
    def _generate_improvement_suggestions(self, comparisons: List[Dict]) -> List[str]:
        """Generate suggestions for improving goal feasibility."""
        
        suggestions = []
        
        for comparison in comparisons:
            analysis = comparison["analysis"]
            if analysis["feasibility_rating"] > 4.0:
                suggestions.append(f"✅ {comparison['scenario_name']}: {analysis['message']}")
        
        if not suggestions:
            suggestions.extend([
                "💡 Consider increasing your monthly contributions by £50-100",
                "⏰ Extending your timeframe by 2-3 years can significantly improve feasibility",
                "🎯 Breaking large goals into smaller milestones makes them more achievable"
            ])
        
        return suggestions