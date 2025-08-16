"""
AI Summarizer Module

This module generates enhanced AI summaries with SHAP explanations,
crash analysis integration, and educational insights.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class AISummaryGenerator:
    """
    Generates enhanced AI summaries for portfolio simulations.
    
    Features:
    - SHAP-enhanced explanations
    - Crash analysis integration
    - Goal-oriented insights
    - Educational content generation
    - Fallback to simple summaries
    """
    
    def __init__(self):
        """Initialize the AI summary generator."""
        self.ai_service = None
        logger.info("🧠 AISummaryGenerator initialized")
    
    async def generate_enhanced_summary(self, stocks_picked: List[Dict], 
                                      user_data: Dict[str, Any], 
                                      risk_profile: Dict[str, Any],
                                      simulation_results: Dict[str, Any],
                                      goal_analysis: Optional[Dict] = None,
                                      recommendation_result: Optional[Dict] = None) -> str:
        """
        Generate enhanced AI summary with all available context.
        
        Args:
            stocks_picked: Selected stocks with allocations
            user_data: User investment parameters
            risk_profile: Risk assessment results
            simulation_results: Portfolio simulation results
            goal_analysis: Smart goal calculation results
            recommendation_result: Enhanced AI recommendation results
            
        Returns:
            Comprehensive AI-generated summary
        """
        
        try:
            logger.info("🧠 Generating enhanced AI summary")
            
            # Initialize AI service if not already done
            await self._initialize_ai_service()
            
            if self.ai_service:
                # Generate enhanced summary with all context
                enhanced_summary = await self._generate_shap_enhanced_summary(
                    stocks_picked, user_data, risk_profile, simulation_results,
                    goal_analysis, recommendation_result
                )
                
                if enhanced_summary:
                    return enhanced_summary
            
            # Fallback to basic summary
            logger.warning("⚠️ Enhanced AI summary not available, using basic summary")
            return await self.generate_basic_summary(
                stocks_picked, user_data, risk_profile, simulation_results
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced summary: {e}")
            return self._generate_simple_fallback_summary(
                stocks_picked, user_data, risk_profile, simulation_results
            )
    
    async def generate_basic_summary(self, stocks_picked: List[Dict], 
                                   user_data: Dict[str, Any], 
                                   risk_profile: Dict[str, Any],
                                   simulation_results: Dict[str, Any]) -> str:
        """
        Generate basic AI summary without enhanced features.
        """
        
        try:
            logger.info("🧠 Generating basic AI summary")
            
            # Initialize AI service if not already done
            await self._initialize_ai_service()
            
            if self.ai_service:
                basic_summary = await self.ai_service.generate_portfolio_summary(
                    stocks_picked=stocks_picked,
                    user_data=user_data,
                    risk_score=risk_profile["score"],
                    risk_label=risk_profile["label"],
                    simulation_results=simulation_results
                )
                
                if basic_summary:
                    return basic_summary
            
            # Fallback to simple summary
            return self._generate_simple_fallback_summary(
                stocks_picked, user_data, risk_profile, simulation_results
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Basic AI summary failed: {e}. Using simple summary.")
            return self._generate_simple_fallback_summary(
                stocks_picked, user_data, risk_profile, simulation_results
            )
    
    async def _initialize_ai_service(self):
        """Initialize the AI analysis service."""
        
        if self.ai_service is None:
            try:
                from services.ai_analysis import AIAnalysisService
                self.ai_service = AIAnalysisService()
                logger.info("✅ AI analysis service initialized")
            except ImportError:
                logger.warning("⚠️ AI analysis service not available")
                self.ai_service = False
    
    async def _generate_shap_enhanced_summary(self, stocks_picked: List[Dict],
                                            user_data: Dict[str, Any], 
                                            risk_profile: Dict[str, Any],
                                            simulation_results: Dict[str, Any],
                                            goal_analysis: Optional[Dict] = None,
                                            recommendation_result: Optional[Dict] = None) -> Optional[str]:
        """
        Generate enhanced summary with SHAP explanations and all context.
        """
        
        try:
            # Create comprehensive context for AI
            enhanced_context = {
                "user_data": user_data,
                "risk_profile": risk_profile,
                "simulation_results": simulation_results,
                "stocks_picked": stocks_picked,
                "goal_analysis": goal_analysis,
                "recommendation_result": recommendation_result
            }
            
            # Build detailed prompt for AI
            prompt = self._build_enhanced_prompt(enhanced_context)
            
            # Get AI response
            enhanced_summary = await self.ai_service._get_groq_response(prompt)
            
            # Add crash analysis section if available
            crash_analysis = simulation_results.get("market_crash_analysis")
            if crash_analysis and crash_analysis.get("crashes_detected", 0) > 0:
                crash_section = self._generate_crash_summary_section(crash_analysis)
                enhanced_summary += f"\n\n{crash_section}"
            
            # Add goal analysis section if available
            if goal_analysis:
                goal_section = self._generate_goal_summary_section(goal_analysis, user_data)
                enhanced_summary += f"\n\n{goal_section}"
            
            logger.info("✅ Enhanced AI summary generated successfully")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"❌ Error generating SHAP-enhanced summary: {e}")
            return None
    
    def _build_enhanced_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build comprehensive prompt for enhanced AI summary.
        """
        
        user_data = context["user_data"]
        risk_profile = context["risk_profile"]
        simulation_results = context["simulation_results"]
        stocks_picked = context["stocks_picked"]
        goal_analysis = context.get("goal_analysis")
        recommendation_result = context.get("recommendation_result")
        
        # Extract key metrics
        goal = user_data.get("goal", "wealth building")
        timeframe = user_data.get("timeframe", 10)
        start_value = simulation_results.get("starting_value", 0)
        end_value = simulation_results.get("end_value", 0)
        target_value = user_data.get("target_value", 50000)
        portfolio_return = simulation_results.get("portfolio_return", 0)
        risk_label = risk_profile.get("label", "Medium")
        
        # Stock allocation summary
        stock_list = ", ".join([f"{stock['symbol']} ({stock['allocation']*100:.0f}%)" 
                               for stock in stocks_picked])
        
        # Target achievement
        target_achieved = end_value >= target_value
        
        # Goal analysis context
        goal_context = ""
        if goal_analysis:
            required_return = goal_analysis.get("required_return_percent", 0)
            can_reach = goal_analysis.get("can_reach_with_contributions", False)
            feasibility = goal_analysis.get("feasibility_rating", 3)
            
            goal_context = f"""
GOAL ANALYSIS CONTEXT:
- Required return for goal: {required_return:.1f}% annually
- Can reach with contributions alone: {can_reach}
- Feasibility rating: {feasibility}/5.0
- Goal message: {goal_analysis.get('message', '')}
"""
        
        # SHAP explanation context
        shap_context = ""
        if recommendation_result and recommendation_result.get("shap_explanation"):
            shap_data = recommendation_result["shap_explanation"]
            if "human_readable_explanation" in shap_data:
                shap_context = f"""
AI REASONING (SHAP Analysis):
The AI considered these key factors when building your portfolio:
"""
                for factor, explanation in shap_data["human_readable_explanation"].items():
                    if explanation and len(explanation) > 10:
                        shap_context += f"• {explanation}\n"
        
        prompt = f"""
Create an educational and engaging portfolio simulation summary for a user with the following results:

USER PROFILE:
- Investment goal: {goal}
- Risk tolerance: {risk_label} ({risk_profile.get('score', 0)}/100)
- Investment timeframe: {timeframe} years
- Target amount: £{target_value:,.2f}

PORTFOLIO RESULTS:
- Total invested: £{start_value:,.2f}
- Final value: £{end_value:,.2f}
- Portfolio return: {portfolio_return:.1%}
- Target {'✅ ACHIEVED' if target_achieved else '📈 PARTIALLY ACHIEVED'}
- Stock allocation: {stock_list}

{goal_context}

{shap_context}

Please create a comprehensive summary that includes:

1. **Results Overview**: Celebrate their achievement or explain progress toward goal
2. **Portfolio Strategy**: Explain why these specific investments were chosen for their risk profile and goals
3. **Educational Insights**: Help them understand how diversification and time horizon work
4. **Market Reality**: Acknowledge that real investing involves volatility (crashes, recoveries)
5. **Next Steps**: Practical advice for implementing this strategy

Make it encouraging, educational, and personalized to their specific situation. Use a warm, professional tone that builds confidence in long-term investing. Include specific numbers and percentages to make it concrete.

Focus on helping them understand WHY this portfolio makes sense for their goals, not just WHAT the results are.
"""
        
        return prompt
    
    def _generate_crash_summary_section(self, crash_analysis: Dict[str, Any]) -> str:
        """
        Generate summary section about market crashes in the simulation.
        """
        
        crashes_detected = crash_analysis.get("crashes_detected", 0)
        overall_message = crash_analysis.get("overall_message", "")
        key_insights = crash_analysis.get("key_insights", [])
        
        section = f"""
## 📉 Market Volatility During Your Investment Period

{overall_message}

**Key Lessons from Market History:**
"""
        
        for insight in key_insights[:3]:  # Top 3 insights
            section += f"• {insight}\n"
        
        section += """
**Remember:** Market crashes are temporary setbacks, but long-term growth is the permanent trend. Every major crash in history has been followed by recovery and new market highs. Your simulation includes these real market events to show how staying invested through volatility leads to long-term success.
"""
        
        return section
    
    def _generate_goal_summary_section(self, goal_analysis: Dict[str, Any], 
                                     user_data: Dict[str, Any]) -> str:
        """
        Generate summary section about goal analysis.
        """
        
        can_reach = goal_analysis.get("can_reach_with_contributions", False)
        required_return = goal_analysis.get("required_return_percent", 0)
        feasibility_rating = goal_analysis.get("feasibility_rating", 3)
        message = goal_analysis.get("message", "")
        
        if can_reach:
            section = f"""
## 🎯 Excellent News About Your Goal!

{message}

Your disciplined saving approach means you're already on track! The portfolio growth we've targeted will:
• Help you beat inflation and preserve purchasing power
• Provide a financial cushion above your target
• Give you more flexibility and security in reaching your {user_data.get('goal', 'goal')}

This demonstrates the power of consistent saving combined with smart investing.
"""
        else:
            feasibility_text = {
                5: "Very achievable",
                4: "Quite achievable", 
                3: "Moderately challenging",
                2: "Ambitious but possible",
                1: "Very challenging"
            }.get(int(feasibility_rating), "Achievable")
            
            section = f"""
## 🎯 Your Goal Analysis

To reach your target, you need approximately {required_return:.1f}% annual returns. This is {feasibility_text.lower()} with the right investment strategy.

**Strategy Assessment:** {message}

**Why This Matters:** Understanding your required return helps ensure your portfolio is appropriately positioned. Our AI has recommended investments that align with this target while respecting your risk tolerance.
"""
        
        return section
    
    def _generate_simple_fallback_summary(self, stocks_picked: List[Dict], 
                                        user_data: Dict[str, Any], 
                                        risk_profile: Dict[str, Any],
                                        simulation_results: Dict[str, Any]) -> str:
        """
        Generate simple summary when AI services are unavailable.
        """
        
        goal = user_data.get("goal", "wealth building")
        timeframe = user_data.get("timeframe", 10)
        start_value = simulation_results.get("starting_value", 0)
        end_value = simulation_results.get("end_value", 0)
        target_value = user_data.get("target_value", 50000)
        portfolio_return = simulation_results.get("portfolio_return", 0)
        risk_label = risk_profile.get("label", "Medium")
        
        stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])
        target_achieved = end_value >= target_value
        
        return f"""
## 🎯 Your {goal.title()} Portfolio Results

Your {risk_label.lower()} risk portfolio, invested in {stock_list}, grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years.

**Performance Summary:**
• Portfolio return: {portfolio_return:.1%}
• Target of £{target_value:,.2f}: {'✅ Achieved' if target_achieved else '📈 Partially achieved'}
• Investment approach: Diversified across multiple asset classes

**Key Insights:**
• Diversification helps manage risk while pursuing growth
• Long-term investing allows you to ride out market volatility
• Consistent contributions amplify the power of compound growth
• Your {risk_label.lower()} risk profile balanced growth potential with stability

**Next Steps:**
Consider implementing this strategy through low-cost index funds or ETFs that track these asset classes. Remember that real investing involves market ups and downs, but staying disciplined with your plan is key to long-term success.

*This simulation demonstrates how patient, diversified investing can help you work toward your financial goals over time.*
"""
    
    async def create_shap_visualization(self, simulation_id: int, db) -> Optional[str]:
        """
        Generate SHAP visualization for a simulation.
        
        Args:
            simulation_id: ID of the simulation
            db: Database session
            
        Returns:
            Path to generated visualization or None
        """
        
        try:
            # Check if WealthWise visualization is available
            try:
                from ai_models.stock_model.explainable_ai import VisualizationEngine
                
                # Get simulation from database
                from database import models
                simulation = db.query(models.Simulation).filter(
                    models.Simulation.id == simulation_id
                ).first()
                
                if not simulation or not simulation.results.get("shap_explanation"):
                    logger.warning(f"⚠️ No SHAP data available for simulation {simulation_id}")
                    return None
                
                # Initialize visualization engine
                viz_engine = VisualizationEngine()
                
                # Create SHAP visualization
                save_path = f"./static/visualizations/shap_explanation_{simulation_id}.png"
                result = viz_engine.create_shap_waterfall_chart(
                    simulation.results["shap_explanation"], save_path
                )
                
                if "saved" in result:
                    logger.info(f"✅ SHAP visualization created: {save_path}")
                    return save_path
                else:
                    logger.warning(f"⚠️ SHAP visualization failed: {result}")
                    return None
                    
            except ImportError:
                logger.warning("⚠️ WealthWise visualization engine not available")
                return await self._create_simple_visualization(simulation_id, db)
                
        except Exception as e:
            logger.error(f"❌ Error creating SHAP visualization: {e}")
            return None
    
    async def _create_simple_visualization(self, simulation_id: int, db) -> Optional[str]:
        """
        Create a simple visualization when SHAP visualization is not available.
        """
        
        try:
            # This would create a basic chart showing portfolio allocation
            # You could use matplotlib or similar to create a simple pie chart
            logger.info(f"📊 Creating simple visualization for simulation {simulation_id}")
            
            # Get simulation data
            from database import models
            simulation = db.query(models.Simulation).filter(
                models.Simulation.id == simulation_id
            ).first()
            
            if not simulation:
                return None
            
            # Create basic portfolio allocation chart
            import matplotlib.pyplot as plt
            import os
            
            stocks_picked = simulation.results.get("stocks_picked", [])
            if not stocks_picked:
                return None
            
            # Extract data for chart
            labels = [stock.get("symbol", "Unknown") for stock in stocks_picked]
            sizes = [stock.get("allocation", 0) * 100 for stock in stocks_picked]
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Portfolio Allocation - {simulation.name}', fontsize=16, fontweight='bold')
            
            # Save chart
            os.makedirs("./static/visualizations", exist_ok=True)
            save_path = f"./static/visualizations/portfolio_allocation_{simulation_id}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Simple visualization created: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error creating simple visualization: {e}")
            return None
    
    def get_summary_insights(self, simulation_results: Dict[str, Any]) -> List[str]:
        """
        Extract key insights for summary generation.
        
        Args:
            simulation_results: Complete simulation results
            
        Returns:
            List of key insights
        """
        
        insights = []
        
        # Portfolio performance insights
        portfolio_return = simulation_results.get("portfolio_return", 0)
        if portfolio_return > 0.1:  # 10%+ return
            insights.append(f"Strong performance with {portfolio_return:.1%} average annual returns")
        elif portfolio_return > 0.05:  # 5-10% return
            insights.append(f"Solid performance with {portfolio_return:.1%} average annual returns")
        else:
            insights.append(f"Conservative growth with {portfolio_return:.1%} average annual returns")
        
        # Target achievement insights
        target_reached = simulation_results.get("target_reached", False)
        if target_reached:
            insights.append("Successfully reached your financial target through disciplined investing")
        else:
            end_value = simulation_results.get("end_value", 0)
            starting_value = simulation_results.get("starting_value", 1)
            growth_multiple = end_value / starting_value if starting_value > 0 else 1
            insights.append(f"Grew your investment by {growth_multiple:.1f}x through compound growth")
        
        # Crash analysis insights
        crash_analysis = simulation_results.get("market_crash_analysis")
        if crash_analysis and crash_analysis.get("crashes_detected", 0) > 0:
            crashes = crash_analysis["crashes_detected"]
            insights.append(f"Weathered {crashes} market crash(es) - demonstrating resilience of long-term investing")
        
        # Goal analysis insights
        goal_analysis = simulation_results.get("goal_analysis")
        if goal_analysis:
            if goal_analysis.get("can_reach_with_contributions"):
                insights.append("Your disciplined saving approach puts you on track to reach your goals")
            else:
                feasibility = goal_analysis.get("feasibility_rating", 3)
                if feasibility >= 4:
                    insights.append("Your financial goals are highly achievable with this strategy")
                elif feasibility >= 3:
                    insights.append("Your financial goals are realistic with proper planning")
        
        return insights
    
    async def generate_educational_content(self, topic: str, user_context: Dict[str, Any]) -> str:
        """
        Generate educational content about investing topics.
        
        Args:
            topic: Educational topic (e.g., "market_crashes", "diversification")
            user_context: User-specific context for personalization
            
        Returns:
            Educational content tailored to the user
        """
        
        try:
            await self._initialize_ai_service()
            
            if not self.ai_service:
                return self._get_fallback_educational_content(topic)
            
            prompt = self._build_educational_prompt(topic, user_context)
            educational_content = await self.ai_service._get_groq_response(prompt)
            
            return educational_content
            
        except Exception as e:
            logger.error(f"❌ Error generating educational content: {e}")
            return self._get_fallback_educational_content(topic)
    
    def _build_educational_prompt(self, topic: str, user_context: Dict[str, Any]) -> str:
        """Build prompt for educational content generation."""
        
        risk_profile = user_context.get("risk_profile", {})
        timeframe = user_context.get("timeframe", 10)
        experience = user_context.get("experience", 0)
        
        base_prompt = f"""
Create educational content about {topic} for an investor with:
- Risk tolerance: {risk_profile.get('label', 'Medium')}
- Investment timeframe: {timeframe} years
- Experience level: {experience} years

Make it practical, encouraging, and specific to their situation.
Include concrete examples and actionable insights.
Keep it beginner-friendly but not condescending.
"""
        
        topic_specific = {
            "market_crashes": "Focus on how crashes are normal, recoveries happen, and how to stay calm during volatility.",
            "diversification": "Explain how spreading investments reduces risk and improves long-term outcomes.",
            "compound_growth": "Show the power of time and consistent investing with specific examples.",
            "risk_management": "Help them understand how to balance growth and safety based on their goals."
        }
        
        return base_prompt + topic_specific.get(topic, "Provide comprehensive, practical guidance.")
    
    def _get_fallback_educational_content(self, topic: str) -> str:
        """Get fallback educational content when AI is unavailable."""
        
        content_library = {
            "market_crashes": """
# Understanding Market Crashes

Market crashes are a normal part of investing. Here's what you need to know:

**Key Facts:**
• Major corrections (20%+ drops) happen every 3-4 years on average
• Severe crashes (40%+ drops) occur roughly once per decade
• Every major crash in history has been followed by recovery

**What to Do:**
• Stay invested - timing the market is nearly impossible
• Continue regular contributions during downturns
• Focus on your long-term goals, not short-term volatility
• Remember that crashes create buying opportunities

**Historical Perspective:**
The 2008 financial crisis, 2020 pandemic crash, and dot-com bubble all seemed devastating at the time, but patient investors who stayed the course were ultimately rewarded with strong long-term returns.
""",
            
            "diversification": """
# The Power of Diversification

Diversification is your best defense against investment risk:

**What It Means:**
• Spreading investments across different asset classes
• Not putting all your eggs in one basket
• Reducing the impact of any single investment's poor performance

**How It Helps:**
• Smooths out portfolio volatility
• Provides more consistent returns over time
• Reduces the risk of major losses

**Practical Application:**
• Mix stocks, bonds, and other assets
• Include domestic and international investments
• Consider different company sizes and sectors
• Rebalance periodically to maintain target allocations
""",
            
            "compound_growth": """
# The Magic of Compound Growth

Compound growth is the most powerful force in investing:

**How It Works:**
• Your returns earn returns
• Small amounts grow into large sums over time
• Starting early makes an enormous difference

**Example:**
£1,000 invested at 7% annual returns:
• After 10 years: £1,967
• After 20 years: £3,870
• After 30 years: £7,612

**Key Lessons:**
• Time is more important than timing
• Consistent contributions amplify the effect
• Even small amounts can grow significantly
• The earlier you start, the less you need to save
"""
        }
        
        return content_library.get(topic, f"Educational content about {topic} is not available in offline mode.")