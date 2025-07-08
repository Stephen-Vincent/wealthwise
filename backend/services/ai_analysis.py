from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json
import asyncio
import aiohttp

# Set up logging
logger = logging.getLogger(__name__)

class AIAnalysisService:
    """
    AI-powered portfolio analysis service using Ollama
    Focuses on educational explanations and insights
    """
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "llama3.1:8b"
    
    async def generate_portfolio_summary(
        self, 
        stocks_picked: List[Dict], 
        user_data: Dict[str, Any], 
        risk_score: int, 
        risk_label: str, 
        simulation_results: Dict[str, Any]
    ) -> str:
        """
        Generate AI summary for portfolio simulation results
        """
        try:
            goal = user_data.get("goal", "wealth building")
            lump_sum = user_data.get("lump_sum", 0)
            monthly = user_data.get("monthly", 0)
            timeframe = user_data.get("timeframe", 10)
            target_value = user_data.get("target_value", 50000)
            
            start_value = simulation_results.get("starting_value", 0)
            end_value = simulation_results.get("end_value", 0)
            
            total_contributed = lump_sum + (monthly * timeframe * 12)
            target_achieved = end_value >= target_value
            
            try:
                total_return_percentage = ((end_value - total_contributed) / total_contributed) * 100
                total_return_str = f"{total_return_percentage:.2f}%"
            except:
                total_return_str = "N/A"

            stock_list = ", ".join([stock.get("symbol", "UNKNOWN") for stock in stocks_picked])

            prompt = f"""
Act as a friendly financial educator explaining a portfolio simulation to a beginner investor.

PORTFOLIO SIMULATION RESULTS:
- Investment Goal: {goal}
- Target Amount: £{target_value:,.2f}
- Amount Invested: £{total_contributed:,.2f} (£{lump_sum:,.2f} lump sum + £{monthly:,.2f}/month for {timeframe} years)
- Final Portfolio Value: £{end_value:,.2f}
- Total Return: {total_return_str}
- Goal {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}
- Risk Profile: {risk_label} (score: {risk_score})
- Investments: {stock_list}

Please provide a clear, educational explanation that:
1. Summarizes what happened in simple terms
2. Explains whether this is good performance and why
3. Explains how the risk profile influenced the results
4. Provides 2-3 key lessons about investing
5. Keeps language encouraging and educational
6. Maximum 250 words

Focus on education, not advice.
"""
            
            response = await self._get_ollama_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return f"Portfolio simulation completed. Your portfolio grew from £{start_value:,.2f} to £{end_value:,.2f} over {timeframe} years with a {risk_label.lower()} risk strategy."
    
    async def analyze_portfolio_performance(
        self, 
        portfolio_data: Dict, 
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze existing portfolio performance with AI explanations
        """
        try:
            analysis_data = self._prepare_analysis_data(portfolio_data)
            prompt = self._create_performance_prompt(analysis_data, user_context)
            response = await self._get_ollama_response(prompt)
            
            return {
                "success": True,
                "analysis": response,
                "metrics": analysis_data,
                "timestamp": datetime.now().isoformat(),
                "type": "performance_analysis"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_analysis": self._get_fallback_analysis(portfolio_data)
            }
    
    async def analyze_risk_allocation(self, portfolio_data: Dict) -> Dict:
        """
        Analyze portfolio risk and allocation with educational explanations
        """
        try:
            risk_data = self._prepare_risk_data(portfolio_data)
            prompt = self._create_risk_prompt(risk_data)
            response = await self._get_ollama_response(prompt)
            
            return {
                "success": True,
                "analysis": response,
                "risk_data": risk_data,
                "timestamp": datetime.now().isoformat(),
                "type": "risk_analysis"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_analysis": self._get_fallback_risk_analysis(portfolio_data)
            }
    
    async def explain_portfolio_changes(
        self, 
        portfolio_data: Dict, 
        previous_data: Optional[Dict] = None
    ) -> Dict:
        """
        Explain what happened to the portfolio and why (educational focus)
        """
        try:
            changes_data = self._prepare_changes_data(portfolio_data, previous_data)
            prompt = self._create_explanation_prompt(changes_data)
            response = await self._get_ollama_response(prompt)
            
            return {
                "success": True,
                "explanation": response,
                "changes_data": changes_data,
                "timestamp": datetime.now().isoformat(),
                "type": "portfolio_explanation"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_explanation": self._get_fallback_explanation(portfolio_data)
            }
    
    # ========== PRIVATE HELPER METHODS ==========
    
    async def _get_ollama_response(self, prompt: str) -> str:
        """Get response from local Ollama instance"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
                
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "Analysis completed but response empty.")
                    else:
                        raise Exception(f"Ollama API returned status {response.status}")
                        
        except Exception as e:
            raise Exception(f"Failed to get Ollama response: {str(e)}")
    
    def _prepare_analysis_data(self, portfolio_data: Dict) -> Dict:
        """Extract key metrics for analysis"""
        results = portfolio_data.get("results", {})
        
        return {
            "total_value": results.get("end_value", 0),
            "total_invested": portfolio_data.get("lump_sum", 0) + 
                           (portfolio_data.get("monthly", 0) * 12 * portfolio_data.get("timeframe", 0)),
            "total_return": results.get("end_value", 0) - (
                portfolio_data.get("lump_sum", 0) + 
                (portfolio_data.get("monthly", 0) * 12 * portfolio_data.get("timeframe", 0))
            ),
            "timeframe": portfolio_data.get("timeframe", 0),
            "risk_label": portfolio_data.get("risk_label", "Unknown"),
            "goal": portfolio_data.get("goal", ""),
            "target_value": portfolio_data.get("target_value", 0),
            "holdings": portfolio_data.get("results", {}).get("stocks_picked", [])
        }
    
    def _prepare_risk_data(self, portfolio_data: Dict) -> Dict:
        """Extract risk-related data"""
        holdings = portfolio_data.get("results", {}).get("stocks_picked", [])
        
        return {
            "risk_profile": portfolio_data.get("risk_label", "Unknown"),
            "risk_score": portfolio_data.get("risk_score", 0),
            "total_holdings": len(holdings),
            "largest_holding": max(holdings, key=lambda x: x.get("allocation", 0)) if holdings else None
        }
    
    def _prepare_changes_data(self, current_data: Dict, previous_data: Optional[Dict]) -> Dict:
        """Prepare data showing what changed"""
        current_value = current_data.get("results", {}).get("end_value", 0)
        
        if previous_data:
            previous_value = previous_data.get("results", {}).get("end_value", 0)
            change = current_value - previous_value
            change_percent = (change / previous_value * 100) if previous_value > 0 else 0
        else:
            change = 0
            change_percent = 0
        
        return {
            "current_value": current_value,
            "change_amount": change,
            "change_percent": change_percent,
            "timeframe": current_data.get("timeframe", 0),
            "holdings": current_data.get("results", {}).get("stocks_picked", [])
        }
    
    def _create_performance_prompt(self, data: Dict, user_context: Optional[Dict]) -> str:
        """Create educational prompt for performance analysis"""
        total_invested = data["total_invested"]
        total_value = data["total_value"]
        total_return = data["total_return"]
        return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        user_level = user_context.get("experience_level", "beginner") if user_context else "beginner"
        
        return f"""
You are a friendly financial educator helping a {user_level} investor understand their portfolio performance.

PORTFOLIO DETAILS:
- Amount Invested: £{total_invested:,.2f}
- Current Value: £{total_value:,.2f}
- Total Return: £{total_return:,.2f} ({return_percent:+.1f}%)
- Investment Period: {data['timeframe']} years
- Investment Goal: {data['goal']}
- Risk Profile: {data['risk_label']}

Explain in simple terms what these numbers mean, whether this is good performance, and provide 2-3 educational insights. Maximum 300 words.
"""
    
    def _create_risk_prompt(self, data: Dict) -> str:
        """Create prompt for risk analysis explanation"""
        return f"""
Explain portfolio risk and diversification to a beginner investor.

PORTFOLIO RISK DETAILS:
- Risk Profile: {data['risk_profile']}
- Risk Score: {data['risk_score']}/100
- Number of Holdings: {data['total_holdings']}

Explain what their risk profile means, assess diversification, and provide educational insights. Maximum 250 words.
"""
    
    def _create_explanation_prompt(self, data: Dict) -> str:
        """Create prompt for explaining portfolio changes"""
        return f"""
You are a financial educator helping someone understand what happened to their investment portfolio.

PORTFOLIO CHANGES:
- Current Value: £{data['current_value']:,.2f}
- Change: £{data['change_amount']:+,.2f} ({data['change_percent']:+.1f}%)
- Investment Period: {data['timeframe']} years

Explain what these changes mean, why portfolios fluctuate, and provide educational context. Maximum 200 words.
"""
    
    def _get_fallback_analysis(self, portfolio_data: Dict) -> str:
        """Fallback analysis when AI is unavailable"""
        total_invested = portfolio_data.get("lump_sum", 0) + (portfolio_data.get("monthly", 0) * 12 * portfolio_data.get("timeframe", 0))
        current_value = portfolio_data.get("results", {}).get("end_value", 0)
        
        if current_value > total_invested:
            return f"Your portfolio has grown from £{total_invested:,.2f} to £{current_value:,.2f}! This positive return shows your investments have increased in value over time."
        else:
            return f"Your portfolio is currently valued at £{current_value:,.2f} compared to your investment of £{total_invested:,.2f}. Temporary decreases are normal in investing."
    
    def _get_fallback_risk_analysis(self, portfolio_data: Dict) -> str:
        """Fallback risk analysis"""
        risk_label = portfolio_data.get("risk_label", "Unknown")
        return f"Your portfolio has a {risk_label} risk profile, balancing potential growth with volatility you're comfortable with."
    
    def _get_fallback_explanation(self, portfolio_data: Dict) -> str:
        """Fallback explanation"""
        return """
Investment portfolios change in value due to market movements, company performance, and economic factors. 
This is completely normal and expected. The key principles to remember are: diversification helps reduce risk, 
time in the market is important, and staying focused on long-term goals helps you weather short-term volatility.
"""