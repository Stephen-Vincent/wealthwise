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
    Focuses on educational explanations and insights with beginner-friendly language
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
        Generate AI summary for portfolio simulation results with educational focus
        Enhanced with detailed underwater period analysis
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
            
            # Calculate annual return
            try:
                annual_return = ((end_value / total_contributed) ** (1/timeframe) - 1) * 100
                annual_return_str = f"{annual_return:.1f}%"
            except:
                annual_return_str = "N/A"

            # Get investment types for education
            stock_names = []
            for stock in stocks_picked:
                symbol = stock.get("symbol", "UNKNOWN")
                name = stock.get("name", symbol)
                allocation = stock.get("allocation", 0)
                stock_names.append(f"{symbol} ({name}) - {allocation:.1%}")
            
            investment_breakdown = "\n".join([f"  • {stock}" for stock in stock_names[:5]])
            if len(stock_names) > 5:
                investment_breakdown += f"\n  • ... and {len(stock_names) - 5} other investments"

            # Get detailed drawdown analysis
            drawdown_analysis = self._analyze_drawdowns_detailed(simulation_results, user_data)
            underwater_explanation = drawdown_analysis["explanation"]

            prompt = f"""
You are a friendly, patient financial educator speaking to someone who is new to investing. Your job is to explain their portfolio simulation results in a comprehensive, well-structured way that helps them understand what happened and learn about investing.

SIMULATION RESULTS TO EXPLAIN:
- Their Goal: {goal}
- Target Amount: £{target_value:,.0f}
- Total Investment: £{total_contributed:,.0f} (£{lump_sum:,.0f} upfront + £{monthly:,.0f} monthly for {timeframe} years)
- Final Portfolio Value: £{end_value:,.0f}
- Goal Status: {'✅ ACHIEVED!' if target_achieved else '❌ Not quite reached'}
- Risk Level: {risk_label} (Score: {risk_score}/100)
- Annual Growth Rate: {annual_return_str}

THEIR INVESTMENT PORTFOLIO:
{investment_breakdown}

UNDERWATER PERIODS ANALYSIS:
{underwater_explanation}

FORMATTING REQUIREMENTS:
- Use clear headings with emojis (## 🎯 Your Results, ## 📈 What This Means, etc.)
- Break content into well-structured sections
- Use bullet points for key information
- Include paragraph breaks for readability
- Use bold text for important numbers and concepts
- Add line breaks between major sections

CONTENT STRUCTURE:
## 🎯 Your Investment Results
[Summarize what happened in simple terms]

## 📈 What This Means
[Explain if this is good performance and put it in perspective]

## 🌊 Understanding the Ups and Downs
[Include the underwater periods analysis and explain what it means]

## 🧠 Key Investing Lesson
[Explain ONE important concept they demonstrated]

## 📚 What You Can Learn
[Share 2-3 practical lessons from this experience]

## 🚀 Your Financial Journey
[End with encouragement and next steps]

WRITING STYLE:
- Use simple, everyday language (avoid complex financial jargon)
- Explain concepts like you're talking to a friend over coffee
- Use analogies and comparisons to everyday things
- Be encouraging and positive, even if goals weren't met
- Use emojis appropriately but not excessively
- Make it educational, not sales-focused

IMPORTANT: Please incorporate the underwater periods analysis into your response, explaining why periods where the portfolio was worth less than contributions are normal and part of successful long-term investing.

Do NOT limit the length - provide a comprehensive, well-formatted explanation that truly educates the user about their investment journey.
"""
            
            response = await self._get_ollama_response(prompt)
            return self._format_ai_response(response)
            
        except Exception as e:
            logger.error(f"Error generating AI summary: {e}")
            return self._get_formatted_fallback_summary(total_contributed, end_value, timeframe, target_achieved, risk_label, stocks_picked)
    
    def _analyze_drawdowns_detailed(self, simulation_results: Dict, user_data: Dict) -> Dict:
        """
        Enhanced drawdown analysis that specifically identifies when portfolio value 
        was below invested amount and explains what happened
        """
        try:
            timeline = simulation_results.get("timeline", [])
            lump_sum = user_data.get("lump_sum", 0)
            monthly = user_data.get("monthly", 0)
            
            if not timeline:
                return self._get_estimated_drawdown_explanation(simulation_results, user_data)
            
            underwater_periods = []
            significant_drawdowns = []
            max_drawdown = 0
            peak_value = 0
            total_contributed_running = lump_sum
            
            for i, period in enumerate(timeline):
                portfolio_value = period.get("value", 0)
                year = period.get("year", i + 1)
                month = period.get("month", 1)
                
                # Calculate running total contributions
                if i > 0:  # Add monthly contribution for each period after initial
                    total_contributed_running += monthly
                
                # Track peak portfolio value
                if portfolio_value > peak_value:
                    peak_value = portfolio_value
                
                # Calculate drawdown from peak
                if peak_value > 0:
                    current_drawdown = (peak_value - portfolio_value) / peak_value * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                    
                    # Record significant drawdowns (>10%)
                    if current_drawdown > 10:
                        significant_drawdowns.append({
                            "period": f"Year {year}",
                            "drawdown_pct": current_drawdown,
                            "peak_value": peak_value,
                            "current_value": portfolio_value,
                            "loss_amount": peak_value - portfolio_value
                        })
                
                # Check if portfolio is "underwater" (below total contributions)
                if portfolio_value < total_contributed_running:
                    underwater_amount = total_contributed_running - portfolio_value
                    underwater_pct = (underwater_amount / total_contributed_running) * 100
                    
                    underwater_periods.append({
                        "period": f"Year {year}" + (f" Month {month}" if month > 1 else ""),
                        "portfolio_value": portfolio_value,
                        "total_contributed": total_contributed_running,
                        "underwater_amount": underwater_amount,
                        "underwater_pct": underwater_pct
                    })
            
            return {
                "has_underwater_periods": len(underwater_periods) > 0,
                "underwater_periods": underwater_periods,
                "significant_drawdowns": significant_drawdowns,
                "max_drawdown": max_drawdown,
                "total_underwater_periods": len(underwater_periods),
                "explanation": self._generate_underwater_explanation(underwater_periods, significant_drawdowns, user_data)
            }
            
        except Exception as e:
            logger.warning(f"Error in detailed drawdown analysis: {e}")
            return self._get_estimated_drawdown_explanation(simulation_results, user_data)

    def _generate_underwater_explanation(self, underwater_periods: List, significant_drawdowns: List, user_data: Dict) -> str:
        """Generate educational explanation for underwater periods"""
        
        if not underwater_periods:
            return """
🎉 Great News: Always Above Water!

Your portfolio stayed above the amount you invested throughout the entire period. This means your investments were always worth more than what you put in - excellent!

What this tells us:
• Your investment strategy was well-suited to market conditions
• You experienced growth with manageable volatility
• Your risk level and timing worked in your favor

Remember: Even successful portfolios can have temporary declines, so this is a particularly good outcome! 📈
"""
        
        # Find the worst underwater period
        worst_period = max(underwater_periods, key=lambda x: x["underwater_pct"])
        total_periods = len(underwater_periods)
        
        # Determine likely causes based on timing and market knowledge
        causes_explanation = self._get_likely_market_causes(underwater_periods, user_data)
        
        return f"""
📊 Understanding When Your Portfolio Was "Underwater"

Your portfolio experienced **{total_periods} periods** where it was worth less than the money you had invested. This is completely normal and happens to most investors!

🌊 Your Worst "Underwater" Period

When: {worst_period['period']}
Your Contributions: £{worst_period['total_contributed']:,.0f}
Portfolio Value: £{worst_period['portfolio_value']:,.0f}
Temporary Loss: £{worst_period['underwater_amount']:,.0f} ({worst_period['underwater_pct']:.1f}% below contributions)

🤔 Why This Happened

{causes_explanation}

💡 This Is Normal - Here's Why

Think of it like a roller coaster: You bought your ticket (made your investment), and sometimes you're going downhill. But you stay on the ride because you know it goes back up!

• Market cycles: All markets go through ups and downs
• Economic events: Recessions, panics, and uncertainty affect prices
• Company performance: Individual stocks can drag down portfolios temporarily
• Investor emotions: Fear can drive prices below true value

🎯 What Successful Investors Do

• Stay calm: Panic selling locks in losses
• Keep investing: Often the best buying opportunities come during declines
• Focus long-term: Temporary setbacks don't change long-term growth potential
• Learn from it: Each experience makes you a better investor

🚀 The Recovery Story

The good news? Your portfolio recovered from these underwater periods and went on to achieve your goals! This shows the power of staying invested through market turbulence.

Key lesson: Time in the market beats timing the market! 📈
"""

    def _get_likely_market_causes(self, underwater_periods: List, user_data: Dict) -> str:
        """Provide likely explanations for underwater periods based on timing"""
        
        timeframe = user_data.get("timeframe", 5)
        risk_level = user_data.get("risk_score", 50)
        
        # Generic explanation that works for most scenarios
        causes = []
        
        if risk_level > 70:
            causes.append("High-growth strategy volatility: Your aggressive approach naturally experiences bigger swings")
        
        if timeframe >= 5:
            causes.append("Market corrections: Normal periodic adjustments that happen every few years")
            causes.append("Economic uncertainty: Periods of recession, inflation fears, or geopolitical events")
        
        causes.extend([
            "Earnings disappointments: Some companies in your portfolio may have reported lower-than-expected profits",
            "Sector rotation: Money flowing out of your investment sectors into others",
            "Interest rate changes: Central bank policies affecting stock valuations",
            "General market sentiment: Periods when investors became more cautious overall"
        ])
        
        return "The underwater periods likely resulted from a combination of:\n\n" + "\n".join([f"• {cause}" for cause in causes])

    def _get_estimated_drawdown_explanation(self, simulation_results: Dict, user_data: Dict) -> Dict:
        """Fallback explanation when timeline data isn't available"""
        
        risk_score = simulation_results.get("risk_score", user_data.get("risk_score", 50))
        timeframe = user_data.get("timeframe", 5)
        
        # Estimate likelihood of underwater periods based on risk and timeframe
        if risk_score > 70 and timeframe >= 3:
            estimated_underwater = "very likely"
            estimated_periods = f"3-{min(timeframe//2, 8)} periods"
            max_underwater = f"15-25%"
        elif risk_score > 50 and timeframe >= 2:
            estimated_underwater = "likely"
            estimated_periods = f"1-{min(timeframe//3, 4)} periods"
            max_underwater = f"8-15%"
        elif timeframe >= 5:
            estimated_underwater = "possible"
            estimated_periods = "1-2 periods"
            max_underwater = f"5-10%"
        else:
            estimated_underwater = "unlikely"
            estimated_periods = "0-1 periods"
            max_underwater = f"less than 5%"
        
        explanation = f"""
🎢 Expected Portfolio Ups and Downs

Based on your **{user_data.get('risk_label', 'moderate')} risk approach** and **{timeframe}-year timeline**, underwater periods were **{estimated_underwater}**.

What to expect:
• Estimated underwater periods: {estimated_periods}
• Typical depth: Portfolio could be {max_underwater} below contributions temporarily
• Recovery time: Usually 6-18 months to get back above water

Why this happens:
• Market corrections occur every 2-3 years on average
• Higher-risk portfolios experience deeper but shorter-lived declines
• These temporary setbacks are the "price" of long-term growth

The key insight: Staying invested through these periods is what separates successful long-term investors from those who miss out on recovery gains! 📈
"""
        
        return {
            "has_underwater_periods": estimated_underwater in ["likely", "very likely"],
            "underwater_periods": [],
            "significant_drawdowns": [],
            "max_drawdown": 0,
            "total_underwater_periods": 0,
            "explanation": explanation,
            "is_estimated": True
        }
    
    def _format_ai_response(self, response: str) -> str:
        """Format AI response with better structure and readability"""
        # Clean up the response
        response = response.strip()
        
        # Ensure proper spacing around headings
        response = response.replace("##", "\n\n##")
        
        # Add spacing around bullet points
        lines = response.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
                
            # Add extra space before headings
            if line.startswith('##'):
                if formatted_lines and formatted_lines[-1] != '':
                    formatted_lines.append('')
                formatted_lines.append(line)
                formatted_lines.append('')
            # Handle bullet points
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # Join and clean up multiple empty lines
        formatted_response = '\n'.join(formatted_lines)
        while '\n\n\n' in formatted_response:
            formatted_response = formatted_response.replace('\n\n\n', '\n\n')
        
        return formatted_response.strip()
    
    async def _get_ollama_response(self, prompt: str) -> str:
        """Get response from local Ollama instance with better error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2500,  # Increased for longer, more comprehensive responses
                        "stop": ["Human:", "User:", "Question:"]
                    }
                }
                
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # Longer timeout for comprehensive responses
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("response", "").strip()
                        
                        if ai_response:
                            return ai_response
                        else:
                            raise Exception("Empty response from AI")
                    else:
                        raise Exception(f"Ollama API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Ollama request failed: {str(e)}")
            raise Exception(f"Failed to get AI response: {str(e)}")
    
    def _get_formatted_fallback_summary(self, total_contributed: float, end_value: float, timeframe: int, target_achieved: bool, risk_label: str, stocks_picked: List[Dict]) -> str:
        """Enhanced fallback with proper formatting when AI is unavailable"""
        growth = end_value - total_contributed
        return_pct = (growth / total_contributed * 100) if total_contributed > 0 else 0
        
        # Format stock holdings
        stock_list = []
        for stock in stocks_picked[:5]:
            symbol = stock.get("symbol", "UNKNOWN")
            name = stock.get("name", symbol)
            allocation = stock.get("allocation", 0)
            stock_list.append(f"  • {symbol} ({name}) - {allocation:.1%}")
        
        holdings_text = "\n".join(stock_list)
        if len(stocks_picked) > 5:
            holdings_text += f"\n  • ... and {len(stocks_picked) - 5} other investments"
        
        # Estimate drawdown explanation based on risk level
        drawdown_explanation = ""
        if risk_label.lower() in ["aggressive", "moderate aggressive", "ultra aggressive"]:
            drawdown_explanation = f"""

## ⚠️ Understanding Portfolio Ups and Downs

With your **{risk_label.lower()}** risk approach, your portfolio might experience temporary declines during market downturns. This is completely normal! Here's why:

• **Market cycles happen**: Even the best investments go through periods of decline
• **Higher growth potential = more volatility**: Your strategy aims for better long-term returns, but this means more ups and downs along the way
• **Temporary setbacks are normal**: Think of it like the weather - storms pass, but the seasons keep changing
• **Stay focused on long-term goals**: Successful investors don't panic during temporary market declines

**What causes portfolio declines?**
- Economic uncertainty or recession fears
- Global events affecting markets
- Company-specific news or earnings disappointments
- Interest rate changes by central banks
- General market sentiment shifts

Remember: These declines are temporary bumps on your long-term wealth-building journey! 📈"""
        
        if target_achieved:
            return f"""## 🎉 Congratulations - Goal Achieved!

Your investment plan was a success! You put in **£{total_contributed:,.0f}** over {timeframe} years, and it grew to **£{end_value:,.0f}**.

## 📈 Your Investment Performance

• **Total Growth**: £{growth:,.0f} ({return_pct:+.1f}%)
• **Strategy**: {risk_label} approach
• **Time Horizon**: {timeframe} years

## 🏆 Your Investment Portfolio

{holdings_text}

## 💡 Key Success Factors

• **Consistent investing**: You stuck to your plan with regular contributions
• **Appropriate risk level**: Your {risk_label.lower()} strategy matched your goals
• **Time in the market**: You gave your investments time to grow
• **Diversification**: You spread risk across multiple investments

{drawdown_explanation}

## 🚀 What This Means for Your Future

This success shows the power of patient, consistent investing. Your money worked hard while you focused on other things - that's the magic of long-term wealth building!

**Next steps**: Consider whether you want to set new, bigger financial goals or maintain this successful strategy for other objectives. You've proven you can make investing work for you! 💰"""
        else:
            shortfall = target_achieved - end_value
            return f"""## 📈 Your Investment Progress

Your investment journey shows important progress! You invested **£{total_contributed:,.0f}** over {timeframe} years, and it's now worth **£{end_value:,.0f}**.

## 📊 Your Results

• **Portfolio Growth**: £{growth:,.0f} ({return_pct:+.1f}%)
• **Target Gap**: £{abs(shortfall):,.0f} short of your goal
• **Strategy**: {risk_label} approach
• **Achievement**: Strong foundation built for future growth

## 🏗️ Your Investment Portfolio

{holdings_text}

## 🌱 Why This Is Still Progress

Remember: investing is like planting a tree. Sometimes it takes longer than expected to reach full height, but with patience and consistent care, it gets there.

• **You're building wealth**: Any growth is better than keeping money in low-yield savings
• **You're learning**: Every investment experience teaches valuable lessons
• **You're developing discipline**: Regular investing builds great financial habits
• **Time is on your side**: Compound growth accelerates over longer periods

{drawdown_explanation}

## 🎯 Moving Forward

Consider these options:
- **Extend your timeline**: Give your investments more time to grow
- **Increase contributions**: Boost monthly investments if possible
- **Adjust risk level**: Consider a slightly more aggressive approach if appropriate
- **Stay the course**: Continue your current successful strategy

You're on the right track - every successful investor has had portfolios that needed more time to reach their goals! 🌟"""

    async def analyze_portfolio_performance(
        self, 
        portfolio_data: Dict, 
        user_context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze existing portfolio performance with educational explanations
        """
        try:
            analysis_data = self._prepare_analysis_data(portfolio_data)
            prompt = self._create_educational_performance_prompt(analysis_data, user_context)
            response = await self._get_ollama_response(prompt)
            
            return {
                "success": True,
                "analysis": self._format_ai_response(response),
                "metrics": analysis_data,
                "timestamp": datetime.now().isoformat(),
                "type": "performance_analysis",
                "educational_focus": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_analysis": self._get_educational_fallback_analysis(portfolio_data)
            }
    
    def _prepare_analysis_data(self, portfolio_data: Dict) -> Dict:
        """Extract key metrics for analysis with better formatting"""
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
            "goal": portfolio_data.get("goal", "building wealth"),
            "target_value": portfolio_data.get("target_value", 0),
            "holdings": portfolio_data.get("results", {}).get("stocks_picked", [])
        }
    
    def _create_educational_performance_prompt(self, data: Dict, user_context: Optional[Dict]) -> str:
        """Create educational prompt for performance analysis with better formatting"""
        total_invested = data["total_invested"]
        total_value = data["total_value"]
        total_return = data["total_return"]
        return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        user_level = user_context.get("experience_level", "beginner") if user_context else "beginner"
        
        return f"""
You are teaching a {user_level} investor about their portfolio performance. Create a well-formatted, comprehensive educational response.

THEIR PORTFOLIO NUMBERS:
- Money They Put In: £{total_invested:,.0f}
- What It's Worth Now: £{total_value:,.0f}
- Their Gain/Loss: £{total_return:,.0f} ({return_percent:+.1f}%)
- Time Invested: {data['timeframe']} years
- Their Goal: {data['goal']}
- Risk Level: {data['risk_label']}

FORMAT YOUR RESPONSE WITH:
- Clear headings using ## and emojis
- Bullet points for key information
- Bold text for important numbers
- Proper paragraph breaks
- Educational analogies and comparisons

EXPLAIN:
1. What these numbers mean in simple terms
2. Whether this is good performance (compare to savings accounts, inflation)
3. Why portfolios experience ups and downs
4. What they can learn from this experience
5. How this fits their long-term goals
6. Any potential drawdown scenarios and why they happen

Use encouraging language and focus on education. Do NOT limit the response length - provide comprehensive, well-formatted education.
"""
    
    def _get_educational_fallback_analysis(self, portfolio_data: Dict) -> str:
        """Educational fallback for performance analysis with better formatting"""
        return """## 📊 Understanding Your Investment Performance

Investment performance is like the weather - it changes daily, but what matters is the long-term pattern.

## 🎢 Why Portfolio Values Change

Your portfolio's value will go up and down, and that's completely normal:

• **Daily market movements**: Thousands of factors influence prices every day
• **Economic news**: Interest rates, inflation, and economic reports affect markets
• **Company performance**: Earnings reports and business developments impact stock prices
• **Global events**: Political changes and world events can cause market reactions
• **Investor emotions**: Fear and greed drive short-term market movements

## ✈️ Think of It Like Air Travel

Investing is like a plane ride - there's turbulence along the way, but you're still heading toward your destination.

## 🔑 Key Things to Remember

• **Stay diversified**: Don't put all your eggs in one basket
• **Think long-term**: Successful investing is measured in years, not days
• **Keep learning**: The more you understand, the more confident you'll feel
• **Stay consistent**: Regular investing builds wealth over time

## 🌟 You're Building Your Future

Every experienced investor has been where you are now. You're building wealth for your future, and that's something to be proud of! 🎯"""
    
    def _get_fallback_lesson(self, topic: str) -> str:
        """Fallback educational lesson with better formatting"""
        lessons = {
            "diversification": """## 🥚 Diversification: Don't Put All Your Eggs in One Basket

Diversification is like not putting all your eggs in one basket. If you drop the basket, you don't lose everything!

## 💡 How It Works

In investing, this means owning different types of investments instead of just one:

• **Multiple companies**: Don't invest in just one business
• **Different sectors**: Technology, healthcare, finance, etc.
• **Various regions**: Domestic and international markets
• **Asset classes**: Stocks, bonds, real estate

## 🍽️ Think of It Like a Balanced Meal

You want vegetables, protein, and grains, not just one food. Your investment portfolio works the same way!

## 🎯 The Benefits

• **Reduced risk**: One bad investment won't ruin your portfolio
• **Smoother returns**: Ups and downs balance each other out
• **Better sleep**: Less worry about any single investment""",
            
            "compound_interest": """## ❄️ Compound Interest: The Snowball Effect

Compound interest is like a snowball rolling down a hill - it starts small but gets bigger and bigger as it picks up more snow.

## 🔄 How It Works

With investments, you earn money on your original investment, then you earn money on the money you earned! This creates a powerful cycle:

• **Year 1**: Earn money on your initial investment
• **Year 2**: Earn money on original + Year 1 earnings  
• **Year 3**: Earn money on everything from Years 1 & 2
• **And so on**: The growth accelerates over time

## ⚡ Einstein's Opinion

Einstein supposedly called it "the most powerful force in the universe" - and starting early gives you more time for this magic to work!

## 🚀 The Key Takeaway

Time is your greatest asset. The earlier you start, the more compound interest can work its magic! ✨""",
            
            "default": f"""## 📚 Understanding {topic.title()}

{topic.title()} is an important part of building wealth over time.

## 🔑 Key Principles

• **Start early**: Time is your greatest advantage
• **Stay consistent**: Regular investing builds wealth
• **Keep learning**: Knowledge builds confidence
• **Be patient**: Good things take time

## 🎯 Remember

You don't need to be an expert to start investing - you just need to begin and keep learning along the way!

## 🌟 Your Journey

Every small step you take today builds toward a more secure financial future."""
        }
        
        return lessons.get(topic.lower(), lessons["default"])