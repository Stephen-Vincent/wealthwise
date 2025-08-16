"""
Enhanced Portfolio Simulator Service - Main Entry Point

This is the main service that orchestrates the entire portfolio simulation workflow.
It coordinates all the modular components and provides the public API.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging

# Import all the modular components
from .modules.goal_calculator import SmartGoalCalculator
from .modules.stock_recommender import EnhancedStockRecommender
from .modules.data_manager import StockDataManager
from .modules.portfolio_optimizer import PortfolioOptimizer
from .modules.simulation_engine import SimulationEngine
from .modules.crash_analyzer import MarketCrashAnalyzer
from .modules.ai_summarizer import AISummaryGenerator
from .modules.database_manager import DatabaseManager
from .modules.serialization_utils import SerializationManager

logger = logging.getLogger(__name__)

class EnhancedPortfolioSimulator:
    """
    Main orchestrator for the enhanced portfolio simulation system.
    
    This class coordinates all the modular components to provide:
    1. Goal-oriented portfolio optimization
    2. Market crash detection and news analysis
    3. SHAP explainable AI explanations
    4. Enhanced educational summaries
    5. Robust error handling and fallbacks
    """
    
    def __init__(self):
        # Initialize all module components
        self.goal_calculator = SmartGoalCalculator()
        self.stock_recommender = EnhancedStockRecommender()
        self.data_manager = StockDataManager()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.simulation_engine = SimulationEngine()
        self.crash_analyzer = MarketCrashAnalyzer()
        self.ai_summarizer = AISummaryGenerator()
        self.database_manager = DatabaseManager()
        self.serialization_manager = SerializationManager()
        
        logger.info("✅ Enhanced Portfolio Simulator initialized with all modules")
    
    async def simulate_portfolio(self, sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """
        Main entry point for portfolio simulation with all enhanced features.
        
        Args:
            sim_input: Dictionary containing user onboarding data
            db: Database session for saving results
        
        Returns:
            Enhanced simulation results with crash analysis and SHAP explanations
        """
        
        try:
            logger.info("🚀 Starting enhanced portfolio simulation")
            
            # STEP 1: Extract and validate user data
            user_data = self._extract_user_data(sim_input)
            risk_profile = self._extract_risk_profile(sim_input)
            
            logger.info(f"👤 User profile: {user_data['goal']}, £{user_data['target_value']:,.2f} target")
            logger.info(f"⚖️ Risk profile: {risk_profile['score']}/100 ({risk_profile['label']})")
            
            # STEP 2: Smart goal analysis
            goal_analysis = self.goal_calculator.calculate_smart_required_return(
                target_value=user_data["target_value"],
                current_investment=user_data["lump_sum"],
                timeframe=user_data["timeframe"],
                monthly_contribution=user_data["monthly"]
            )
            
            # STEP 3: Enhanced AI stock recommendations
            recommendation_result = await self.stock_recommender.get_enhanced_recommendations(
                goal_analysis=goal_analysis,
                risk_profile=risk_profile,
                user_data=user_data
            )
            
            # 🔍 CRITICAL: Extract SHAP explanations from recommendation result
            shap_explanations = {}
            if 'shap_explanations' in recommendation_result:
                shap_explanations = recommendation_result['shap_explanations']
                logger.info("✅ SHAP explanations found in recommendation result")
            elif 'shap_explanation' in recommendation_result:
                shap_explanations = recommendation_result['shap_explanation']
                logger.info("✅ SHAP explanation (singular) found in recommendation result")
            else:
                logger.warning("⚠️ No SHAP explanations found in recommendation result")
                logger.info(f"🔍 Recommendation result keys: {list(recommendation_result.keys())}")
            
            # STEP 4: Download and validate market data
            stock_data = await self.data_manager.download_stock_data(
                tickers=recommendation_result["stocks"],
                timeframe=user_data["timeframe"]
            )
            
            # STEP 5: Optimize portfolio weights
            weights = self.portfolio_optimizer.calculate_enhanced_weights(
                data=stock_data,
                risk_score=risk_profile["score"],
                recommendation_result=recommendation_result
            )
            
            # STEP 6: Create structured stock allocation
            stocks_picked = self._create_stock_allocation(
                tickers=recommendation_result["stocks"],
                weights=weights,
                recommendation_result=recommendation_result
            )
            
            # STEP 7: Run portfolio simulation
            simulation_results = self.simulation_engine.simulate_portfolio_growth(
                data=stock_data,
                weights=weights,
                lump_sum=user_data["lump_sum"],
                monthly=user_data["monthly"],
                timeframe=user_data["timeframe"]
            )
            
            # STEP 8: Market crash analysis with news integration
            enhanced_results = await self.crash_analyzer.add_crash_analysis(
                simulation_results=simulation_results,
                stock_data=stock_data,
                stocks_picked=stocks_picked
            )
            
            # STEP 9: Generate enhanced AI summary
            ai_summary = await self.ai_summarizer.generate_enhanced_summary(
                stocks_picked=stocks_picked,
                user_data=user_data,
                risk_profile=risk_profile,
                simulation_results=enhanced_results,
                goal_analysis=goal_analysis,
                recommendation_result=recommendation_result
            )
            
            # STEP 10: Save to database with proper serialization
            simulation = await self.database_manager.save_enhanced_simulation(
                db=db,
                sim_input=sim_input,
                user_data=user_data,
                risk_profile=risk_profile,
                ai_summary=ai_summary,
                stocks_picked=stocks_picked,
                simulation_results=enhanced_results,
                goal_analysis=goal_analysis,
                recommendation_result=recommendation_result,
                shap_explanations=shap_explanations  # 🔍 Pass SHAP data explicitly
            )
            
            logger.info(f"✅ Enhanced simulation completed successfully (ID: {simulation.id})")
            
            # 🔍 CRITICAL: Include SHAP in response
            response = self._format_response(simulation)
            response["shap_explanations"] = shap_explanations  # Include at top level
            response["recommendation_result"] = recommendation_result  # Include full recommendation data
            
            # Debug logging
            if shap_explanations:
                logger.info(f"✅ Including SHAP explanations in response: {type(shap_explanations)}")
            else:
                logger.warning("⚠️ No SHAP explanations to include in response")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Enhanced simulation failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return await self._handle_fallback_simulation(sim_input, db, str(e))
    
    def _extract_user_data(self, sim_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate user investment data from input."""
        return {
            "experience": sim_input.get("years_of_experience", 0),
            "goal": sim_input.get("goal", "wealth building"),
            "target_value": float(sim_input.get("target_value", 50000)),
            "lump_sum": float(sim_input.get("lump_sum", 0) or 0),
            "monthly": float(sim_input.get("monthly", 0) or 0),
            "timeframe": int(sim_input.get("timeframe", 10)),
            "income_bracket": sim_input.get("income_bracket", "medium")
        }
    
    def _extract_risk_profile(self, sim_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk assessment results from input."""
        return {
            "score": sim_input.get("risk_score", 35),
            "label": sim_input.get("risk_label", "Medium")
        }
    
    def _create_stock_allocation(self, tickers: List[str], weights, 
                               recommendation_result: Dict[str, Any]) -> List[Dict]:
        """Create structured list of selected stocks with allocations."""
        return [
            {
                "symbol": ticker,
                "name": self._get_company_name(ticker),
                "allocation": round(float(weight), 4),
                "explanation": self._get_stock_explanation(ticker, recommendation_result)
            }
            for ticker, weight in zip(tickers, weights)
        ]
    
    def _get_company_name(self, ticker: str) -> str:
        """Get human-readable company name for ticker."""
        name_mapping = {
            "VTI": "Vanguard Total Stock Market ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "VEA": "Vanguard FTSE Developed Markets ETF",
            "VTEB": "Vanguard Tax-Exempt Bond ETF",
            "VWO": "Vanguard Emerging Markets ETF",
            "VNQ": "Vanguard Real Estate ETF",
            "VGT": "Vanguard Information Technology ETF",
            "VUG": "Vanguard Growth ETF",
            "ARKK": "ARK Innovation ETF",
            "QQQ": "Invesco QQQ Trust ETF",
            "BITO": "ProShares Bitcoin Strategy ETF",
            "ARKQ": "ARK Autonomous Technology & Robotics ETF",
            "IBB": "iShares Biotechnology ETF",
            "FINX": "Global X FinTech ETF",
            "COIN": "Coinbase Global Inc"
        }
        return name_mapping.get(ticker, ticker)
    
    def _get_stock_explanation(self, ticker: str, recommendation_result: Dict[str, Any]) -> str:
        """Get explanation for why a specific stock was recommended."""
        if recommendation_result.get("method") == "fallback":
            return f"{ticker} selected based on risk profile matching"
        
        # Extract explanations from SHAP and factor analysis
        shap_explanation = recommendation_result.get("shap_explanations", {}) or recommendation_result.get("shap_explanation", {})
        
        if shap_explanation and "human_readable_explanation" in shap_explanation:
            explanations = shap_explanation["human_readable_explanation"]
            if explanations:
                for key, explanation in explanations.items():
                    if len(explanation) > 20:
                        return f"{ticker}: {explanation[:100]}..."
        
        return f"{ticker} recommended by AI analysis for your goals and risk profile"
    
    def _format_response(self, simulation) -> Dict[str, Any]:
        """Format simulation response for API return."""
        # Extract SHAP from multiple possible locations
        shap_data = {}
        if simulation.results:
            # Check for shap_explanations (plural)
            if 'shap_explanations' in simulation.results:
                shap_data = simulation.results['shap_explanations']
            # Check for shap_explanation (singular)
            elif 'shap_explanation' in simulation.results:
                shap_data = simulation.results['shap_explanation']
            # Check in portfolio_recommendations
            elif 'portfolio_recommendations' in simulation.results:
                recs = simulation.results['portfolio_recommendations']
                if isinstance(recs, dict):
                    if 'shap_explanations' in recs:
                        shap_data = recs['shap_explanations']
                    elif 'shap_explanation' in recs:
                        shap_data = recs['shap_explanation']
        
        # Log what we found
        if shap_data:
            logger.info(f"✅ Found SHAP data in simulation results: {type(shap_data)}")
        else:
            logger.warning("⚠️ No SHAP data found in simulation results")
            if simulation.results:
                logger.info(f"🔍 Available results keys: {list(simulation.results.keys())}")
        
        return {
            "id": simulation.id,
            "user_id": simulation.user_id,
            "name": simulation.name,
            "goal": simulation.goal,
            "target_value": simulation.target_value,
            "lump_sum": simulation.lump_sum,
            "monthly": simulation.monthly,
            "timeframe": simulation.timeframe,
            "target_achieved": simulation.target_achieved,
            "income_bracket": simulation.income_bracket,
            "risk_score": simulation.risk_score,
            "risk_label": simulation.risk_label,
            "ai_summary": simulation.ai_summary,
            "results": simulation.results,
            "created_at": simulation.created_at.isoformat() if simulation.created_at else datetime.utcnow().isoformat(),
            # Enhanced features
            "wealthwise_enhanced": simulation.results.get("wealthwise_enhanced", False) if simulation.results else False,
            "has_crash_analysis": bool(simulation.results.get("market_crash_analysis")) if simulation.results else False,
            "has_shap_explanations": bool(shap_data),  # Based on actual SHAP data found
            "methodology": simulation.results.get("methodology", "Enhanced portfolio simulation") if simulation.results else "Enhanced portfolio simulation",
            # 🔍 Include SHAP data in response
            "shap_explanations": shap_data
        }
    
    async def _handle_fallback_simulation(self, sim_input: Dict[str, Any], 
                                        db: Session, error: str) -> Dict[str, Any]:
        """Handle fallback to basic simulation when enhanced version fails."""
        logger.warning("🔄 Falling back to basic simulation")
        
        try:
            # Use basic components for fallback
            user_data = self._extract_user_data(sim_input)
            risk_profile = self._extract_risk_profile(sim_input)
            
            # Basic goal calculation
            goal_analysis = self.goal_calculator.calculate_smart_required_return(
                user_data["target_value"], user_data["lump_sum"], 
                user_data["timeframe"], user_data["monthly"]
            )
            
            # Fallback stock selection
            tickers = self.stock_recommender.get_fallback_stocks(risk_profile["score"])
            
            # Basic simulation
            stock_data = await self.data_manager.download_stock_data(tickers, user_data["timeframe"])
            weights = self.portfolio_optimizer.calculate_basic_weights(stock_data, risk_profile["score"])
            
            stocks_picked = [
                {"symbol": ticker, "name": self._get_company_name(ticker), "allocation": round(float(weight), 4)}
                for ticker, weight in zip(tickers, weights)
            ]
            
            simulation_results = self.simulation_engine.simulate_portfolio_growth(
                stock_data, weights, user_data["lump_sum"], user_data["monthly"], user_data["timeframe"]
            )
            
            # Basic AI summary
            ai_summary = await self.ai_summarizer.generate_basic_summary(
                stocks_picked, user_data, risk_profile, simulation_results
            )
            
            # Save basic simulation
            simulation = await self.database_manager.save_basic_simulation(
                db, sim_input, user_data, risk_profile, ai_summary, stocks_picked, simulation_results
            )
            
            response = self._format_response(simulation)
            response["fallback_used"] = True
            response["original_error"] = error
            response["has_shap_explanations"] = False
            response["shap_explanations"] = {}
            
            return response
            
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback simulation failed: {fallback_error}")
            raise ValueError(f"Both enhanced and fallback simulations failed: {error}, {fallback_error}")

# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

async def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Public API function for portfolio simulation.
    
    This is the main entry point that external code should use.
    """
    simulator = EnhancedPortfolioSimulator()
    return await simulator.simulate_portfolio(sim_input, db)

async def get_simulation_crash_analysis(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Get detailed crash analysis for a specific simulation.
    """
    try:
        crash_analyzer = MarketCrashAnalyzer()
        return await crash_analyzer.get_simulation_crash_details(simulation_id, db)
    except Exception as e:
        logger.error(f"❌ Error getting crash analysis: {e}")
        return {"error": str(e)}

async def generate_shap_visualization(simulation_id: int, db: Session) -> Optional[str]:
    """
    Generate SHAP visualization for a simulation.
    """
    try:
        ai_summarizer = AISummaryGenerator()
        return await ai_summarizer.create_shap_visualization(simulation_id, db)
    except Exception as e:
        logger.error(f"❌ Error generating SHAP visualization: {e}")
        return None

async def get_shap_explanations(simulation_id: int, db: Session) -> Dict[str, Any]:
    """
    Get SHAP explanations for a specific simulation.
    
    This function extracts SHAP explanations from the database and returns them
    in a format suitable for frontend consumption.
    """
    try:
        logger.info(f"🔍 Getting SHAP explanations for simulation {simulation_id}")
        
        from database import models
        simulation = db.query(models.Simulation).filter(models.Simulation.id == simulation_id).first()
        
        if not simulation:
            return {"error": "Simulation not found"}
        
        # Extract SHAP data from multiple possible locations
        shap_data = {}
        
        if simulation.results:
            # Check various possible locations for SHAP data
            locations_to_check = [
                ('shap_explanations', simulation.results.get('shap_explanations')),
                ('shap_explanation', simulation.results.get('shap_explanation')),
                ('portfolio_recommendations.shap_explanations', 
                 simulation.results.get('portfolio_recommendations', {}).get('shap_explanations') if isinstance(simulation.results.get('portfolio_recommendations'), dict) else None),
                ('portfolio_recommendations.shap_explanation', 
                 simulation.results.get('portfolio_recommendations', {}).get('shap_explanation') if isinstance(simulation.results.get('portfolio_recommendations'), dict) else None)
            ]
            
            for location_name, data in locations_to_check:
                if data:
                    shap_data = data
                    logger.info(f"✅ Found SHAP data at: {location_name}")
                    break
        
        if not shap_data:
            logger.warning(f"⚠️ No SHAP explanations found for simulation {simulation_id}")
            return {
                "simulation_id": simulation_id,
                "shap_explanations": {},
                "message": "No SHAP explanations available for this simulation",
                "available_data": list(simulation.results.keys()) if simulation.results else []
            }
        
        return {
            "simulation_id": simulation_id,
            "shap_explanations": shap_data,
            "message": "SHAP explanations retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting SHAP explanations: {e}")
        return {"error": str(e)}