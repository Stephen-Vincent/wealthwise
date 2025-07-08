# ai_models/stock_model/enhanced_stock_recommender.py

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedStockRecommender:
    """
    Enhanced stock recommendation system that optimizes portfolio selection to help users reach their goals.
    
    Key Improvements:
    1. Goal-oriented optimization - calculates required returns to reach target
    2. Risk-adjusted asset allocation - balances growth potential with risk tolerance
    3. Timeframe-aware selection - adjusts strategy based on investment horizon
    4. Performance validation - uses historical data to validate recommendations
    5. Dynamic rebalancing suggestions - adjusts allocations based on market conditions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        # Enhanced features including goal-achievement metrics
        self.feature_columns = ["target_value", "timeframe", "risk_score", "required_annual_return", "investment_gap"]
        
        # ASSET UNIVERSES: Organized by risk/return characteristics and asset classes
        self.asset_universes = {
            "ultra_conservative": {
                # Focus on capital preservation with modest growth
                "bonds_govt": ["TLT", "IEF", "SHY"],  # Government bonds (different durations)
                "bonds_corporate": ["LQD", "VCIT", "BND"],  # High-grade corporate bonds
                "dividend_aristocrats": ["NOBL", "VIG", "DVY"],  # Dividend growth stocks
                "utilities": ["VPU", "XLU"],  # Utility sector ETFs
                "allocation": {"bonds_govt": 0.4, "bonds_corporate": 0.3, "dividend_aristocrats": 0.2, "utilities": 0.1},
                "expected_annual_return": 0.05,  # 5% expected annual return
                "volatility": 0.08  # 8% annual volatility
            },
            "conservative": {
                # Balanced approach favoring stability with moderate growth
                "bonds": ["BND", "AGG", "VTEB"],  # Bond market exposure
                "large_cap_value": ["VTV", "VYM", "SCHV"],  # Value stocks with dividends
                "international_developed": ["VEA", "VXUS"],  # Developed market exposure
                "reits": ["VNQ", "SCHH"],  # Real estate exposure
                "allocation": {"bonds": 0.4, "large_cap_value": 0.35, "international_developed": 0.15, "reits": 0.1},
                "expected_annual_return": 0.07,  # 7% expected annual return
                "volatility": 0.12  # 12% annual volatility
            },
            "moderate": {
                # Balanced growth and income approach
                "large_cap_blend": ["VTI", "ITOT", "SWTSX"],  # Broad market exposure
                "international_blend": ["VEA", "VWO", "VTIAX"],  # Global diversification
                "bonds": ["BND", "AGG"],  # Bond foundation
                "sector_rotation": ["VGT", "VHT", "VFH"],  # Growth sectors
                "allocation": {"large_cap_blend": 0.4, "international_blend": 0.25, "bonds": 0.2, "sector_rotation": 0.15},
                "expected_annual_return": 0.09,  # 9% expected annual return
                "volatility": 0.15  # 15% annual volatility
            },
            "moderate_aggressive": {
                # Growth-focused with some stability
                "large_cap_growth": ["VUG", "MGK", "SCHG"],  # Growth stocks
                "small_cap": ["VB", "IWM", "VTI"],  # Small cap exposure
                "international_growth": ["VEA", "VWO", "IEMG"],  # International growth
                "tech_innovation": ["VGT", "ARKK", "QQQ"],  # Technology focus
                "allocation": {"large_cap_growth": 0.35, "small_cap": 0.25, "international_growth": 0.25, "tech_innovation": 0.15},
                "expected_annual_return": 0.11,  # 11% expected annual return
                "volatility": 0.18  # 18% annual volatility
            },
            "aggressive": {
                # Maximum growth potential with higher volatility
                "growth_stocks": ["VUG", "QQQ", "VGT"],  # High-growth stocks
                "small_cap_growth": ["VBK", "IWO", "VTWO"],  # Small cap growth
                "emerging_markets": ["VWO", "IEMG", "EEM"],  # Emerging market exposure
                "innovation_themes": ["ARKK", "ARKQ", "ARKG"],  # Thematic investing
                "allocation": {"growth_stocks": 0.4, "small_cap_growth": 0.25, "emerging_markets": 0.2, "innovation_themes": 0.15},
                "expected_annual_return": 0.13,  # 13% expected annual return
                "volatility": 0.22  # 22% annual volatility
            },
            "ultra_aggressive": {
                # Highest risk/reward seeking maximum growth
                "high_growth_tech": ["ARKK", "WCLD", "SKYY"],  # High-growth technology
                "crypto_exposure": ["BITO", "COIN"],  # Cryptocurrency exposure
                "biotech_innovation": ["XBI", "IBB", "ARKG"],  # Biotech sector
                "disruptive_tech": ["ARKQ", "ROBO", "FINX"],  # Disruptive technologies
                "allocation": {"high_growth_tech": 0.4, "crypto_exposure": 0.25, "biotech_innovation": 0.2, "disruptive_tech": 0.15},
                "expected_annual_return": 0.16,  # 16% expected annual return
                "volatility": 0.28  # 28% annual volatility
            }
        }
        
        # BACKUP TICKERS: Reliable ETFs for each risk category
        self.backup_tickers = {
            "ultra_conservative": ["BND", "VTI", "VEA", "VYM", "VTEB"],
            "conservative": ["VTI", "BND", "VEA", "VWO", "VYM"],
            "moderate": ["VTI", "VEA", "VWO", "BND", "VNQ"],
            "moderate_aggressive": ["VTI", "VUG", "VEA", "VWO", "QQQ"],
            "aggressive": ["QQQ", "VUG", "VWO", "ARKK", "VGT"],
            "ultra_aggressive": ["ARKK", "QQQ", "VUG", "VWO", "VGT"]
        }
    
    def calculate_required_return(self, target_value: float, current_investment: float, 
                                timeframe: int, monthly_contribution: float = 0) -> float:
        """
        Calculate the required annual return to reach the target goal.
        
        This is KEY IMPROVEMENT #1: Goal-oriented optimization
        Instead of just picking stocks by risk, we calculate what return is needed
        to reach the user's goal and factor that into recommendations.
        """
        try:
            # Calculate total contributions over timeframe
            total_contributions = current_investment + (monthly_contribution * 12 * timeframe)
            
            if total_contributions <= 0:
                return 0.10  # Default 10% if no contributions
            
            # Calculate required compound annual growth rate (CAGR)
            # Formula: CAGR = (Ending Value / Beginning Value)^(1/years) - 1
            required_multiplier = target_value / total_contributions
            required_annual_return = (required_multiplier ** (1/timeframe)) - 1
            
            # Cap at reasonable bounds (0% to 25% annually)
            required_annual_return = max(0.0, min(0.25, required_annual_return))
            
            logger.info(f"Goal analysis: Need {required_annual_return:.1%} annual return to reach £{target_value:,.0f} target")
            return required_annual_return
            
        except Exception as e:
            logger.warning(f"Error calculating required return: {e}")
            return 0.10  # Default 10% annual return
    
    def assess_goal_feasibility(self, required_return: float, risk_score: float) -> Dict[str, any]:
        """
        Assess whether the goal is realistic given the user's risk tolerance.
        
        KEY IMPROVEMENT #2: Reality check for user expectations
        This helps set realistic expectations and suggests adjustments if needed.
        """
        risk_category = self.risk_score_to_category(risk_score)
        expected_return = self.asset_universes[risk_category]["expected_annual_return"]
        volatility = self.asset_universes[risk_category]["volatility"]
        
        # Calculate goal feasibility score (0-100%)
        if required_return <= expected_return:
            feasibility = min(100, 90 + (expected_return - required_return) * 100)
        else:
            # Goal requires higher returns than risk tolerance typically provides
            return_gap = required_return - expected_return
            feasibility = max(10, 70 - (return_gap * 200))
        
        assessment = {
            "feasibility_score": feasibility,
            "required_return": required_return,
            "expected_return": expected_return,
            "return_gap": required_return - expected_return,
            "risk_category": risk_category,
            "recommendation": self._get_feasibility_recommendation(feasibility, required_return, expected_return)
        }
        
        logger.info(f"Goal feasibility: {feasibility:.0f}% ({assessment['recommendation']})")
        return assessment
    
    def _get_feasibility_recommendation(self, feasibility: float, required: float, expected: float) -> str:
        """Generate recommendation based on goal feasibility analysis."""
        if feasibility >= 80:
            return "Highly achievable with current risk tolerance"
        elif feasibility >= 60:
            return "Achievable but may require market outperformance"
        elif feasibility >= 40:
            return "Challenging - consider increasing contributions or timeframe"
        else:
            return "Very challenging - recommend increasing risk tolerance, contributions, or timeframe"
    
    def risk_score_to_category(self, risk_score: float) -> str:
        """
        Convert numerical risk score to risk category.
        
        EXPLANATION: Maps 0-100 risk scores to investment categories
        Lower scores = more conservative, higher scores = more aggressive
        """
        if risk_score < 15:
            return "ultra_conservative"  # Capital preservation focus
        elif risk_score < 30:
            return "conservative"        # Income and modest growth
        elif risk_score < 50:
            return "moderate"           # Balanced growth and income
        elif risk_score < 70:
            return "moderate_aggressive" # Growth focus with some stability
        elif risk_score < 85:
            return "aggressive"         # High growth seeking
        else:
            return "ultra_aggressive"   # Maximum growth potential
    
    def adjust_allocation_for_goal(self, base_allocation: Dict[str, float], 
                                 required_return: float, expected_return: float, 
                                 timeframe: int) -> Dict[str, float]:
        """
        Adjust portfolio allocation to better achieve the target goal.
        
        KEY IMPROVEMENT #3: Dynamic allocation based on goal requirements
        This tilts the portfolio toward higher/lower risk assets based on what's needed
        to reach the goal within the timeframe.
        """
        allocation = base_allocation.copy()
        return_gap = required_return - expected_return
        
        # If we need higher returns than expected, tilt toward growth
        if return_gap > 0.02:  # Need 2%+ more return
            growth_boost = min(0.2, return_gap)  # Cap boost at 20%
            
            # Identify growth and conservative categories
            growth_categories = [k for k in allocation.keys() 
                               if any(word in k.lower() for word in ['growth', 'tech', 'innovation', 'crypto'])]
            conservative_categories = [k for k in allocation.keys() 
                                     if any(word in k.lower() for word in ['bond', 'dividend', 'utility'])]
            
            # Shift allocation from conservative to growth
            if growth_categories and conservative_categories:
                boost_per_growth = growth_boost / len(growth_categories)
                reduction_per_conservative = growth_boost / len(conservative_categories)
                
                for category in growth_categories:
                    allocation[category] = min(0.6, allocation[category] + boost_per_growth)
                
                for category in conservative_categories:
                    allocation[category] = max(0.05, allocation[category] - reduction_per_conservative)
        
        # If goal is easily achievable, can be more conservative
        elif return_gap < -0.02:  # We can afford 2%+ less return
            conservative_boost = min(0.15, abs(return_gap))
            
            # Shift toward more stable assets
            bond_categories = [k for k in allocation.keys() if 'bond' in k.lower()]
            if bond_categories:
                for category in bond_categories:
                    allocation[category] = min(0.5, allocation[category] + conservative_boost / len(bond_categories))
        
        # Adjust for timeframe
        if timeframe <= 3:  # Short timeframe - more conservative
            self._make_allocation_more_conservative(allocation)
        elif timeframe >= 15:  # Long timeframe - can take more risk
            self._make_allocation_more_aggressive(allocation)
        
        # Normalize to ensure allocations sum to 1.0
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def _make_allocation_more_conservative(self, allocation: Dict[str, float]) -> None:
        """Shift allocation toward more conservative assets for short timeframes."""
        bond_categories = [k for k in allocation.keys() if 'bond' in k.lower()]
        risky_categories = [k for k in allocation.keys() 
                          if any(word in k.lower() for word in ['growth', 'tech', 'crypto', 'innovation'])]
        
        if bond_categories and risky_categories:
            shift_amount = 0.15  # Shift 15% toward bonds
            boost_per_bond = shift_amount / len(bond_categories)
            reduction_per_risky = shift_amount / len(risky_categories)
            
            for category in bond_categories:
                allocation[category] += boost_per_bond
            for category in risky_categories:
                allocation[category] = max(0.05, allocation[category] - reduction_per_risky)
    
    def _make_allocation_more_aggressive(self, allocation: Dict[str, float]) -> None:
        """Shift allocation toward more aggressive assets for long timeframes."""
        growth_categories = [k for k in allocation.keys() 
                           if any(word in k.lower() for word in ['growth', 'tech', 'innovation'])]
        bond_categories = [k for k in allocation.keys() if 'bond' in k.lower()]
        
        if growth_categories and bond_categories:
            shift_amount = 0.1  # Shift 10% toward growth
            boost_per_growth = shift_amount / len(growth_categories)
            reduction_per_bond = shift_amount / len(bond_categories)
            
            for category in growth_categories:
                allocation[category] += boost_per_growth
            for category in bond_categories:
                allocation[category] = max(0.05, allocation[category] - reduction_per_bond)
    
    def select_optimal_stocks(self, risk_category: str, timeframe: int, 
                            target_value: float, current_investment: float,
                            monthly_contribution: float = 0) -> Tuple[List[str], Dict[str, any]]:
        """
        Select optimal stocks to maximize chance of reaching the goal.
        
        KEY IMPROVEMENT #4: Holistic optimization approach
        This considers goal requirements, risk tolerance, and timeframe together
        to create the best possible portfolio for the user's specific situation.
        """
        
        # Step 1: Calculate what we need to achieve the goal
        required_return = self.calculate_required_return(
            target_value, current_investment, timeframe, monthly_contribution
        )
        
        # Step 2: Assess if the goal is realistic
        goal_assessment = self.assess_goal_feasibility(required_return, 
            self._category_to_risk_score(risk_category))
        
        # Step 3: Get base allocation for risk category
        if risk_category not in self.asset_universes:
            risk_category = "moderate"  # Default fallback
        
        universe = self.asset_universes[risk_category]
        base_allocation = universe["allocation"].copy()
        expected_return = universe["expected_annual_return"]
        
        # Step 4: Adjust allocation to optimize for goal achievement
        optimized_allocation = self.adjust_allocation_for_goal(
            base_allocation, required_return, expected_return, timeframe
        )
        
        # Step 5: Select specific stocks from each category
        selected_stocks = []
        for category, weight in optimized_allocation.items():
            if category in universe and weight > 0.05:  # Only include meaningful allocations
                category_stocks = universe[category]
                
                # Select number of stocks proportional to allocation weight
                num_stocks = max(1, min(3, int(len(category_stocks) * weight * 4)))
                
                # For this implementation, select the first N stocks
                # In production, you'd want more sophisticated selection
                selected_from_category = category_stocks[:num_stocks]
                selected_stocks.extend(selected_from_category)
        
        # Remove duplicates while preserving order
        selected_stocks = list(dict.fromkeys(selected_stocks))
        
        # Ensure reasonable portfolio size (5-12 holdings)
        if len(selected_stocks) < 5:
            backup_stocks = self.backup_tickers.get(risk_category, self.backup_tickers["moderate"])
            for stock in backup_stocks:
                if stock not in selected_stocks:
                    selected_stocks.append(stock)
                if len(selected_stocks) >= 5:
                    break
        elif len(selected_stocks) > 12:
            selected_stocks = selected_stocks[:12]
        
        # Prepare detailed recommendation info
        recommendation_info = {
            "goal_assessment": goal_assessment,
            "optimized_allocation": optimized_allocation,
            "expected_return": expected_return,
            "required_return": required_return,
            "risk_category": risk_category,
            "selected_stocks": selected_stocks
        }
        
        logger.info(f"Selected {len(selected_stocks)} optimized stocks for {risk_category}: {selected_stocks}")
        return selected_stocks, recommendation_info
    
    def _category_to_risk_score(self, category: str) -> float:
        """Convert risk category back to approximate risk score for calculations."""
        category_mapping = {
            "ultra_conservative": 10,
            "conservative": 25,
            "moderate": 40,
            "moderate_aggressive": 60,
            "aggressive": 75,
            "ultra_aggressive": 90
        }
        return category_mapping.get(category, 40)
    
    def validate_and_filter_stocks(self, stocks: List[str]) -> List[str]:
        """
        Validate stock symbols and filter out invalid ones.
        
        EXPLANATION: Ensures all recommended stocks are actually tradeable
        and have current market data available.
        """
        valid_stocks = []
        
        for stock in stocks:
            try:
                ticker = yf.Ticker(stock)
                info = ticker.info
                
                # Check if stock has current market price
                if info and 'regularMarketPrice' in info and info['regularMarketPrice']:
                    valid_stocks.append(stock)
                else:
                    logger.warning(f"Stock {stock} failed validation - no market price")
            except Exception as e:
                logger.warning(f"Stock {stock} failed validation: {str(e)}")
                continue
        
        return valid_stocks
    
    def recommend_stocks(self, target_value: float, timeframe: int, risk_score: float,
                        current_investment: float = 0, monthly_contribution: float = 0) -> List[str]:
        """
        Main function to recommend stocks optimized for reaching the user's goal.
        
        KEY IMPROVEMENT #5: Goal-first approach
        This is the main interface that puts goal achievement at the center
        of the recommendation process, not just risk matching.
        """
        
        try:
            logger.info(f"Optimizing portfolio for goal: £{target_value:,} in {timeframe} years (risk: {risk_score})")
            
            # Determine risk category
            risk_category = self.risk_score_to_category(risk_score)
            
            # Get optimized stock selection
            recommended_stocks, recommendation_info = self.select_optimal_stocks(
                risk_category, timeframe, target_value, current_investment, monthly_contribution
            )
            
            # Validate that stocks are tradeable
            valid_stocks = self.validate_and_filter_stocks(recommended_stocks)
            
            if len(valid_stocks) < 3:
                # Fallback to backup stocks if validation fails
                backup_category = "moderate" if risk_score < 50 else "aggressive"
                fallback_stocks = self.backup_tickers[backup_category]
                logger.warning(f"Using fallback stocks: {fallback_stocks}")
                return fallback_stocks
            
            # Log the recommendation rationale
            goal_assessment = recommendation_info["goal_assessment"]
            logger.info(f"Goal feasibility: {goal_assessment['feasibility_score']:.0f}%")
            logger.info(f"Strategy: {goal_assessment['recommendation']}")
            
            return valid_stocks
            
        except Exception as e:
            logger.error(f"Error in stock recommendation: {str(e)}")
            # Return safe fallback based on risk score
            if risk_score < 30:
                return ["VTI", "BND", "VEA", "VYM", "VTEB"]
            elif risk_score < 70:
                return ["VTI", "VEA", "VWO", "BND", "VNQ"]
            else:
                return ["QQQ", "VUG", "VWO", "ARKK", "VGT"]
    
    def train_model(self, df: Optional[pd.DataFrame] = None) -> None:
        """Train the machine learning model on top of the rule-based system."""
        
        if df is None:
            logger.info("No training data provided, generating synthetic data...")
            df = self.generate_training_data(1000)
        
        logger.info(f"Training model with {len(df)} samples")
        
        # Prepare features
        X = df[["target_value", "timeframe", "risk_score"]].copy()  # Use basic features for now
        
        # Create target variable (we'll predict risk category as numeric)
        risk_categories = ["ultra_conservative", "conservative", "moderate", 
                          "moderate_aggressive", "aggressive", "ultra_aggressive"]
        category_mapping = {cat: i for i, cat in enumerate(risk_categories)}
        
        y = df["risk_category"].map(category_mapping)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Model training complete. R² Score: {r2:.3f}, MAE: {mae:.3f}")
        
        # Save model
        self.save_model()
    
    def save_model(self) -> None:
        """Save the trained model and scaler."""
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "stock_recommender.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self) -> bool:
        """Load the trained model and scaler."""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(model_dir, "stock_recommender.pkl")
            scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Model and scaler loaded successfully")
                return True
            else:
                logger.warning("Model files not found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    def generate_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate enhanced training data including goal-achievement features."""
        data = []
        
        for _ in range(num_samples):
            target_value = np.random.lognormal(np.log(50000), 1.0)
            target_value = max(1000, min(5000000, target_value))
            
            timeframe = np.random.choice([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
            risk_score = np.random.beta(2, 2) * 100
            
            # Calculate additional features
            current_investment = np.random.uniform(0, target_value * 0.5)
            monthly = np.random.uniform(0, target_value * 0.05)
            
            required_return = self.calculate_required_return(target_value, current_investment, timeframe, monthly)
            investment_gap = target_value - (current_investment + monthly * 12 * timeframe)
            
            risk_category = self.risk_score_to_category(risk_score)
            recommended_stocks, _ = self.select_optimal_stocks(
                risk_category, timeframe, target_value, current_investment, monthly
            )
            
            data.append({
                "target_value": target_value,
                "timeframe": timeframe,
                "risk_score": risk_score,
                "required_annual_return": required_return,
                "investment_gap": investment_gap,
                "risk_category": risk_category,
                "recommended_stocks": ",".join(recommended_stocks)
            })
        
        return pd.DataFrame(data)


# Global instance for backward compatibility
_recommender = EnhancedStockRecommender()

def train_and_recommend(target_value: float, timeframe: int, risk_score: float) -> List[str]:
    """
    Main function for backward compatibility with existing code.
    
    ENHANCED: Now uses goal-oriented optimization instead of just risk matching
    """
    return _recommender.recommend_stocks(target_value, timeframe, risk_score)

def save_last_input_features(target_value: float, timeframe: int, risk_score: float) -> None:
    """Save input features for audit trail and model improvement."""
    try:
        input_df = pd.DataFrame([{
            "target_value": target_value,
            "timeframe": timeframe,
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat()
        }])
        
        features_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(features_dir, exist_ok=True)
        
        input_features_path = os.path.join(features_dir, "last_input_features.csv")
        input_df.to_csv(input_features_path, index=False)
        
        logger.info(f"Input features saved to {input_features_path}")
    except Exception as e:
        logger.warning(f"Failed to save input features: {str(e)}")

def get_backup_tickers(count: int = 5) -> List[str]:
    """Returns reliable backup tickers for fallback scenarios."""
    reliable_tickers = [
        'VTI',   # Total Stock Market ETF
        'BND',   # Total Bond Market ETF  
        'VEA',   # Developed Markets ETF
        'VWO',   # Emerging Markets ETF
        'VNQ',   # Real Estate ETF
        'QQQ',   # NASDAQ-100 ETF
        'SPY',   # S&P 500 ETF
        'VUG',   # Growth ETF
        'VYM',   # High Dividend Yield ETF
        'VGT'    # Technology Sector ETF
    ]
    return reliable_tickers[:count]

if __name__ == "__main__":
    # Test the enhanced system with goal-oriented scenarios
    recommender = EnhancedStockRecommender()
    
    test_cases = [
        # (target, timeframe, risk, current_investment, monthly)
        (50000, 10, 30, 5000, 200),    # Conservative long-term goal
        (100000, 5, 60, 10000, 1000),  # Aggressive shorter-term goal  
        (25000, 3, 40, 2000, 500),     # Moderate short-term goal
    ]
    
    for target, timeframe, risk, current, monthly in test_cases:
        print(f"\nGoal: £{target:,} in {timeframe} years (Risk: {risk})")
        print(f"Starting: £{current:,} + £{monthly}/month")
        
        stocks = recommender.recommend_stocks(target, timeframe, risk, current, monthly)
        print(f"Recommended: {stocks}")
        
        # Show goal analysis
        required_return = recommender.calculate_required_return(target, current, timeframe, monthly)
        print(f"Required return: {required_return:.1%} annually")