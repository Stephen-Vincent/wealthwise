# services/risk_assessor.py

"""
Enhanced risk assessment module that evaluates user risk based on comprehensive 
behavioral and financial input using a trained ML model.

Returns:
- A numerical risk score (1-100)
- A detailed risk profile with allocation guidance
- Component scores for transparency
"""

import pandas as pd
import pickle
import os
import logging
from typing import Dict, Any, Union

# Set up logging
logger = logging.getLogger(__name__)

class RiskAssessmentError(Exception):
    """Custom exception for risk assessment errors."""
    pass

def calculate_user_risk(sim_input: Union[Dict[str, Any], object]) -> Dict[str, Any]:
    """
    Calculates comprehensive user risk profile using the enhanced ML model.

    Args:
        sim_input: User input data, can be a dict or a Pydantic model with .dict() method.
                  Expected fields:
                  - years_of_experience: int
                  - loss_tolerance: str
                  - panic_behavior: str  
                  - financial_behavior: str
                  - engagement_level: str
                  - goal: str (investment goal)
                  - target_value: float
                  - lump_sum: float (optional)
                  - monthly: float (optional)
                  - timeframe: int
                  - income_bracket: str

    Returns:
        dict: Comprehensive risk profile containing:
            - risk_score: float (1-100)
            - risk_level: str
            - risk_description: str
            - allocation_guidance: str
            - recommended_stock_allocation: int
            - recommended_bond_allocation: int
            - explanation: str
    """
    try:
        # Convert Pydantic model to dictionary if needed
        if hasattr(sim_input, "dict"):
            sim_input = sim_input.dict()

        # Debug input received (safer logging)
        input_keys = list(sim_input.keys()) if isinstance(sim_input, dict) else "Invalid input type"
        logger.info(f"Risk assessment input received with keys: {input_keys}")
        
        # Validate required fields
        required_fields = [
            'years_of_experience', 'goal', 'target_value', 'timeframe', 'income_bracket'
        ]
        
        missing_fields = [field for field in required_fields if field not in sim_input or sim_input[field] is None]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            raise RiskAssessmentError(f"Missing required fields: {missing_fields}")

        # Load the enhanced trained model with better path resolution
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "..", "ai_models", "risk_model", "enhanced_model.pkl")
        model_path = os.path.normpath(model_path)

        logger.info(f"Attempting to load model from: {model_path}")

        if not os.path.exists(model_path):
            # Fallback to old model path if enhanced model doesn't exist
            fallback_path = os.path.join(current_dir, "..", "..", "ai_models", "risk_model", "model.pkl")
            fallback_path = os.path.normpath(fallback_path)
            logger.warning(f"Enhanced model not found at {model_path}")
            logger.warning(f"Trying fallback path: {fallback_path}")
            
            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                logger.error(f"No model found at either path: {model_path} or {fallback_path}")
                raise RiskAssessmentError(f"Risk model not found. Checked paths: {model_path}, {fallback_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise RiskAssessmentError(f"Failed to load risk model: {str(e)}")

        # Map incoming input to expected model input format with better error handling
        try:
            mapped_input = {
                # Core financial data
                "years_of_experience": int(sim_input["years_of_experience"]),
                "investment_goal": str(sim_input["goal"]),
                "target_amount": float(sim_input["target_value"]),
                "lump_sum_investment": float(sim_input.get("lump_sum", 0) or 0),
                "monthly_investment": float(sim_input.get("monthly", 0) or 0),
                "timeframe": int(sim_input["timeframe"]),
                "income": str(sim_input["income_bracket"]),
                
                # Behavioral risk factors (with fallbacks for backward compatibility)
                "loss_tolerance": str(sim_input.get("loss_tolerance", "wait_and_see")),
                "panic_behavior": str(sim_input.get("panic_behavior", "no_experience")),
                "financial_behavior": str(sim_input.get("financial_behavior", "save_half")),
                "engagement_level": str(sim_input.get("engagement_level", "monthly")),
            }
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error mapping input data: {str(e)}")
            raise RiskAssessmentError(f"Invalid input data format: {str(e)}")

        # Validate behavioral inputs with better error messages
        valid_values = {
            "loss_tolerance": ["sell_immediately", "wait_and_see", "buy_more"],
            "panic_behavior": ["yes_always", "yes_sometimes", "no_never", "no_experience"],
            "financial_behavior": ["invest_all", "save_half", "save_all", "spend_it"],
            "engagement_level": ["daily", "weekly", "monthly", "quarterly", "rarely"],
            "investment_goal": ["buy a house", "vacation", "emergency fund", "retirement", "save for a car", "wealth building"],
            "income": ["low", "medium", "high"]
        }

        # Clean and validate inputs
        for field, valid_options in valid_values.items():
            current_value = mapped_input[field].lower().strip() if mapped_input[field] else ""
            
            # Try to find a match (case-insensitive)
            matched_value = None
            for option in valid_options:
                if current_value == option.lower():
                    matched_value = option
                    break
            
            if matched_value:
                mapped_input[field] = matched_value
            else:
                logger.warning(f"Invalid {field}: '{mapped_input[field]}', using default")
                # Use middle/safe option as default
                defaults = {
                    "loss_tolerance": "wait_and_see",
                    "panic_behavior": "no_experience", 
                    "financial_behavior": "save_half",
                    "engagement_level": "monthly",
                    "investment_goal": "wealth building",
                    "income": "medium"
                }
                mapped_input[field] = defaults.get(field, valid_options[0])

        logger.info(f"Mapped and validated input for model: {mapped_input}")

        # Create DataFrame for model prediction
        try:
            input_df = pd.DataFrame([mapped_input])
            logger.info(f"Created DataFrame with shape: {input_df.shape}")
            logger.info(f"DataFrame columns: {list(input_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {str(e)}")
            raise RiskAssessmentError(f"Failed to prepare data for model: {str(e)}")

        # Predict risk score using the enhanced model
        try:
            risk_score_raw = model.predict(input_df)[0]
            risk_score = float(max(1, min(100, risk_score_raw)))  # Clamp to 1-100 range
            logger.info(f"Raw prediction: {risk_score_raw}, Clamped: {risk_score}")
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            logger.error(f"Input DataFrame info: {input_df.info()}")
            raise RiskAssessmentError(f"Model prediction failed: {str(e)}")

        # Generate comprehensive risk profile
        try:
            risk_profile = generate_risk_profile(risk_score, mapped_input)
            logger.info(f"Generated risk profile: score={risk_score}, level={risk_profile['risk_level']}")
        except Exception as e:
            logger.error(f"Failed to generate risk profile: {str(e)}")
            raise RiskAssessmentError(f"Failed to generate risk profile: {str(e)}")
        
        return {
            "risk_score": round(risk_score, 1),
            **risk_profile
        }

    except RiskAssessmentError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {str(e)}")
        logger.error(f"Input type: {type(sim_input)}")
        logger.error(f"Input content: {sim_input}")
        
        # Return a safe fallback instead of crashing
        return {
            "risk_score": 35.0,
            "risk_level": "Moderate",
            "risk_description": "Balanced Risk",
            "allocation_guidance": "Balanced stock and bond allocation",
            "recommended_stock_allocation": 60,
            "recommended_bond_allocation": 40,
            "explanation": "Default risk profile due to assessment error. Please contact support."
        }


def generate_risk_profile(risk_score: float, user_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive risk profile based on score and user characteristics.
    
    Args:
        risk_score: Predicted risk score (1-100)
        user_input: User input data for context
        
    Returns:
        dict: Detailed risk profile information
    """
    
    # Enhanced risk categories with more nuanced levels
    if risk_score < 15:
        risk_level = "Ultra Conservative"
        risk_description = "Extremely Low Risk"
        allocation_guidance = "Focus on savings accounts, CDs, and government bonds"
        stock_allocation = 10
        bond_allocation = 90
        
    elif risk_score < 25:
        risk_level = "Conservative"
        risk_description = "Very Low Risk"
        allocation_guidance = "Primarily bonds and stable value funds with minimal stock exposure"
        stock_allocation = 20
        bond_allocation = 80
        
    elif risk_score < 40:
        risk_level = "Moderate Conservative"
        risk_description = "Low to Moderate Risk"
        allocation_guidance = "Balanced portfolio with bond emphasis for stability"
        stock_allocation = 35
        bond_allocation = 65
        
    elif risk_score < 55:
        risk_level = "Moderate"
        risk_description = "Balanced Risk"
        allocation_guidance = "Balanced stock and bond allocation for steady growth"
        stock_allocation = 60
        bond_allocation = 40
        
    elif risk_score < 70:
        risk_level = "Moderate Aggressive"
        risk_description = "Growth Oriented"
        allocation_guidance = "Stock-heavy portfolio with some bonds for diversification"
        stock_allocation = 75
        bond_allocation = 25
        
    elif risk_score < 85:
        risk_level = "Aggressive"
        risk_description = "High Growth Focus"
        allocation_guidance = "Primarily stocks with minimal bond allocation"
        stock_allocation = 85
        bond_allocation = 15
        
    else:  # 85-100
        risk_level = "Ultra Aggressive"
        risk_description = "Maximum Growth Potential"
        allocation_guidance = "Almost entirely stocks for maximum long-term growth"
        stock_allocation = 95
        bond_allocation = 5

    # Generate personalized explanation
    explanation = generate_risk_explanation(risk_score, user_input, risk_level)

    return {
        "risk_level": risk_level,
        "risk_description": risk_description,
        "allocation_guidance": allocation_guidance,
        "recommended_stock_allocation": stock_allocation,
        "recommended_bond_allocation": bond_allocation,
        "explanation": explanation
    }


def generate_risk_explanation(risk_score: float, user_input: Dict[str, Any], risk_level: str) -> str:
    """
    Generate personalized explanation for the risk assessment.
    
    Args:
        risk_score: Risk score (1-100)
        user_input: User input data
        risk_level: Determined risk level
        
    Returns:
        str: Personalized explanation
    """
    
    explanations = []
    
    try:
        # Experience-based insights
        experience = user_input.get("years_of_experience", 0)
        if experience == 0:
            explanations.append("As a new investor, this assessment considers your limited market experience.")
        elif experience >= 15:
            explanations.append("Your extensive investment experience allows for more sophisticated strategies.")
        elif experience >= 5:
            explanations.append("Your moderate investment experience provides a good foundation for balanced approaches.")
        
        # Goal-based insights
        goal = user_input.get("investment_goal", "")
        goal_context = {
            "emergency fund": "Emergency funds require conservative, liquid investments.",
            "retirement": "Long-term retirement goals allow for more growth-oriented strategies.",
            "buy a house": "House purchases typically require moderate risk due to specific timing needs.",
            "wealth building": "Wealth building goals can accommodate higher risk for greater growth potential.",
            "vacation": "Short-term vacation goals favor conservative approaches.",
            "save for a car": "Car purchases require balanced risk considering the timeline."
        }
        
        if goal in goal_context:
            explanations.append(goal_context[goal])
        
        # Behavioral insights
        loss_tolerance = user_input.get("loss_tolerance", "")
        if loss_tolerance == "buy_more":
            explanations.append("Your tendency to buy during market dips indicates strong risk tolerance.")
        elif loss_tolerance == "sell_immediately":
            explanations.append("Your preference to avoid losses suggests a conservative approach is appropriate.")
        
        # Timeframe insights
        timeframe = user_input.get("timeframe", 0)
        if timeframe >= 15:
            explanations.append("Your long investment timeframe allows for higher risk tolerance.")
        elif timeframe <= 3:
            explanations.append("Your short timeframe necessitates more conservative investments.")
        
        # Combine explanations
        if explanations:
            explanation = " ".join(explanations)
        else:
            explanation = f"Your {risk_level.lower()} profile balances your investment goals with your risk comfort level."
            
    except Exception as e:
        logger.warning(f"Error generating explanation: {str(e)}")
        explanation = f"Your {risk_level.lower()} profile is based on your investment preferences and financial situation."
    
    return explanation


def calculate_user_risk_legacy(sim_input) -> tuple:
    """
    Legacy function that returns (risk_score, risk_label) tuple for backward compatibility.
    
    Args:
        sim_input: User input data
        
    Returns:
        tuple: (risk_score, risk_label)
    """
    try:
        logger.info("Using legacy risk calculation wrapper")
        result = calculate_user_risk(sim_input)
        
        # Convert new risk levels to legacy format
        legacy_mapping = {
            "Ultra Conservative": "Low",
            "Conservative": "Low", 
            "Moderate Conservative": "Low",
            "Moderate": "Medium",
            "Moderate Aggressive": "Medium",
            "Aggressive": "High",
            "Ultra Aggressive": "High"
        }
        
        risk_score = result["risk_score"]
        legacy_label = legacy_mapping.get(result["risk_level"], "Medium")
        
        logger.info(f"Legacy result: score={risk_score}, label={legacy_label}")
        return int(risk_score), legacy_label
        
    except Exception as e:
        logger.error(f"Legacy risk calculation failed: {str(e)}")
        # Return safe defaults
        return 35, "Medium"


# Debugging function to test model loading
def test_model_loading():
    """Test function to check if the model can be loaded."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_paths = [
            os.path.normpath(os.path.join(current_dir, "..", "..", "ai_models", "risk_model", "enhanced_model.pkl")),
            os.path.normpath(os.path.join(current_dir, "..", "..", "ai_models", "risk_model", "model.pkl"))
        ]
        
        print("Testing model loading...")
        for path in model_paths:
            print(f"Checking path: {path}")
            print(f"Path exists: {os.path.exists(path)}")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✅ Successfully loaded model from: {path}")
                    return True
                except Exception as e:
                    print(f"❌ Failed to load model from {path}: {str(e)}")
        
        print("❌ No valid model found")
        return False
        
    except Exception as e:
        print(f"❌ Error testing model loading: {str(e)}")
        return False


# For testing purposes
if __name__ == "__main__":
    # Test model loading first
    print("=" * 50)
    print("MODEL LOADING TEST")
    print("=" * 50)
    test_model_loading()
    
    print("\n" + "=" * 50)
    print("RISK ASSESSMENT TEST")
    print("=" * 50)
    
    # Test cases
    test_conservative = {
        "years_of_experience": 1,
        "loss_tolerance": "sell_immediately",
        "panic_behavior": "yes_always",
        "financial_behavior": "save_all",
        "engagement_level": "rarely",
        "goal": "emergency fund",
        "target_value": 10000,
        "lump_sum": 5000,
        "monthly": 200,
        "timeframe": 2,
        "income_bracket": "low"
    }
    
    test_aggressive = {
        "years_of_experience": 20,
        "loss_tolerance": "buy_more",
        "panic_behavior": "no_never", 
        "financial_behavior": "invest_all",
        "engagement_level": "daily",
        "goal": "retirement",
        "target_value": 1000000,
        "lump_sum": 100000,
        "monthly": 3000,
        "timeframe": 25,
        "income_bracket": "high"
    }
    
    try:
        print("Testing Conservative Profile:")
        result = calculate_user_risk(test_conservative)
        print(f"Risk Score: {result['risk_score']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Allocation: {result['recommended_stock_allocation']}% stocks, {result['recommended_bond_allocation']}% bonds")
        print()
        
        print("Testing Aggressive Profile:")
        result = calculate_user_risk(test_aggressive)
        print(f"Risk Score: {result['risk_score']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Allocation: {result['recommended_stock_allocation']}% stocks, {result['recommended_bond_allocation']}% bonds")
        
        print("\nTesting Legacy Function:")
        score, label = calculate_user_risk_legacy(test_conservative)
        print(f"Legacy result: {score}, {label}")
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")