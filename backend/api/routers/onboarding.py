from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging


from database import schemas
from database.db import get_db 
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy
from services.portfolio_simulator import simulate_portfolio  # Now async
from services.ai_analysis import AIAnalysisService  # Updated import

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize AI service
ai_service = AIAnalysisService()

# Utility function to sanitize floats in the response to ensure JSON compliance
def sanitize_floats(data):
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_floats(i) for i in data]
    elif isinstance(data, float):
        return 0.0 if math.isnan(data) or math.isinf(data) else data
    return data

"""
Endpoint: POST /onboarding/

Expected input (schemas.OnboardingCreate):
{
    "years_of_experience": int,       # e.g., 4
    "loss_tolerance": str,            # e.g., "wait_and_see" (NEW)
    "panic_behavior": str,            # e.g., "no_never" (NEW) 
    "financial_behavior": str,        # e.g., "invest_all" (NEW)
    "engagement_level": str,          # e.g., "monthly" (NEW)
    "goal": str,                      # e.g., "retirement"
    "target_value": float,            # e.g., 50000.0
    "lump_sum": float,                # e.g., 3000.0
    "monthly": float,                 # e.g., 250.0
    "timeframe": int,                 # e.g., 5
    "income_bracket": str,           # e.g., "medium"
    "consent": bool,                  # e.g., True
    "name": str,                      # e.g., "Stephen Vincent"
    "user_id": int                    # e.g., 1
}

Workflow:
1. Accept onboarding data from frontend.
2. Pass it to `calculate_user_risk()` to get comprehensive risk profile.
3. Merge this risk data with the original input.
4. Pass the merged input to `simulate_portfolio()` to generate portfolio results.
5. Return the result to frontend in the shape of `schemas.SimulationResponse`.
"""

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Processing onboarding for user {onboarding_data.user_id}")
        
        # Step 1: Calculate comprehensive risk profile from onboarding input
        risk_profile = calculate_user_risk(onboarding_data)
        
        logger.info(f"Risk assessment completed: score={risk_profile['risk_score']}, level={risk_profile['risk_level']}")

        # Step 2: Extract risk score and label for backward compatibility
        risk_score = risk_profile["risk_score"] 
        risk_label = risk_profile["risk_level"]
        
        # Convert new detailed risk levels to legacy format for existing simulation logic
        legacy_risk_mapping = {
            "Ultra Conservative": "Low",
            "Conservative": "Low", 
            "Moderate Conservative": "Low",
            "Moderate": "Medium",
            "Moderate Aggressive": "Medium", 
            "Aggressive": "High",
            "Ultra Aggressive": "High"
        }
        
        legacy_risk_label = legacy_risk_mapping.get(risk_label, "Medium")

        # Step 3: Merge risk data with onboarding input for simulation
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = legacy_risk_label  # Use legacy format for simulation
        simulation_input["detailed_risk_profile"] = risk_profile  # Include full profile

        # Step 4: Run portfolio simulation with enhanced input (now async)
        simulation_result = await simulate_portfolio(simulation_input, db)
        
        logger.info(f"Portfolio simulation completed for user {onboarding_data.user_id}")

        # AI summary is now generated within simulate_portfolio, no need to generate again

        # Step 7: Construct enhanced response payload with new risk information
        response_payload = {
            "id": simulation_result["id"],
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "target_achieved": simulation_result["results"]["target_reached"],
            "income_bracket": onboarding_data.income_bracket,
            
            # Enhanced risk information
            "risk_score": risk_score,
            "risk_label": risk_label,  # New detailed risk level
            "legacy_risk_label": legacy_risk_label,  # Backward compatibility
            "risk_description": risk_profile["risk_description"],
            "allocation_guidance": risk_profile["allocation_guidance"],
            "recommended_stock_allocation": risk_profile["recommended_stock_allocation"],
            "recommended_bond_allocation": risk_profile["recommended_bond_allocation"],
            "risk_explanation": risk_profile["explanation"],
            
            # Include all simulation results
            **simulation_result,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Onboarding completed successfully for user {onboarding_data.user_id}")

        # Step 8: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"Onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding processing failed: {str(e)}"
        )


# Simplified legacy endpoint without problematic import
@router.post("/legacy", status_code=status.HTTP_201_CREATED)
async def create_onboarding_legacy(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Legacy endpoint that returns the old format for backward compatibility.
    Use the main endpoint for enhanced risk assessment features.
    """
    try:
        logger.info(f"Processing legacy onboarding for user {onboarding_data.user_id}")
        
        # Use the legacy function that's already imported at the top
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        # Step 2: Merge risk data with onboarding input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        # Step 3: Run portfolio simulation with full input (now async)
        simulation_result = await simulate_portfolio(simulation_input, db)

        # Step 4: AI summary is now generated within simulate_portfolio

        # Step 4.5: No need to store AI summary separately - it's already in simulation_result

        # Step 5: Construct response payload (original format)
        response_payload = {
            "id": simulation_result["id"],
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "target_achieved": simulation_result["results"]["target_reached"],
            "income_bracket": onboarding_data.income_bracket,
            "risk_score": risk_score,
            "risk_label": risk_label,
            **simulation_result,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Legacy onboarding completed successfully for user {onboarding_data.user_id}")

        # Step 6: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"Legacy onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legacy onboarding processing failed: {str(e)}"
        )