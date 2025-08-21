from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging

from database import schemas
from database.db import get_db
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Try to import AI analysis service (optional)
try:
    from services.ai_analysis import AIAnalysisService
    ai_service = AIAnalysisService()
    AI_ANALYSIS_AVAILABLE = True
except Exception as e:
    logger.warning(f"AI Analysis service not available: {e}")
    ai_service = None
    AI_ANALYSIS_AVAILABLE = False


# Utility function to sanitize floats in the response to ensure JSON compliance
def sanitize_floats(data):
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_floats(i) for i in data]
    elif isinstance(data, float):
        return 0.0 if math.isnan(data) or math.isinf(data) else data
    return data


async def get_portfolio_simulation(simulation_input: dict, db: Session):
    """
    Get portfolio simulation using the NEW modular simulator only.
    """
    try:
        # Import inside the function so we can catch and log import-time errors cleanly
        from services.portfolio_simulator import simulate_portfolio_workflow
        logger.info("Using modular portfolio simulator")
        return await simulate_portfolio_workflow(simulation_input, db)
    except Exception as e:
        # Log full traceback to reveal root cause (missing deps, bad export, init error, etc.)
        logger.exception("Modular simulator unavailable or failed")
        raise HTTPException(
            status_code=503,
            detail="Portfolio simulation failed. Check server logs for details."
        ) from e


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Enhanced onboarding endpoint with comprehensive risk assessment and
    portfolio simulation using the modular simulator.
    """
    try:
        logger.info(f"Processing onboarding for user {onboarding_data.user_id}")

        # Step 1: Calculate comprehensive risk profile from onboarding input
        risk_profile = calculate_user_risk(onboarding_data)
        logger.info(
            "Risk assessment completed: score=%s, level=%s",
            risk_profile["risk_score"],
            risk_profile["risk_level"],
        )

        # Step 2: Extract risk score and label (keep legacy label only for backwards compatibility in UI, if needed)
        risk_score = risk_profile["risk_score"]
        risk_label = risk_profile["risk_level"]

        legacy_risk_mapping = {
            "Ultra Conservative": "Low",
            "Conservative": "Low",
            "Moderate Conservative": "Low",
            "Moderate": "Medium",
            "Moderate Aggressive": "Medium",
            "Aggressive": "High",
            "Ultra Aggressive": "High",
        }
        legacy_risk_label = legacy_risk_mapping.get(risk_label, "Medium")

        # Step 3: Merge risk data with onboarding input for simulation
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = legacy_risk_label  # if your simulator expects the legacy label
        simulation_input["detailed_risk_profile"] = risk_profile

        # Step 4: Run portfolio simulation (modular only)
        simulation_result = await get_portfolio_simulation(simulation_input, db)
        logger.info("Portfolio simulation completed for user %s", onboarding_data.user_id)

        # Step 5: Extract simulation data safely
        simulation_id = simulation_result.get("id")

        # New modular format preferred
        target_reached = simulation_result.get("target_achieved", False)
        simulation_data = simulation_result

        # Step 6: Construct response payload
        response_payload = {
            "id": simulation_id,
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "target_achieved": target_reached,
            "income_bracket": onboarding_data.income_bracket,

            # Enhanced risk info
            "risk_score": risk_score,
            "risk_label": risk_label,
            "legacy_risk_label": legacy_risk_label,
            "risk_description": risk_profile["risk_description"],
            "allocation_guidance": risk_profile["allocation_guidance"],
            "recommended_stock_allocation": risk_profile["recommended_stock_allocation"],
            "recommended_bond_allocation": risk_profile["recommended_bond_allocation"],
            "risk_explanation": risk_profile["explanation"],

            # Include all simulation results
            **simulation_data,
            "created_at": datetime.utcnow().isoformat(),

            # Metadata about simulator used
            "simulator_type": simulation_result.get("simulator_type", "Modular"),
            "enhanced_features": simulation_result.get("enhanced_features_enabled", {}),
            "has_shap_explanations": simulation_result.get("has_shap_explanations", False),
            "has_visualizations": simulation_result.get("has_visualizations", False),
        }

        logger.info("Onboarding completed successfully for user %s", onboarding_data.user_id)
        return sanitize_floats(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Onboarding failed for user %s: %s", onboarding_data.user_id, str(e))
        import traceback
        logger.error("Full traceback: %s", traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding processing failed: {str(e)}",
        ) from e


@router.post("/legacy", status_code=status.HTTP_201_CREATED)
async def create_onboarding_legacy(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Legacy endpoint kept for compatibility; it now routes through the modular simulator.
    """
    try:
        logger.info(f"Processing legacy onboarding for user {onboarding_data.user_id}")

        # Use the legacy risk calculation but still run the modular simulator
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        simulation_result = await get_portfolio_simulation(simulation_input, db)

        simulation_id = simulation_result.get("id")
        target_reached = simulation_result.get("target_achieved", False)

        response_payload = {
            "id": simulation_id,
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "target_achieved": target_reached,
            "income_bracket": onboarding_data.income_bracket,
            "risk_score": risk_score,
            "risk_label": risk_label,
            **simulation_result,
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info("Legacy onboarding (modular-backed) completed successfully for user %s", onboarding_data.user_id)
        return sanitize_floats(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy onboarding failed for user %s: %s", onboarding_data.user_id, str(e))
        import traceback
        logger.error("Full traceback: %s", traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legacy onboarding processing failed: {str(e)}",
        ) from e


@router.get("/health")
async def health_check():
    """Check the health of onboarding service dependencies (modular simulator only)."""
    try:
        health_status = {
            "service": "onboarding",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "ai_analysis": "available" if AI_ANALYSIS_AVAILABLE else "unavailable",
        }

        try:
            from services.portfolio_simulator import simulate_portfolio_workflow  # noqa: F401
            health_status["modular_simulator"] = "available"
        except Exception as e:
            health_status["modular_simulator"] = f"unavailable: {e}"
            health_status["status"] = "degraded"

        # Risk assessor check
        try:
            from services.risk_assessor import calculate_user_risk  # noqa: F401
            health_status["risk_assessor"] = "available"
        except Exception as e:
            health_status["risk_assessor"] = f"unavailable: {e}"
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        return {
            "service": "onboarding",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }