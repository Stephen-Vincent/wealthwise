from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging

from database import schemas
from database.db import get_db
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy

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


def sanitize_floats(data):
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_floats(i) for i in data]
    elif isinstance(data, float):
        return 0.0 if math.isnan(data) or math.isinf(data) else data
    return data


def _normalize_simulation(sim_result: dict) -> dict:
    """
    Normalize the simulator response so the UI always finds canonical keys.
    We keep the original payload too, but ensure these exist:
    - id
    - stocks  (list)
    - breakdown (dict)
    - results (dict)
    - target_achieved (bool)
    """
    if not isinstance(sim_result, dict):
        sim_result = {}

    sim_id = sim_result.get("id")
    stocks = (
        sim_result.get("stocks")
        or sim_result.get("recommended_stocks")
        or []
    )
    breakdown = (
        sim_result.get("breakdown")
        or sim_result.get("allocation_breakdown")
        or {}
    )
    results = (
        sim_result.get("results")
        or sim_result.get("projection")
        or {}
    )
    target_achieved = sim_result.get("target_achieved", False)

    return {
        "id": sim_id,
        "stocks": stocks,
        "breakdown": breakdown,
        "results": results,
        "target_achieved": target_achieved,
    }


async def get_portfolio_simulation(simulation_input: dict, db: Session):
    """
    Use the NEW modular simulator only, importing directly from its module
    to avoid package-level side effects or missing re-exports.
    """
    try:
        # IMPORTANT: bypass services.portfolio_simulator __init__.py
        from services.portfolio_simulator.main_service import simulate_portfolio_workflow
        logger.info("Using modular portfolio simulator (direct module import)")
        return await simulate_portfolio_workflow(simulation_input, db)
    except Exception as e:
        logger.exception("Modular simulator unavailable or failed")
        raise HTTPException(
            status_code=503,
            detail="Portfolio simulation failed. Check server logs for details.",
        ) from e


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Enhanced onboarding endpoint with comprehensive risk assessment and
    portfolio simulation using the modular simulator.
    """
    try:
        logger.info("Processing onboarding for user %s", onboarding_data.user_id)

        # 1) Risk assessment
        risk_profile = calculate_user_risk(onboarding_data)
        risk_score = risk_profile["risk_score"]
        risk_label = risk_profile["risk_level"]
        logger.info(
            "Risk assessment completed: score=%s, level=%s",
            risk_score,
            risk_label,
        )

        # 2) Provide legacy mapping only if your simulator expects it
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

        # 3) Build simulator input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = legacy_risk_label
        simulation_input["detailed_risk_profile"] = risk_profile

        # 4) Run simulator
        simulation_result = await get_portfolio_simulation(simulation_input, db)
        logger.info("Portfolio simulation completed for user %s", onboarding_data.user_id)

        # 5) Normalize for UI
        normalized = _normalize_simulation(simulation_result)

        # 6) Build response
        response_payload = {
            "id": normalized["id"],
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "income_bracket": onboarding_data.income_bracket,

            # Risk info
            "risk_score": risk_score,
            "risk_label": risk_label,
            "legacy_risk_label": legacy_risk_label,
            "risk_description": risk_profile.get("risk_description"),
            "allocation_guidance": risk_profile.get("allocation_guidance"),
            "recommended_stock_allocation": risk_profile.get("recommended_stock_allocation"),
            "recommended_bond_allocation": risk_profile.get("recommended_bond_allocation"),
            "risk_explanation": risk_profile.get("explanation"),

            # Canonical UI keys
            "target_achieved": normalized["target_achieved"],
            "stocks": normalized["stocks"],
            "breakdown": normalized["breakdown"],
            "results": normalized["results"],

            # Include full simulator payload too
            **simulation_result,

            # Metadata
            "created_at": datetime.utcnow().isoformat(),
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
        logger.info("Processing legacy onboarding for user %s", onboarding_data.user_id)
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        simulation_result = await get_portfolio_simulation(simulation_input, db)
        normalized = _normalize_simulation(simulation_result)

        response_payload = {
            "id": normalized["id"],
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "income_bracket": onboarding_data.income_bracket,
            "risk_score": risk_score,
            "risk_label": risk_label,

            # Canonical UI keys
            "target_achieved": normalized["target_achieved"],
            "stocks": normalized["stocks"],
            "breakdown": normalized["breakdown"],
            "results": normalized["results"],

            # Full payload & metadata
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
    """
    Check the health of onboarding service dependencies (modular simulator only).
    Import directly from the module to avoid package-level side effects.
    """
    try:
        health_status = {
            "service": "onboarding",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "ai_analysis": "available" if AI_ANALYSIS_AVAILABLE else "unavailable",
        }

        try:
            # IMPORTANT: bypass services.portfolio_simulator __init__.py
            from services.portfolio_simulator.main_service import simulate_portfolio_workflow  # noqa: F401
            health_status["modular_simulator"] = "available"
        except Exception as e:
            health_status["modular_simulator"] = f"unavailable: {e}"
            health_status["status"] = "degraded"

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