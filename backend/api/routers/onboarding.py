from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging
import asyncio

from database import schemas
from database.db import get_db
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Try to import AI analysis service
try:
    from services.ai_analysis import AIAnalysisService
    ai_service = AIAnalysisService()
    AI_ANALYSIS_AVAILABLE = True
except ImportError as e:
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
    Get portfolio simulation using the best available simulator.
    Tries new modular simulator first, falls back to legacy.
    """
    try:
        # Try new modular portfolio simulator first
        from services.portfolio_simulator import simulate_portfolio_workflow
        logger.info("Using new modular portfolio simulator")
        return await simulate_portfolio_workflow(simulation_input, db)
        
    except ImportError:
        logger.info("New modular simulator not available, trying legacy simulator")
        try:
            # Try legacy portfolio simulator
            from services.portfolio_simulator import simulate_portfolio
            logger.info("Using legacy portfolio simulator")
            return await simulate_portfolio(simulation_input, db)
            
        except ImportError:
            logger.error("No portfolio simulator available")
            raise HTTPException(
                status_code=503,
                detail="Portfolio simulation service is not available. Please ensure the portfolio simulator module is properly installed."
            )

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Enhanced onboarding endpoint with comprehensive risk assessment and portfolio simulation.
    
    Uses the best available portfolio simulator (modular or legacy) and provides
    detailed risk profiling with allocation guidance.
    """
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

        # Step 4: Run portfolio simulation with enhanced input
        simulation_result = await get_portfolio_simulation(simulation_input, db)
        
        logger.info(f"Portfolio simulation completed for user {onboarding_data.user_id}")

        # Step 5: Extract simulation data safely (handle different response formats)
        simulation_id = simulation_result.get("id")
        
        # Handle different result formats from modular vs legacy simulators
        if "results" in simulation_result:
            # Legacy format
            target_reached = simulation_result["results"].get("target_reached", False)
            simulation_data = simulation_result
        else:
            # New modular format
            target_reached = simulation_result.get("target_achieved", False)
            simulation_data = simulation_result

        # Step 6: Construct enhanced response payload with new risk information
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
            **simulation_data,
            "created_at": datetime.utcnow().isoformat(),
            
            # Add metadata about which simulator was used
            "simulator_type": simulation_result.get("simulator_type", "Legacy"),
            "enhanced_features": simulation_result.get("enhanced_features_enabled", {}),
            "has_shap_explanations": simulation_result.get("has_shap_explanations", False),
            "has_visualizations": simulation_result.get("has_visualizations", False)
        }

        logger.info(f"Onboarding completed successfully for user {onboarding_data.user_id}")

        # Step 7: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding processing failed: {str(e)}"
        )

@router.post("/legacy", status_code=status.HTTP_201_CREATED)
async def create_onboarding_legacy(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Legacy endpoint that returns the old format for backward compatibility.
    Use the main endpoint for enhanced risk assessment features.
    """
    try:
        logger.info(f"Processing legacy onboarding for user {onboarding_data.user_id}")
        
        # Use the legacy function
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        # Step 2: Merge risk data with onboarding input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        # Step 3: Run portfolio simulation with full input
        simulation_result = await get_portfolio_simulation(simulation_input, db)

        # Step 4: Extract data safely (handle different response formats)
        simulation_id = simulation_result.get("id")
        
        # Handle different result formats
        if "results" in simulation_result:
            # Legacy format
            target_reached = simulation_result["results"].get("target_reached", False)
        else:
            # New modular format
            target_reached = simulation_result.get("target_achieved", False)

        # Step 5: Construct response payload (original format)
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
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Legacy onboarding completed successfully for user {onboarding_data.user_id}")

        # Step 6: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Legacy onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legacy onboarding processing failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Check the health of onboarding service dependencies"""
    try:
        health_status = {
            "service": "onboarding",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check portfolio simulator availability
        try:
            from services.portfolio_simulator import simulate_portfolio_workflow
            health_status["modular_simulator"] = "available"
        except ImportError:
            try:
                from services.portfolio_simulator import simulate_portfolio
                health_status["legacy_simulator"] = "available"
                health_status["modular_simulator"] = "unavailable"
            except ImportError:
                health_status["simulators"] = "unavailable"
                health_status["status"] = "degraded"
        
        # Check AI analysis service
        health_status["ai_analysis"] = "available" if AI_ANALYSIS_AVAILABLE else "unavailable"
        
        # Check risk assessor
        try:
            from services.risk_assessor import calculate_user_risk
            health_status["risk_assessor"] = "available"
        except ImportError:
            health_status["risk_assessor"] = "unavailable"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "service": "onboarding",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }