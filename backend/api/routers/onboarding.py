from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging
import os

from database import schemas
from database.db import get_db 
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy
from services.ai_analysis import AIAnalysisService

# Set up logging
logger = logging.getLogger(__name__)

# 🎯 ENHANCED: Import modular portfolio simulator with fallback
try:
    from services.portfolio_simulator.main import simulate_portfolio, get_simulation_crash_analysis, generate_shap_visualization
    MODULAR_SIMULATOR_AVAILABLE = True
    logger.info("✅ Enhanced modular portfolio simulator loaded successfully")
except ImportError as e:
    # Fallback to existing portfolio simulator
    from services.portfolio_simulator import simulate_portfolio
    MODULAR_SIMULATOR_AVAILABLE = False
    logger.warning(f"⚠️ Modular simulator not available: {e}. Using standard simulator.")
    
    # Create placeholder functions for enhanced features
    async def get_simulation_crash_analysis(simulation_id: int, db):
        return {
            "simulation_id": simulation_id,
            "message": "Enhanced crash analysis not available - using standard simulator",
            "status": "placeholder"
        }
    
    async def generate_shap_visualization(simulation_id: int, db):
        return None

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
Enhanced Onboarding Endpoint with Modular Portfolio Simulator

This endpoint automatically detects and uses:
1. Enhanced modular portfolio simulator (when available)
2. Standard portfolio simulator (fallback)

Enhanced features include:
- Smart goal analysis (fixes 0% return issue)
- Market crash detection with news analysis
- SHAP explanations (when available)
- Enhanced AI summaries
- Robust error handling
"""

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    try:
        simulator_type = "Enhanced Modular" if MODULAR_SIMULATOR_AVAILABLE else "Standard"
        logger.info(f"🚀 Processing onboarding for user {onboarding_data.user_id} using {simulator_type} simulator")
        
        # Step 1: Calculate comprehensive risk profile
        risk_profile = calculate_user_risk(onboarding_data)
        logger.info(f"⚖️ Risk assessment completed: score={risk_profile['risk_score']}, level={risk_profile['risk_level']}")

        # Step 2: Extract risk data for simulation
        risk_score = risk_profile["risk_score"] 
        risk_label = risk_profile["risk_level"]
        
        # Convert detailed risk levels to legacy format for simulation compatibility
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

        # Step 3: Prepare simulation input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = legacy_risk_label
        simulation_input["detailed_risk_profile"] = risk_profile

        # Step 4: Run portfolio simulation (enhanced or standard)
        if MODULAR_SIMULATOR_AVAILABLE:
            logger.info("📈 Running enhanced portfolio simulation with crash analysis and SHAP explanations")
        else:
            logger.info("📊 Running standard portfolio simulation")
            
        simulation_result = await simulate_portfolio(simulation_input, db)
        
        logger.info(f"✅ Portfolio simulation completed for user {onboarding_data.user_id}")
        
        # Step 5: Extract enhanced features (if available)
        enhanced_features = {
            "modular_simulator_used": MODULAR_SIMULATOR_AVAILABLE,
            "has_crash_analysis": simulation_result.get("has_crash_analysis", False),
            "has_shap_explanations": simulation_result.get("has_shap_explanations", False),
            "wealthwise_enhanced": simulation_result.get("wealthwise_enhanced", False),
            "methodology": simulation_result.get("methodology", "Standard simulation")
        }
        
        # Log enhanced features
        if enhanced_features["has_crash_analysis"]:
            logger.info("📉 Market crash analysis included in results")
        if enhanced_features["has_shap_explanations"]:
            logger.info("🔍 SHAP explanations available for portfolio recommendations")

        # Step 6: Extract results data
        results = simulation_result.get("results", {})
        goal_analysis = results.get("goal_analysis", {})
        market_crash_analysis = results.get("market_crash_analysis", {})
        
        # Generate basic goal analysis if enhanced version not available
        if not goal_analysis and not MODULAR_SIMULATOR_AVAILABLE:
            goal_analysis = _generate_basic_goal_analysis(onboarding_data)
        
        # Step 7: Construct comprehensive response
        response_payload = {
            "id": simulation_result["id"],
            "user_id": onboarding_data.user_id,
            "name": onboarding_data.name,
            "goal": onboarding_data.goal,
            "target_value": onboarding_data.target_value,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "target_achieved": results.get("target_reached", False),
            "income_bracket": onboarding_data.income_bracket,
            
            # Enhanced risk information
            "risk_score": risk_score,
            "risk_label": risk_label,
            "legacy_risk_label": legacy_risk_label,
            "risk_description": risk_profile["risk_description"],
            "allocation_guidance": risk_profile["allocation_guidance"],
            "recommended_stock_allocation": risk_profile["recommended_stock_allocation"],
            "recommended_bond_allocation": risk_profile["recommended_bond_allocation"],
            "risk_explanation": risk_profile["explanation"],
            
            # 🎯 Enhanced simulation features
            "enhanced_features": enhanced_features,
            
            # 🎯 Smart goal analysis
            "goal_analysis": {
                "required_return_percent": goal_analysis.get("required_return_percent"),
                "can_reach_with_contributions": goal_analysis.get("can_reach_with_contributions"),
                "feasibility_rating": goal_analysis.get("feasibility_rating"),
                "message": goal_analysis.get("message"),
                "calculation_type": goal_analysis.get("calculation_type")
            } if goal_analysis else None,
            
            # 🎯 Market crash analysis summary
            "market_events": {
                "crashes_detected": market_crash_analysis.get("crashes_detected", 0),
                "overall_message": market_crash_analysis.get("overall_message"),
                "key_insights": market_crash_analysis.get("key_insights", [])[:3],
                "educational_summary": market_crash_analysis.get("educational_summary")
            } if market_crash_analysis else None,
            
            # Include all simulation results
            **simulation_result,
            "created_at": datetime.utcnow().isoformat()
        }

        success_msg = f"🎉 {'Enhanced' if MODULAR_SIMULATOR_AVAILABLE else 'Standard'} onboarding completed successfully"
        logger.info(f"{success_msg} for user {onboarding_data.user_id}")

        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"❌ Onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Onboarding processing failed: {str(e)}"
        )


@router.post("/legacy", status_code=status.HTTP_201_CREATED)
async def create_onboarding_legacy(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Legacy endpoint for backward compatibility.
    Returns original format regardless of which simulator is used.
    """
    try:
        logger.info(f"🔄 Processing legacy onboarding for user {onboarding_data.user_id}")
        
        # Use legacy risk calculation
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        # Prepare simulation input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        # Run simulation (will use whatever simulator is available)
        simulation_result = await simulate_portfolio(simulation_input, db)

        # Return legacy format response
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

        logger.info(f"✅ Legacy onboarding completed successfully for user {onboarding_data.user_id}")
        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"❌ Legacy onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legacy onboarding processing failed: {str(e)}"
        )


# 🎯 Enhanced feature endpoints
@router.get("/{simulation_id}/crash-analysis")
async def get_crash_analysis(simulation_id: int, db: Session = Depends(get_db)):
    """
    Get detailed crash analysis for a simulation.
    Works with both enhanced and standard simulators.
    """
    try:
        logger.info(f"📉 Getting crash analysis for simulation {simulation_id}")
        
        crash_analysis = await get_simulation_crash_analysis(simulation_id, db)
        
        if isinstance(crash_analysis, dict) and "error" in crash_analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=crash_analysis["error"]
            )
        
        return sanitize_floats(crash_analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting crash analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crash analysis: {str(e)}"
        )


@router.get("/{simulation_id}/shap-visualization")
async def get_shap_visualization(simulation_id: int, db: Session = Depends(get_db)):
    """
    Generate SHAP visualization for portfolio recommendations.
    Works with both enhanced and standard simulators.
    """
    try:
        logger.info(f"🔍 Generating SHAP visualization for simulation {simulation_id}")
        
        visualization_path = await generate_shap_visualization(simulation_id, db)
        
        if visualization_path is None:
            if MODULAR_SIMULATOR_AVAILABLE:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="SHAP visualization not available for this simulation"
                )
            else:
                return {
                    "simulation_id": simulation_id,
                    "message": "SHAP visualizations require the enhanced modular simulator",
                    "status": "not_available_in_standard_mode"
                }
        
        return {
            "simulation_id": simulation_id,
            "visualization_path": visualization_path,
            "message": "SHAP visualization generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating SHAP visualization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SHAP visualization: {str(e)}"
        )


@router.get("/health/enhanced-features")
async def check_enhanced_features():
    """
    Check the availability of all enhanced features.
    """
    try:
        # Check WealthWise availability
        wealthwise_available = False
        try:
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            wealthwise_available = True
        except ImportError:
            pass
        
        # Check news analysis availability
        news_analysis_available = False
        try:
            from services.news_analysis import NewsAnalysisService
            import os
            if os.getenv("FINNHUB_API_KEY"):
                news_analysis_available = True
        except ImportError:
            pass
        
        # Check AI analysis availability
        ai_analysis_available = False
        try:
            ai_service_test = AIAnalysisService()
            ai_analysis_available = True
        except Exception:
            pass
        
        feature_status = {
            "modular_portfolio_simulator": MODULAR_SIMULATOR_AVAILABLE,
            "wealthwise_shap_system": wealthwise_available,
            "news_analysis_service": news_analysis_available,
            "market_crash_detection": MODULAR_SIMULATOR_AVAILABLE,
            "enhanced_ai_summaries": ai_analysis_available,
            "smart_goal_calculation": True,  # Always available
            "robust_serialization": MODULAR_SIMULATOR_AVAILABLE,
            "status": "fully_enhanced" if MODULAR_SIMULATOR_AVAILABLE else "standard_with_fallbacks",
            "simulator_type": "Enhanced Modular" if MODULAR_SIMULATOR_AVAILABLE else "Standard",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return feature_status
        
    except Exception as e:
        logger.error(f"❌ Error checking enhanced features: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# 🛠️ Helper Functions

def _generate_basic_goal_analysis(onboarding_data: schemas.OnboardingCreate) -> dict:
    """Generate basic goal analysis when enhanced version isn't available."""
    
    target_value = onboarding_data.target_value
    lump_sum = onboarding_data.lump_sum
    monthly = onboarding_data.monthly
    timeframe = onboarding_data.timeframe
    
    # Calculate total contributions
    total_contributions = lump_sum + (monthly * 12 * timeframe)
    
    if total_contributions >= target_value:
        return {
            "required_return_percent": 4.0,  # Minimum to beat inflation
            "can_reach_with_contributions": True,
            "feasibility_rating": 5.0,
            "message": "Good news! Your contributions alone will reach your goal. We're targeting 4% growth to beat inflation.",
            "calculation_type": "contributions_sufficient"
        }
    else:
        # Simple calculation for required return
        if lump_sum > 0:
            required_return = ((target_value / lump_sum) ** (1/timeframe) - 1) * 100
        else:
            required_return = 7.0  # Default reasonable assumption
        
        return {
            "required_return_percent": round(required_return, 1),
            "can_reach_with_contributions": False,
            "feasibility_rating": 4.0 if required_return <= 10 else 3.0,
            "message": f"You need approximately {required_return:.1f}% annual returns to reach your goal.",
            "calculation_type": "growth_required"
        }