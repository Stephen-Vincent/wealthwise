from datetime import datetime
from fastapi import APIRouter, status, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
import math
import logging

from database import schemas
from database.db import get_db 
from services.risk_assessor import calculate_user_risk, calculate_user_risk_legacy
# 🎯 UPDATED IMPORT: Now using the enhanced modular portfolio simulator
from portfolio_simulator.main import simulate_portfolio, get_simulation_crash_analysis, generate_shap_visualization
from services.ai_analysis import AIAnalysisService

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
Enhanced Onboarding Endpoint with Modular Portfolio Simulator

Expected input (schemas.OnboardingCreate):
{
    "years_of_experience": int,       # e.g., 4
    "loss_tolerance": str,            # e.g., "wait_and_see"
    "panic_behavior": str,            # e.g., "no_never" 
    "financial_behavior": str,        # e.g., "invest_all"
    "engagement_level": str,          # e.g., "monthly"
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

Enhanced Workflow:
1. Accept onboarding data from frontend
2. Calculate comprehensive risk profile
3. Run enhanced portfolio simulation with:
   - Smart goal analysis (fixes 0% return issue)
   - Market crash detection with news analysis
   - SHAP explanations (when available)
   - Enhanced AI summaries
4. Return comprehensive results with enhanced features
"""

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"🚀 Processing enhanced onboarding for user {onboarding_data.user_id}")
        
        # Step 1: Calculate comprehensive risk profile from onboarding input
        risk_profile = calculate_user_risk(onboarding_data)
        
        logger.info(f"⚖️ Risk assessment completed: score={risk_profile['risk_score']}, level={risk_profile['risk_level']}")

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

        # Step 3: Merge risk data with onboarding input for enhanced simulation
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = legacy_risk_label  # Use legacy format for simulation
        simulation_input["detailed_risk_profile"] = risk_profile  # Include full profile

        # 🎯 ENHANCED STEP 4: Run enhanced portfolio simulation with all new features
        logger.info("📈 Running enhanced portfolio simulation with crash analysis and SHAP explanations")
        simulation_result = await simulate_portfolio(simulation_input, db)
        
        logger.info(f"✅ Enhanced portfolio simulation completed for user {onboarding_data.user_id}")
        
        # Check if enhanced features are available
        enhanced_features = {
            "has_crash_analysis": simulation_result.get("has_crash_analysis", False),
            "has_shap_explanations": simulation_result.get("has_shap_explanations", False),
            "wealthwise_enhanced": simulation_result.get("wealthwise_enhanced", False),
            "methodology": simulation_result.get("methodology", "Standard simulation")
        }
        
        if enhanced_features["has_crash_analysis"]:
            logger.info("📉 Market crash analysis included in results")
        
        if enhanced_features["has_shap_explanations"]:
            logger.info("🔍 SHAP explanations available for portfolio recommendations")

        # Step 5: Extract enhanced results data
        results = simulation_result.get("results", {})
        goal_analysis = results.get("goal_analysis", {})
        market_crash_analysis = results.get("market_crash_analysis", {})
        
        # Step 6: Construct enhanced response payload with new features
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
            "risk_label": risk_label,  # New detailed risk level
            "legacy_risk_label": legacy_risk_label,  # Backward compatibility
            "risk_description": risk_profile["risk_description"],
            "allocation_guidance": risk_profile["allocation_guidance"],
            "recommended_stock_allocation": risk_profile["recommended_stock_allocation"],
            "recommended_bond_allocation": risk_profile["recommended_bond_allocation"],
            "risk_explanation": risk_profile["explanation"],
            
            # 🎯 NEW: Enhanced simulation features
            "enhanced_features": enhanced_features,
            
            # 🎯 NEW: Smart goal analysis (fixes 0% return issue)
            "goal_analysis": {
                "required_return_percent": goal_analysis.get("required_return_percent"),
                "can_reach_with_contributions": goal_analysis.get("can_reach_with_contributions"),
                "feasibility_rating": goal_analysis.get("feasibility_rating"),
                "message": goal_analysis.get("message"),
                "calculation_type": goal_analysis.get("calculation_type")
            } if goal_analysis else None,
            
            # 🎯 NEW: Market crash analysis summary
            "market_events": {
                "crashes_detected": market_crash_analysis.get("crashes_detected", 0),
                "overall_message": market_crash_analysis.get("overall_message"),
                "key_insights": market_crash_analysis.get("key_insights", [])[:3],  # Top 3 insights
                "educational_summary": market_crash_analysis.get("educational_summary")
            } if market_crash_analysis else None,
            
            # Include all simulation results (including enhanced data)
            **simulation_result,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"🎉 Enhanced onboarding completed successfully for user {onboarding_data.user_id}")

        # Step 7: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"❌ Enhanced onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced onboarding processing failed: {str(e)}"
        )


@router.post("/legacy", status_code=status.HTTP_201_CREATED)
async def create_onboarding_legacy(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    """
    Legacy endpoint that returns the old format for backward compatibility.
    
    Note: This still uses the enhanced portfolio simulator under the hood,
    but returns results in the original format for backward compatibility.
    """
    try:
        logger.info(f"🔄 Processing legacy onboarding for user {onboarding_data.user_id}")
        
        # Use the legacy function that's already imported at the top
        risk_score, risk_label = calculate_user_risk_legacy(onboarding_data)

        # Step 2: Merge risk data with onboarding input
        simulation_input = onboarding_data.dict()
        simulation_input["risk_score"] = risk_score
        simulation_input["risk_label"] = risk_label

        # Step 3: Run enhanced portfolio simulation (but return legacy format)
        simulation_result = await simulate_portfolio(simulation_input, db)

        # Step 4: Construct response payload (original legacy format)
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

        # Step 5: Sanitize float values to remove NaN or Infinity, ensuring JSON compatibility
        return sanitize_floats(response_payload)

    except Exception as e:
        logger.error(f"❌ Legacy onboarding failed for user {onboarding_data.user_id}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Legacy onboarding processing failed: {str(e)}"
        )


# 🎯 NEW: Additional endpoints for enhanced features
@router.get("/{simulation_id}/crash-analysis")
async def get_crash_analysis(simulation_id: int, db: Session = Depends(get_db)):
    """
    Get detailed crash analysis for a specific simulation.
    
    Returns comprehensive market crash analysis including:
    - Crash timeline and severity
    - News analysis for crash periods
    - Educational insights about market volatility
    - Recovery patterns and lessons learned
    """
    try:
        logger.info(f"📉 Getting crash analysis for simulation {simulation_id}")
        
        crash_analysis = await get_simulation_crash_analysis(simulation_id, db)
        
        if "error" in crash_analysis:
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
    
    Returns path to generated visualization showing:
    - Why specific stocks were recommended
    - Factor importance in the AI decision
    - Transparent explanation of portfolio construction
    """
    try:
        logger.info(f"🔍 Generating SHAP visualization for simulation {simulation_id}")
        
        visualization_path = await generate_shap_visualization(simulation_id, db)
        
        if visualization_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="SHAP visualization not available for this simulation"
            )
        
        return {
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
    Check the availability of enhanced features.
    
    Returns status of:
    - WealthWise SHAP system
    - News analysis service
    - Market crash detection
    - Enhanced AI summaries
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
            "enhanced_portfolio_simulator": True,  # Always available now
            "wealthwise_shap_system": wealthwise_available,
            "news_analysis_service": news_analysis_available,
            "market_crash_detection": True,  # Always available
            "enhanced_ai_summaries": ai_analysis_available,
            "smart_goal_calculation": True,  # Always available
            "robust_serialization": True,  # Always available
            "status": "healthy",
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