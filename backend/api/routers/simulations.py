from typing import List, Optional
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from database.db import get_db 
from database import models, schemas
from core.security import get_current_user

# Try to import enhanced portfolio simulator for additional features
try:
    from services.portfolio_simulator import get_simulation_charts
    ENHANCED_SIMULATOR_AVAILABLE = True
except ImportError:
    ENHANCED_SIMULATOR_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=List[schemas.SimulationResponse])
def list_simulations(
    skip: int = Query(0, ge=0, description="Number of simulations to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of simulations to return"),
    goal_filter: Optional[str] = Query(None, description="Filter by investment goal"),
    risk_label_filter: Optional[str] = Query(None, description="Filter by risk label"),
    target_achieved_filter: Optional[bool] = Query(None, description="Filter by target achievement"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    List simulations for the current user with pagination and filtering.
    """
    try:
        # Build query with filters
        query = db.query(models.Simulation).filter(
            models.Simulation.user_id == current_user.id
        )
        
        # Apply filters
        if goal_filter:
            query = query.filter(models.Simulation.goal.ilike(f"%{goal_filter}%"))
        
        if risk_label_filter:
            query = query.filter(models.Simulation.risk_label == risk_label_filter)
        
        if target_achieved_filter is not None:
            query = query.filter(models.Simulation.target_achieved == target_achieved_filter)
        
        # Apply pagination and ordering
        sims = query.order_by(models.Simulation.created_at.desc()).offset(skip).limit(limit).all()
        
        logger.info(f"Retrieved {len(sims)} simulations for user {current_user.id}")
        return sims
        
    except Exception as e:
        logger.error(f"Error listing simulations for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve simulations"
        )

@router.get("/me", response_model=schemas.UserResponse)
def get_current_user_info(current_user: models.User = Depends(get_current_user)):
    """
    Get the current logged-in user's information.
    """
    return current_user

@router.get("/statistics")
def get_user_statistics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get statistics about the user's simulations.
    """
    try:
        # Get basic statistics
        total_simulations = db.query(models.Simulation).filter(
            models.Simulation.user_id == current_user.id
        ).count()
        
        successful_simulations = db.query(models.Simulation).filter(
            models.Simulation.user_id == current_user.id,
            models.Simulation.target_achieved == True
        ).count()
        
        # Calculate averages
        simulations = db.query(models.Simulation).filter(
            models.Simulation.user_id == current_user.id
        ).all()
        
        if simulations:
            avg_target_value = sum(sim.target_value for sim in simulations) / len(simulations)
            avg_timeframe = sum(sim.timeframe for sim in simulations) / len(simulations)
            success_rate = (successful_simulations / total_simulations) * 100
        else:
            avg_target_value = 0
            avg_timeframe = 0
            success_rate = 0
        
        # Get risk distribution
        risk_distribution = {}
        for sim in simulations:
            risk_label = sim.risk_label or "Unknown"
            risk_distribution[risk_label] = risk_distribution.get(risk_label, 0) + 1
        
        return {
            "total_simulations": total_simulations,
            "successful_simulations": successful_simulations,
            "success_rate": round(success_rate, 1),
            "average_target_value": round(avg_target_value, 2),
            "average_timeframe": round(avg_timeframe, 1),
            "risk_distribution": risk_distribution,
            "enhanced_features_available": ENHANCED_SIMULATOR_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )

@router.get("/{simulation_id}", response_model=schemas.SimulationResponse)
async def get_simulation(  # ← Changed to async
    simulation_id: int,
    include_chart_data: bool = Query(False, description="Include chart data in response"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get a specific simulation by ID.
    """
    try:
        sim = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id,
            models.Simulation.user_id == current_user.id
        ).first()
        
        if not sim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Simulation not found"
            )
        
        # If chart data is requested and enhanced simulator is available
        if include_chart_data and ENHANCED_SIMULATOR_AVAILABLE:
            try:
                chart_data = await get_simulation_charts(simulation_id, db)
                # Add chart data to the response
                sim_dict = sim.__dict__.copy()
                sim_dict['chart_data'] = chart_data.get('chart_data', {})
                return sim_dict
            except Exception as e:
                logger.warning(f"Failed to get chart data for simulation {simulation_id}: {e}")
                # Return simulation without chart data
        
        return sim
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation {simulation_id} for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve simulation"
        )

@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_simulation(
    simulation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Delete a specific simulation.
    """
    try:
        sim = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id,
            models.Simulation.user_id == current_user.id
        ).first()
        
        if not sim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Simulation not found"
            )
        
        # Log the deletion
        logger.info(f"Deleting simulation {simulation_id} for user {current_user.id}")
        
        db.delete(sim)
        db.commit()
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting simulation {simulation_id} for user {current_user.id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete simulation"
        )

@router.patch("/{simulation_id}", response_model=schemas.SimulationResponse)
def update_simulation(
    simulation_id: int,
    update_data: schemas.SimulationUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Update a specific simulation (e.g., change name or goal).
    """
    try:
        sim = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id,
            models.Simulation.user_id == current_user.id
        ).first()
        
        if not sim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Simulation not found"
            )
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            if hasattr(sim, field):
                setattr(sim, field, value)
        
        db.commit()
        db.refresh(sim)
        
        logger.info(f"Updated simulation {simulation_id} for user {current_user.id}")
        return sim
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating simulation {simulation_id} for user {current_user.id}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update simulation"
        )

@router.get("/{simulation_id}/enhanced-data")
async def get_simulation_enhanced_data(  # ← Changed to async
    simulation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get enhanced data for a simulation (chart data, SHAP explanations, etc.).
    """
    try:
        # Verify ownership
        sim = db.query(models.Simulation).filter(
            models.Simulation.id == simulation_id,
            models.Simulation.user_id == current_user.id
        ).first()
        
        if not sim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Simulation not found"
            )
        
        if not ENHANCED_SIMULATOR_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Enhanced features not available"
            )
        
        # Get enhanced data
        enhanced_data = await get_simulation_charts(simulation_id, db)
        return enhanced_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enhanced data for simulation {simulation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve enhanced data"
        )