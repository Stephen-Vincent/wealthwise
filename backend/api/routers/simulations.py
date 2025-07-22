from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.database import get_db
from database import models, schemas
from core.security import get_current_user

router = APIRouter()

@router.get("/", response_model=List[schemas.SimulationResponse])
def list_simulations(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    List all simulations for the current user.
    """
    sims = db.query(models.Simulation).filter(models.Simulation.user_id == current_user.id).all()
    return sims

# Route to get the current logged-in user's information
@router.get("/me", response_model=schemas.UserResponse)
def get_current_user_info(current_user: models.User = Depends(get_current_user)):
    """
    Get the current logged-in user's information.
    """
    return current_user

@router.get("/{simulation_id}", response_model=schemas.SimulationResponse)
def get_simulation(
    simulation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    sim = db.query(models.Simulation).filter(
        models.Simulation.id == simulation_id,
        models.Simulation.user_id == current_user.id
    ).first()
    if not sim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found")
    return sim

@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_simulation(
    simulation_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    sim = db.query(models.Simulation).filter(
        models.Simulation.id == simulation_id,
        models.Simulation.user_id == current_user.id
    ).first()
    if not sim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found")
    db.delete(sim)
    db.commit()
    return None
