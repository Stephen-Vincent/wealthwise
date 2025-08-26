# backend/tests/test_database_service.py
"""
Simple tests for DatabaseService and SimulationResultsFormatter
using an in-memory SQLite DB with a patched database.models.
"""

import sys
import types
import pytest
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# --- Minimal Simulation model just for testing ---
Base = declarative_base()

class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True)
    name = Column(String, default="Simulation")
    goal = Column(String, default="wealth building")
    target_value = Column(Float, default=0)
    lump_sum = Column(Float, default=0)
    monthly = Column(Float, default=0)
    timeframe = Column(Integer, default=0)
    target_achieved = Column(Boolean, default=False)
    income_bracket = Column(String, default="medium")
    risk_score = Column(Integer, default=0)
    risk_label = Column(String, default="Moderate")
    ai_summary = Column(String, default="")
    results = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Patch a fake `database.models` before importing the service ---
fake_database = types.ModuleType("database")
fake_models = types.ModuleType("models")
fake_models.Simulation = Simulation
fake_database.models = fake_models
sys.modules["database"] = fake_database
sys.modules["database.models"] = fake_models

# Now safe to import the real service
from backend.services.portfolio_simulator.database_service import (
    DatabaseService,
    SimulationResultsFormatter,
)


@pytest.fixture(scope="function")
def db_session():
    """Provide a fresh in-memory database session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_save_and_get_simulation(db_session):
    service = DatabaseService()

    input_payload = {
        "goal": "retirement",
        "target_value": 100000,
        "lump_sum": 5000,
        "monthly": 500,
        "timeframe": 10,
        "income_bracket": "high",
    }
    result_payload = {
        "target_achieved": True,
        "portfolio_metrics": {"risk_score": 65, "ending_value": 150000},
    }

    sim = service.save_simulation(
        db_session,
        user_id=1,
        name="Test Simulation",
        input_payload=input_payload,
        result_payload=result_payload,
    )

    assert sim.id is not None
    assert sim.target_value == 100000
    assert sim.risk_score == 65

    fetched = service.get_simulation(db_session, sim.id)
    assert fetched.id == sim.id
    assert fetched.goal == "retirement"


def test_formatter_output(db_session):
    service = DatabaseService()
    formatter = SimulationResultsFormatter()

    sim = service.save_simulation(
        db_session,
        user_id=2,
        name="Formatter Test",
        input_payload={"goal": "growth", "target_value": 200000},
        result_payload={
            "target_achieved": False,
            "portfolio_metrics": {"risk_score": 40, "ending_value": 180000},
            "stocks_picked": [{"symbol": "AAPL", "allocation": 0.5}],
        },
    )

    response = formatter.format_simulation_response(sim)
    assert response["id"] == sim.id
    assert "timeline" in response  # always normalized
    assert isinstance(response["breakdown"], dict)
    assert response["breakdown"].get("AAPL") == 0.5