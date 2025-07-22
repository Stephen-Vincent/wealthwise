from datetime import datetime
import os  # ← ADD THIS LINE
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from .database import Base  # Make sure this import is correct too


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # Added length
    email = Column(String(320), unique=True, index=True, nullable=False)  # Standard email length
    hashed_password = Column(String(255), nullable=False)  # Added length
    created_at = Column(DateTime, default=datetime.utcnow)
    simulations = relationship("Simulation", back_populates="user", cascade="all, delete-orphan")

class Simulation(Base):
    __tablename__ = "simulations"
    # Updated for PostgreSQL compatibility
    __table_args__ = {"postgresql_sequence_name": "simulations_id_seq"} if os.getenv("DATABASE_URL") else {"sqlite_autoincrement": True}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=True)  # Added length
    goal = Column(String(100), nullable=False)  # Added length
    risk_score = Column(Integer, nullable=True, default=50)
    risk_label = Column(String(50), nullable=True)  # Added length
    target_value = Column(Float, nullable=True)
    lump_sum = Column(Float, nullable=True)
    monthly = Column(Float, nullable=True)
    timeframe = Column(String(20), nullable=False)  # Added length
    target_achieved = Column(Boolean, default=False, nullable=False)
    income_bracket = Column(String(50), nullable=False)  # Added length
    results = Column(JSON, nullable=True)  # Works in both databases!
    ai_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="simulations")