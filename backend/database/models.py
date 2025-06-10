from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from .session import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    simulations = relationship("Simulation", back_populates="user", cascade="all, delete-orphan")

class Simulation(Base):
    __tablename__ = "simulations"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=True)
    goal = Column(String, nullable=False)
    risk_score = Column(Integer, nullable=True, default=50)
    risk_label = Column(String, nullable=True)
    target_value = Column(Float, nullable=True)
    lump_sum = Column(Float, nullable=True)
    monthly = Column(Float, nullable=True)
    timeframe = Column(String, nullable=False)
    target_achieved = Column(Boolean, default=False, nullable=False)
    income_bracket = Column(String, nullable=False)
    results = Column(JSON, nullable=True)
    ai_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="simulations")