from datetime import datetime, timedelta
import os 
import secrets
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from .db import Base  


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # Added length
    email = Column(String(320), unique=True, index=True, nullable=False)  # Standard email length
    hashed_password = Column(String(255), nullable=False)  # Added length
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # Added for password reset tracking
    
    # Relationships
    simulations = relationship("Simulation", back_populates="user", cascade="all, delete-orphan")
    password_reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")

class Simulation(Base):
    __tablename__ = "simulations"
    # Updated for PostgreSQL compatibility
    __table_args__ = {"sqlite_autoincrement": True}
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=True)  # Added length
    goal = Column(String(100), nullable=False)  # Added length
    risk_score = Column(Integer, nullable=True, default=50)
    risk_label = Column(String(50), nullable=True)  # Added length
    legacy_risk_label = Column(String(50), nullable=True)
    target_value = Column(Float, nullable=True)
    lump_sum = Column(Float, nullable=True)
    monthly = Column(Float, nullable=True)
    timeframe = Column(Integer, nullable=False)
    target_achieved = Column(Boolean, default=False, nullable=False)
    income_bracket = Column(String(50), nullable=False)  # Added length
    # Optional richer explainability fields (kept nullable for backward compatibility)
    risk_description = Column(Text, nullable=True)
    risk_explanation = Column(Text, nullable=True)
    recommended_stock_allocation = Column(Float, nullable=True)
    recommended_bond_allocation = Column(Float, nullable=True)
    allocation_guidance = Column(JSON, nullable=True)  # can store structured guidance
    results = Column(JSON, nullable=True)  # Works in both databases!
    ai_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="simulations")

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(320), nullable=False, index=True)  # Match User email length
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="password_reset_tokens")
    
    @classmethod
    def create_token(cls, user_id: int, email: str, expiry_hours: int = 1):
        """Create a new password reset token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=expiry_hours)
        
        return cls(
            user_id=user_id,
            email=email.lower(),
            token=token,
            expires_at=expires_at
        )
    
    def is_valid(self) -> bool:
        """Check if token is still valid"""
        return (
            not self.used and 
            datetime.utcnow() < self.expires_at
        )
    
    def mark_as_used(self):
        """Mark token as used"""
        self.used = True
    
    def is_expired(self) -> bool:
        """Check if token has expired"""
        return datetime.utcnow() >= self.expires_at