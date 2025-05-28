from sqlalchemy import Column, Integer, String, Float
from .database import Base

class OnboardingSubmission(Base):
    __tablename__ = "onboarding_submissions"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    experience = Column(Integer)
    goal = Column(String)
    lump_sum = Column(Float)
    monthly = Column(Float)
    timeframe = Column(String)
    income_bracket = Column(String, nullable=True)
    risk = Column(String)
    risk_score = Column(Integer, nullable=True)
    consent = Column(String, nullable=True)
    target_value = Column(Float, nullable=True)