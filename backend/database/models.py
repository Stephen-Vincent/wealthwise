from sqlalchemy import Column, Integer, String, Float
from database.database import Base

class OnboardingSubmission(Base):
    __tablename__ = "onboarding_submissions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    experience = Column(Integer)
    goal = Column(String)
    lump_sum = Column(Float)
    monthly = Column(Float)
    timeframe = Column(String)
    risk = Column(String)
    consent = Column(String, nullable=True)