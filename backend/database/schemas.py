from datetime import datetime
from pydantic import BaseModel, EmailStr, model_validator, Field
from typing import Optional, Dict, Any

# User schemas
class UserCreate(BaseModel):
    name: str                    
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr

    class Config:
        from_attributes = True

# Login schema
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Token schemas
class UserInfo(BaseModel):
    id: int
    email: str
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserInfo

class TokenData(BaseModel):
    user_id: int

# Onboarding schemas
class OnboardingCreate(BaseModel):
    years_of_experience: int
    goal: str
    target_value: float
    lump_sum: Optional[float] = None
    monthly: Optional[float] = None
    timeframe: int
    income_bracket: str
    consent: bool
    name: Optional[str] = Field(default=None, alias="name")
    user_id: Optional[int] = Field(default=None, alias="user_id")
    
    @model_validator(mode="after")
    def at_least_one_investment(self):
        if self.lump_sum is None and self.monthly is None:
            raise ValueError("At least one of lump_sum or monthly must be provided.")
        return self

    class Config:
        populate_by_name = True

# Simulation Response schema with AI-generated summary
class SimulationResponse(BaseModel):
    id: int
    user_id: int
    name: Optional[str] = None
    goal: str
    target_value: Optional[float] = None
    lump_sum: Optional[float] = None
    monthly: Optional[float] = None
    timeframe: int
    target_achieved: bool
    income_bracket: str
    risk_score: int
    risk_label: str
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    ai_summary: Optional[str] = None  # AI-generated summary of the simulation

    class Config:
        from_attributes = True
