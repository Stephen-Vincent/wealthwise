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

# Enhanced Onboarding schemas with behavioral risk assessment
class OnboardingCreate(BaseModel):
    # Basic investment information
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
    
    # New behavioral risk assessment fields
    loss_tolerance: Optional[str] = Field(
        default="wait_and_see",
        description="How user reacts to investment losses: sell_immediately, wait_and_see, buy_more"
    )
    panic_behavior: Optional[str] = Field(
        default="no_experience", 
        description="Past behavior during market crashes: yes_always, yes_sometimes, no_never, no_experience"
    )
    financial_behavior: Optional[str] = Field(
        default="save_half",
        description="What user does with bonus money: invest_all, save_half, save_all, spend_it"
    )
    engagement_level: Optional[str] = Field(
        default="monthly",
        description="How often user reviews investments: daily, weekly, monthly, quarterly, rarely"
    )
    
    @model_validator(mode="after")
    def validate_onboarding_data(self):
        # Validate at least one investment amount
        if self.lump_sum is None and self.monthly is None:
            raise ValueError("At least one of lump_sum or monthly must be provided.")
        
        # Validate behavioral fields have valid values
        valid_loss_tolerance = ["sell_immediately", "wait_and_see", "buy_more"]
        valid_panic_behavior = ["yes_always", "yes_sometimes", "no_never", "no_experience"]
        valid_financial_behavior = ["invest_all", "save_half", "save_all", "spend_it"]
        valid_engagement_level = ["daily", "weekly", "monthly", "quarterly", "rarely"]
        valid_goals = ["buy a house", "vacation", "emergency fund", "retirement", "save for a car", "wealth building"]
        valid_income_brackets = ["low", "medium", "high"]
        
        if self.loss_tolerance not in valid_loss_tolerance:
            raise ValueError(f"loss_tolerance must be one of: {valid_loss_tolerance}")
        
        if self.panic_behavior not in valid_panic_behavior:
            raise ValueError(f"panic_behavior must be one of: {valid_panic_behavior}")
            
        if self.financial_behavior not in valid_financial_behavior:
            raise ValueError(f"financial_behavior must be one of: {valid_financial_behavior}")
            
        if self.engagement_level not in valid_engagement_level:
            raise ValueError(f"engagement_level must be one of: {valid_engagement_level}")
            
        if self.goal not in valid_goals:
            raise ValueError(f"goal must be one of: {valid_goals}")
            
        if self.income_bracket not in valid_income_brackets:
            raise ValueError(f"income_bracket must be one of: {valid_income_brackets}")
        
        return self

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "years_of_experience": 5,
                "loss_tolerance": "wait_and_see",
                "panic_behavior": "no_never",
                "financial_behavior": "invest_all", 
                "engagement_level": "monthly",
                "goal": "retirement",
                "target_value": 100000.0,
                "lump_sum": 10000.0,
                "monthly": 500.0,
                "timeframe": 15,
                "income_bracket": "medium",
                "consent": True,
                "name": "John Doe",
                "user_id": 1
            }
        }

# Enhanced Risk Profile schema for detailed risk information
class RiskProfile(BaseModel):
    risk_score: float = Field(description="Numerical risk score from 1-100")
    risk_level: str = Field(description="Detailed risk level (e.g., 'Moderate Aggressive')")
    risk_description: str = Field(description="Human-readable risk description")
    allocation_guidance: str = Field(description="Investment allocation guidance")
    recommended_stock_allocation: int = Field(description="Recommended percentage in stocks")
    recommended_bond_allocation: int = Field(description="Recommended percentage in bonds")
    explanation: str = Field(description="Personalized explanation of the risk assessment")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 67.3,
                "risk_level": "Moderate Aggressive",
                "risk_description": "Growth Oriented",
                "allocation_guidance": "Stock-heavy portfolio with some bonds for diversification",
                "recommended_stock_allocation": 75,
                "recommended_bond_allocation": 25,
                "explanation": "Your experience allows for more sophisticated strategies. Long-term retirement goals allow for more growth-oriented approaches."
            }
        }

# Enhanced Simulation Response schema with comprehensive risk information
class SimulationResponse(BaseModel):
    # Basic simulation information
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
    
    # Enhanced risk information
    risk_score: float = Field(description="Numerical risk score from 1-100")
    risk_label: str = Field(description="Detailed risk level")
    legacy_risk_label: Optional[str] = Field(default=None, description="Legacy Low/Medium/High format")
    risk_description: Optional[str] = Field(default=None, description="Risk description")
    allocation_guidance: Optional[str] = Field(default=None, description="Allocation guidance")
    recommended_stock_allocation: Optional[int] = Field(default=None, description="Recommended stock %")
    recommended_bond_allocation: Optional[int] = Field(default=None, description="Recommended bond %")
    risk_explanation: Optional[str] = Field(default=None, description="Personalized risk explanation")
    
    # Simulation results and metadata
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    ai_summary: Optional[str] = None  # AI-generated summary of the simulation

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "name": "John Doe",
                "goal": "retirement",
                "target_value": 100000.0,
                "lump_sum": 10000.0,
                "monthly": 500.0,
                "timeframe": 15,
                "target_achieved": True,
                "income_bracket": "medium",
                "risk_score": 67.3,
                "risk_label": "Moderate Aggressive",
                "legacy_risk_label": "Medium",
                "risk_description": "Growth Oriented",
                "allocation_guidance": "Stock-heavy portfolio with some bonds",
                "recommended_stock_allocation": 75,
                "recommended_bond_allocation": 25,
                "risk_explanation": "Your experience allows for sophisticated strategies...",
                "results": {"target_reached": True},
                "created_at": "2025-01-07T12:00:00Z",
                "ai_summary": "Based on your moderate aggressive risk profile..."
            }
        }

# Legacy simulation response for backward compatibility
class LegacySimulationResponse(BaseModel):
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
    risk_score: int  # Integer for legacy compatibility
    risk_label: str  # Simple Low/Medium/High format
    results: Optional[Dict[str, Any]] = None
    created_at: datetime
    ai_summary: Optional[str] = None

    class Config:
        from_attributes = True

# Schema for updating existing simulations with enhanced risk data
class SimulationUpdate(BaseModel):
    risk_description: Optional[str] = None
    allocation_guidance: Optional[str] = None
    recommended_stock_allocation: Optional[int] = None
    recommended_bond_allocation: Optional[int] = None
    risk_explanation: Optional[str] = None
    ai_summary: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "risk_description": "Growth Oriented",
                "allocation_guidance": "Stock-heavy portfolio with some bonds",
                "recommended_stock_allocation": 75,
                "recommended_bond_allocation": 25,
                "risk_explanation": "Your experience allows for sophisticated strategies...",
                "ai_summary": "Updated AI summary..."
            }
        }

# Validation helper schemas
class BehavioralRiskInput(BaseModel):
    """Schema for validating behavioral risk assessment inputs"""
    loss_tolerance: str = Field(description="sell_immediately | wait_and_see | buy_more")
    panic_behavior: str = Field(description="yes_always | yes_sometimes | no_never | no_experience")
    financial_behavior: str = Field(description="invest_all | save_half | save_all | spend_it")
    engagement_level: str = Field(description="daily | weekly | monthly | quarterly | rarely")

    class Config:
        json_schema_extra = {
            "example": {
                "loss_tolerance": "wait_and_see",
                "panic_behavior": "no_never", 
                "financial_behavior": "invest_all",
                "engagement_level": "monthly"
            }
        }