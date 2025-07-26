# api/routers/password_reset.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator
from services.password_reset_service import PasswordResetService
from database.db import get_db
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["password-reset"])  # REMOVE prefix="/auth" - it's added in main.py

# Initialize password reset service
password_reset_service = PasswordResetService()

# Pydantic models for request/response
class PasswordResetRequest(BaseModel):
    email: EmailStr
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
    confirm_password: str
    
    @field_validator('confirm_password')
    def passwords_match(cls, v, info):
        if 'new_password' in info.data and v != info.data['new_password']:
            raise ValueError('Passwords do not match')
        return v
    
    @field_validator('new_password')
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)
        
        if not all([has_upper, has_lower, has_digit, has_special]):
            raise ValueError('Password must contain uppercase, lowercase, number, and special character')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "token": "reset_token_here",
                "new_password": "NewPassword123!",
                "confirm_password": "NewPassword123!"
            }
        }

class TokenVerificationRequest(BaseModel):
    token: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "token": "reset_token_here"
            }
        }

# Response models
class StandardResponse(BaseModel):
    success: bool
    message: str
    timestamp: str = datetime.now().isoformat()

class TokenVerificationResponse(BaseModel):
    valid: bool
    message: str
    email: str = None
    expires_at: str = None
    timestamp: str = datetime.now().isoformat()

@router.post("/request-password-reset", response_model=StandardResponse)
async def request_password_reset(
    request: PasswordResetRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Request a password reset email
    
    This endpoint will always return success to prevent email enumeration attacks.
    If the email exists, a reset email will be sent.
    """
    try:
        # Process reset request in background
        background_tasks.add_task(
            password_reset_service.initiate_password_reset,
            request.email,
            db
        )
        
        # Always return success for security (prevents email enumeration)
        return StandardResponse(
            success=True,
            message="If an account with that email exists, you will receive a password reset link shortly."
        )
        
    except Exception as e:
        logger.error(f"Error in password reset request: {str(e)}")
        # Still return success to prevent information leakage
        return StandardResponse(
            success=True,
            message="If an account with that email exists, you will receive a password reset link shortly."
        )

@router.post("/verify-reset-token", response_model=TokenVerificationResponse)
async def verify_reset_token(
    request: TokenVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    Verify if a password reset token is valid
    
    Use this endpoint to check token validity before showing the reset form
    """
    try:
        result = password_reset_service.verify_reset_token(request.token, db)
        
        return TokenVerificationResponse(
            valid=result["valid"],
            message=result.get("message", ""),
            email=result.get("email"),
            expires_at=result.get("expires_at")
        )
        
    except Exception as e:
        logger.error(f"Error verifying reset token: {str(e)}")
        return TokenVerificationResponse(
            valid=False,
            message="Error verifying token"
        )

@router.post("/reset-password", response_model=StandardResponse)
async def reset_password(
    request: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """
    Reset password using a valid token
    
    This completes the password reset process
    """
    try:
        result = password_reset_service.reset_password(
            request.token,
            request.new_password,
            db
        )
        
        if result["success"]:
            return StandardResponse(
                success=True,
                message=result["message"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while resetting password")

@router.get("/password-requirements")
async def get_password_requirements():
    """
    Get password requirements for the frontend
    """
    return {
        "requirements": {
            "min_length": 8,
            "requires_uppercase": True,
            "requires_lowercase": True,
            "requires_number": True,
            "requires_special_character": True,
            "allowed_special_characters": "!@#$%^&*()_+-=[]{}|;:,.<>?"
        },
        "description": "Password must be at least 8 characters long and contain uppercase, lowercase, number, and special character"
    }