# core/config.py

import os
from typing import List, Union, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "WealthWise API"
    APP_VERSION: str = "2.0.0"
    
    # JWT Settings
    JWT_SECRET_KEY: str = Field(default="your-secret-key")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./wealthwise.db")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # This should ignore extra fields
        case_sensitive = False  # Allow case variations
        validate_assignment = False
        
    @field_validator('BACKEND_CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return ["http://localhost:5173", "http://localhost:3000"]
        return v

    def __init__(self, **kwargs):
        # Filter out problematic keys before calling super().__init__
        filtered_kwargs = {}
        allowed_fields = {
            'APP_NAME', 'APP_VERSION', 'JWT_SECRET_KEY', 'ALGORITHM', 
            'ACCESS_TOKEN_EXPIRE_MINUTES', 'DATABASE_URL', 'BACKEND_CORS_ORIGINS',
            'jwt_secret_key', 'algorithm', 'access_token_expire_minutes', 
            'database_url', 'backend_cors_origins'
        }
        
        for key, value in kwargs.items():
            if key.lower() in [f.lower() for f in allowed_fields]:
                filtered_kwargs[key] = value
        
        super().__init__(**filtered_kwargs)

# Create settings instance with direct environment variable reading
def create_settings():
    """Create settings instance with safe environment variable handling"""
    return Settings(
        JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "your-secret-key"),
        ALGORITHM=os.getenv("ALGORITHM", "HS256"),
        ACCESS_TOKEN_EXPIRE_MINUTES=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
        DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///./wealthwise.db"),
        BACKEND_CORS_ORIGINS=os.getenv("BACKEND_CORS_ORIGINS", '["http://localhost:5173", "http://localhost:3000"]')
    )

settings = create_settings()