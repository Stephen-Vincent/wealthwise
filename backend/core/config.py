# core/config.py

import os
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "WealthWise API"
    APP_VERSION: str = "2.0.0"
    
    # JWT Settings
    JWT_SECRET_KEY: str = Field(default="your-secret-key", env="JWT_SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./wealthwise.db", env="DATABASE_URL")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        env="BACKEND_CORS_ORIGINS"
    )
    
    # AI Configuration - ADD THESE NEW FIELDS
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    
    # Application Settings - ADD THESE NEW FIELDS  
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    SECRET_KEY: str = Field(default="your-secret-key", env="SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # IMPORTANT: Allow extra fields or ignore them
        extra = "ignore"  # This will ignore extra environment variables
        
        # Alternative: Use "allow" if you want to allow extra fields
        # extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse CORS origins if it's a string (from environment variable)
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            import json
            try:
                self.BACKEND_CORS_ORIGINS = json.loads(self.BACKEND_CORS_ORIGINS)
            except (json.JSONDecodeError, TypeError):
                # Fallback to default if parsing fails
                self.BACKEND_CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]

# Create settings instance
settings = Settings()