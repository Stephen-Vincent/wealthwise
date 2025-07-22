# core/config.py

import os
import json
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Simple settings class that bypasses Pydantic completely"""
    
    def __init__(self):
        # App Info
        self.APP_NAME = "WealthWise API"
        self.APP_VERSION = "2.0.0"
        
        # JWT Settings - read directly from environment
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "university-project-secret-key-2025")
        self.ALGORITHM = os.getenv("ALGORITHM", "HS256")
        
        # Convert string to int for token expiry
        try:
            self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        except (ValueError, TypeError):
            self.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            
        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wealthwise.db")
        
        # CORS - handle JSON string from environment
        self.BACKEND_CORS_ORIGINS = self._parse_cors_origins()
        
        # Log what we loaded
        logger.info(f"🔧 Settings loaded - JWT Key: {self.JWT_SECRET_KEY[:10]}...")
        logger.info(f"🔧 Database: {self.DATABASE_URL}")
        logger.info(f"🔧 CORS Origins: {self.BACKEND_CORS_ORIGINS}")
    
    def _parse_cors_origins(self):
        """Safely parse CORS origins from environment"""
        cors_env = os.getenv("BACKEND_CORS_ORIGINS", '["http://localhost:5173", "http://localhost:3000"]')
        
        try:
            # Try to parse as JSON
            origins = json.loads(cors_env)
            if isinstance(origins, list):
                return origins
            else:
                logger.warning(f"CORS origins not a list: {origins}")
                return ["http://localhost:5173", "http://localhost:3000"]
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse CORS origins: {cors_env}")
            return ["http://localhost:5173", "http://localhost:3000"]

# Create settings instance
settings = Settings()

# Log successful initialization
logger.info("✅ Settings initialized successfully without Pydantic!")