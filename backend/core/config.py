from typing import List, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import json

class Settings(BaseSettings):
    # Load variables from a .env file
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
    )

    # Use SQLite by default; override with env var if needed
    DATABASE_URL: str = "sqlite:///./wealthwise.db"

    # Secret key for JWT
    JWT_SECRET_KEY: str

    # App info
    APP_NAME: str = "WealthWise API"
    APP_VERSION: str = "0.1.0"

    # Auth settings
    ALGORITHM: str = 'HS256'
    JWT_ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Server settings
    SERVER_NAME: str = 'WealthWise API'
    SERVER_HOST: AnyHttpUrl = 'http://localhost:8000'
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

# Instantiate settings so validation happens on startup
settings = Settings()
