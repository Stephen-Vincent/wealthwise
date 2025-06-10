from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from database.session import SessionLocal
from fastapi.security import OAuth2PasswordBearer
from core.security import get_current_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a SQLAlchemy session and ensures it is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()