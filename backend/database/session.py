# database/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

# Base class for all ORM models
Base = declarative_base()

from core.config import settings

# Adjust connect_args for SQLite
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

# Create the SQLAlchemy engine using the DATABASE_URL from settings
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=True,        # Log SQL statements for debugging
    future=True       # Use SQLAlchemy 2.0 style
)

# sessionmaker factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Dependency for FastAPI routes to get a DB session
def get_db():
    """
    Provide a transactional database session for a request and ensure it's closed.
    """
    db_session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
