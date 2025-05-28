from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Adjust the path to match your actual DB location
DATABASE_URL = "sqlite:///./backend/database/onboarding.db"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)