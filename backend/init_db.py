# File: backend/init_db.py (create this)
from database import engine, Base
from models import User, Portfolio, Stock  # Import all your models
import logging

logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables"""
    try:
        # This works for both SQLite and PostgreSQL
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False

def init_database():
    """Initialize database with required tables"""
    logger.info("🗄️  Initializing database...")
    
    if create_tables():
        logger.info("🎉 Database initialization complete!")
    else:
        logger.error("💥 Database initialization failed!")

if __name__ == "__main__":
    init_database()