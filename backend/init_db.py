from database.database import engine
from database.models import Base  # Make sure this import exists

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully.")

if __name__ == "__main__":
    init_db()