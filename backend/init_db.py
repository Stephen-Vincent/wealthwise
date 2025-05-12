# backend/init_db.py
from database.database import init_db
from database import models  # <-- This is essential

init_db()
print("✅ Database initialized successfully.")