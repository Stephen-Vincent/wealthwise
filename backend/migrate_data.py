# File: backend/migrate_data.py (if you want to preserve existing data)
import sqlite3
import json
from database import SessionLocal, engine
from models import User, Portfolio, Stock
from sqlalchemy.orm import Session

def migrate_sqlite_to_postgresql():
    """Migrate data from SQLite to PostgreSQL (optional)"""
    
    # Connect to your local SQLite database
    sqlite_conn = sqlite3.connect("wealthwise.db")
    sqlite_conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = sqlite_conn.cursor()
    
    # Create PostgreSQL session
    db: Session = SessionLocal()
    
    try:
        print("🔄 Starting data migration...")
        
        # Migrate users
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        
        for user_row in users:
            user = User(
                id=user_row["id"],
                name=user_row["name"],
                email=user_row["email"],
                hashed_password=user_row["hashed_password"],
                created_at=user_row["created_at"],
                is_active=user_row.get("is_active", True)
            )
            db.add(user)
        
        # Migrate portfolios
        cursor.execute("SELECT * FROM portfolios")
        portfolios = cursor.fetchall()
        
        for portfolio_row in portfolios:
            portfolio = Portfolio(
                id=portfolio_row["id"],
                user_id=portfolio_row["user_id"],
                name=portfolio_row["name"],
                target_value=portfolio_row["target_value"],
                risk_score=portfolio_row["risk_score"],
                timeframe=portfolio_row["timeframe"],
                created_at=portfolio_row["created_at"],
                simulation_data=portfolio_row.get("simulation_data")
            )
            db.add(portfolio)
        
        # Migrate stocks
        cursor.execute("SELECT * FROM stocks")
        stocks = cursor.fetchall()
        
        for stock_row in stocks:
            stock = Stock(
                id=stock_row["id"],
                portfolio_id=stock_row["portfolio_id"],
                symbol=stock_row["symbol"],
                name=stock_row["name"],
                allocation=stock_row["allocation"],
                price=stock_row.get("price")
            )
            db.add(stock)
        
        # Commit all changes
        db.commit()
        print("✅ Data migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
    finally:
        db.close()
        sqlite_conn.close()

if __name__ == "__main__":
    migrate_sqlite_to_postgresql()