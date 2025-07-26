# File: backend/migrate_data.py
import sqlite3
import json
from datetime import datetime
from database.db import SessionLocal, engine
from database.models import User, Simulation, PasswordResetToken, Base
from sqlalchemy.orm import Session
import os

def create_tables():
    """Create all tables in the database"""
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")

def get_sqlite_columns(cursor, table_name):
    """Get column names from SQLite table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    return [col[1] for col in columns_info]  # col[1] is the column name

def safe_get(row, key, default=None):
    """Safely get value from SQLite row object"""
    try:
        return row[key] if key in row.keys() else default
    except (KeyError, IndexError):
        return default

def migrate_sqlite_to_postgresql():
    """Migrate data from SQLite to PostgreSQL (optional)"""
    
    # Check if SQLite database exists
    sqlite_db_path = "wealthwise.db"
    if not os.path.exists(sqlite_db_path):
        print(f"❌ SQLite database '{sqlite_db_path}' not found. Skipping data migration.")
        return
    
    # Connect to your local SQLite database
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = sqlite_conn.cursor()
    
    # Create PostgreSQL session
    db: Session = SessionLocal()
    
    try:
        print("🔄 Starting data migration...")
        
        # Check what tables exist in SQLite
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        print(f"📋 Found tables in SQLite: {table_names}")
        
        # Migrate users
        if 'users' in table_names:
            print("👥 Migrating users...")
            
            # Get available columns in the users table
            user_columns = get_sqlite_columns(cursor, 'users')
            print(f"📋 Available user columns: {user_columns}")
            
            cursor.execute("SELECT * FROM users")
            users = cursor.fetchall()
            
            for user_row in users:
                # Check if user already exists (but query raw SQLite data to avoid SQLAlchemy column issues)
                cursor.execute("SELECT email FROM users WHERE email = ?", (user_row["email"],))
                if cursor.fetchone():
                    print(f"⏭️  User {user_row['email']} already exists in target, skipping...")
                    continue
                
                # Build user data based on available columns
                user_data = {
                    'name': user_row["name"],
                    'email': user_row["email"],
                    'hashed_password': user_row["hashed_password"],
                    'created_at': safe_get(user_row, "created_at", datetime.utcnow()),
                    'updated_at': safe_get(user_row, "updated_at", datetime.utcnow())
                }
                
                # Only set ID if we're working with a different database type
                if os.getenv("ENVIRONMENT") != "development":  # Not SQLite to SQLite
                    user_data['id'] = user_row["id"]
                
                user = User(**user_data)
                db.add(user)
                
            print(f"✅ Migrated {len(users)} users")
        
        # Migrate simulations
        simulation_tables = ['simulations', 'portfolios']  # Check both possible table names
        migrated_simulations = False
        
        for table_name in simulation_tables:
            if table_name in table_names and not migrated_simulations:
                print(f"📊 Migrating simulations from '{table_name}' table...")
                
                # Get available columns
                sim_columns = get_sqlite_columns(cursor, table_name)
                print(f"📋 Available {table_name} columns: {sim_columns}")
                
                cursor.execute(f"SELECT * FROM {table_name}")
                simulations = cursor.fetchall()
                
                for sim_row in simulations:
                    # Map old column names to new ones if necessary
                    simulation_data = {
                        'user_id': sim_row["user_id"],
                        'name': safe_get(sim_row, "name", "Migrated Simulation"),
                        'goal': safe_get(sim_row, "goal", "wealth building"),
                        'risk_score': safe_get(sim_row, "risk_score", 50),
                        'risk_label': safe_get(sim_row, "risk_label", "Moderate"),
                        'target_value': safe_get(sim_row, "target_value", 0),
                        'lump_sum': safe_get(sim_row, "lump_sum", 0),
                        'monthly': safe_get(sim_row, "monthly", 0),
                        'timeframe': str(safe_get(sim_row, "timeframe", "10")),
                        'target_achieved': bool(safe_get(sim_row, "target_achieved", False)),
                        'income_bracket': safe_get(sim_row, "income_bracket", "middle"),
                        'created_at': safe_get(sim_row, "created_at", datetime.utcnow())
                    }
                    
                    # Handle results/simulation_data field
                    if 'results' in sim_columns and safe_get(sim_row, 'results'):
                        simulation_data['results'] = sim_row['results']
                    elif 'simulation_data' in sim_columns and safe_get(sim_row, 'simulation_data'):
                        simulation_data['results'] = sim_row['simulation_data']
                    
                    # Handle AI summary
                    if 'ai_summary' in sim_columns and safe_get(sim_row, 'ai_summary'):
                        simulation_data['ai_summary'] = sim_row['ai_summary']
                    
                    # Only set ID if we're working with a different database type
                    if os.getenv("ENVIRONMENT") != "development":  # Not SQLite to SQLite
                        simulation_data['id'] = sim_row["id"]
                    
                    simulation = Simulation(**simulation_data)
                    db.add(simulation)
                
                print(f"✅ Migrated {len(simulations)} simulations")
                migrated_simulations = True
        
        # Handle existing password reset tokens if they exist
        if 'password_reset_tokens' in table_names:
            print("🔑 Found existing password reset tokens...")
            cursor.execute("SELECT * FROM password_reset_tokens")
            tokens = cursor.fetchall()
            
            for token_row in tokens:
                # Only migrate non-expired, unused tokens
                expires_at_str = safe_get(token_row, 'expires_at')
                if expires_at_str:
                    try:
                        if isinstance(expires_at_str, str):
                            expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                        else:
                            expires_at = expires_at_str
                    except:
                        expires_at = datetime.utcnow()  # Default to now if parsing fails
                else:
                    expires_at = datetime.utcnow()
                
                if not safe_get(token_row, 'used', False) and expires_at > datetime.utcnow():
                    token_data = {
                        'user_id': token_row['user_id'],
                        'email': token_row['email'],
                        'token': token_row['token'],
                        'expires_at': expires_at,
                        'used': safe_get(token_row, 'used', False),
                        'created_at': safe_get(token_row, 'created_at', datetime.utcnow())
                    }
                    
                    if os.getenv("ENVIRONMENT") != "development":
                        token_data['id'] = token_row["id"]
                    
                    reset_token = PasswordResetToken(**token_data)
                    db.add(reset_token)
            
            valid_tokens = [t for t in tokens if not safe_get(t, 'used', False)]
            print(f"✅ Migrated {len(valid_tokens)} valid password reset tokens")
        
        # Handle stocks table if it exists (might need to integrate into simulation results)
        if 'stocks' in table_names:
            print("📈 Found stocks table - you may need to manually integrate this data into simulation results")
            cursor.execute("SELECT * FROM stocks")
            stocks = cursor.fetchall()
            print(f"📊 Found {len(stocks)} stock records (manual integration may be required)")
        
        # Commit all changes
        db.commit()
        print("✅ Data migration completed successfully!")
        
        # Print summary
        user_count = db.query(User).count()
        simulation_count = db.query(Simulation).count()
        token_count = db.query(PasswordResetToken).count()
        print(f"📊 Final counts: {user_count} users, {simulation_count} simulations, {token_count} password reset tokens")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        sqlite_conn.close()

def add_sample_data():
    """Add sample data for testing (optional)"""
    db: Session = SessionLocal()
    
    try:
        # Check if we already have data
        if db.query(User).count() > 0:
            print("📊 Database already contains data, skipping sample data creation")
            return
        
        print("🎯 Adding sample data...")
        
        # Create sample user
        sample_user = User(
            name="Test User",
            email="test@example.com",
            hashed_password="$2b$12$sample_hashed_password",  # This should be properly hashed in real usage
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(sample_user)
        db.flush()  # Get the ID
        
        # Create sample simulation
        sample_simulation = Simulation(
            user_id=sample_user.id,
            name="Sample Portfolio",
            goal="retirement",
            risk_score=60,
            risk_label="Moderate Aggressive",
            target_value=100000,
            lump_sum=10000,
            monthly=500,
            timeframe="15",
            target_achieved=False,
            income_bracket="middle",
            results={"sample": "data"},
            ai_summary="This is a sample simulation for testing purposes.",
            created_at=datetime.utcnow()
        )
        db.add(sample_simulation)
        
        db.commit()
        print("✅ Sample data added successfully!")
        
    except Exception as e:
        print(f"❌ Failed to add sample data: {e}")
        db.rollback()
    finally:
        db.close()

def update_sqlite_schema():
    """Update existing SQLite database to match new schema"""
    sqlite_db_path = "wealthwise.db"
    if not os.path.exists(sqlite_db_path):
        print("❌ SQLite database not found, skipping schema update")
        return
    
    print("🔧 Updating SQLite schema...")
    
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    cursor = sqlite_conn.cursor()
    
    try:
        # Add updated_at column to users table if it doesn't exist
        user_columns = get_sqlite_columns(cursor, 'users')
        if 'updated_at' not in user_columns:
            # SQLite doesn't support CURRENT_TIMESTAMP as default in ALTER TABLE
            # So we'll add it with NULL and then update it
            cursor.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP")
            
            # Update all existing rows to have current timestamp
            current_time = datetime.utcnow().isoformat()
            cursor.execute("UPDATE users SET updated_at = ? WHERE updated_at IS NULL", (current_time,))
            
            print("✅ Added updated_at column to users table")
        
        # Create password_reset_tokens table if it doesn't exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='password_reset_tokens';")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    email VARCHAR(320) NOT NULL,
                    token VARCHAR(255) UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0 NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX idx_password_reset_tokens_user_id ON password_reset_tokens(user_id)")
            cursor.execute("CREATE INDEX idx_password_reset_tokens_email ON password_reset_tokens(email)")
            cursor.execute("CREATE INDEX idx_password_reset_tokens_token ON password_reset_tokens(token)")
            cursor.execute("CREATE INDEX idx_password_reset_tokens_expires_at ON password_reset_tokens(expires_at)")
            
            print("✅ Created password_reset_tokens table with indexes")
        
        sqlite_conn.commit()
        print("✅ SQLite schema updated successfully!")
        
    except Exception as e:
        print(f"❌ Schema update failed: {e}")
        sqlite_conn.rollback()
        raise
    finally:
        sqlite_conn.close()

def main():
    """Main migration function"""
    print("🚀 Starting database setup and migration...")
    
    try:
        # Step 0: Update SQLite schema if in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            update_sqlite_schema()
        
        # Step 1: Create tables
        create_tables()
        
        # Step 2: Migrate existing data (if SQLite database exists)
        migrate_sqlite_to_postgresql()
        
        # Step 3: Add sample data if no data exists
        add_sample_data()
        
        print("🎉 Database setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test the application with the updated database")
        print("2. Test password reset functionality")
        print("3. Verify all existing data migrated correctly")
        
    except Exception as e:
        print(f"💥 Setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your database connection settings in .env")
        print("2. Ensure your database server is running")
        print("3. Verify database user has proper permissions")
        print("4. Try running the script again after fixing schema issues")

if __name__ == "__main__":
    main()