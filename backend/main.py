# backend/main.py

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Updated imports for new database structure
from core.config import settings
from api.routers import auth, onboarding, simulations, instruments, ai_analysis
from database.database import engine, Base  # Updated import path
from database.models import User, Simulation  # Import models to ensure they're registered

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting WealthWise API...")
    
    # Create all database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Log database type being used
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            logger.info("🗄️  Using SQLite for local development")
        else:
            logger.info("🐘 Using PostgreSQL for production")
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    # Log environment info
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(f"🌍 Environment: {environment}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down WealthWise API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered investment portfolio simulation and analysis for university projects",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced CORS middleware for production deployment
# Enhanced CORS middleware for production deployment
def setup_cors():
    """Setup CORS for both development and production"""
    
    # Start with default CORS origins from settings
    cors_origins = getattr(settings, 'BACKEND_CORS_ORIGINS', [])
    
    # Add essential development origins
    dev_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # Add ALL current Vercel deployment URLs
    prod_origins = [
        # Old deployment URLs
        "https://wealthwise-qfjdrpesk-stephen-vincents-projects.vercel.app",
        "https://wealthwise-c3jjtfc2i-stephen-vincents-projects.vercel.app",
        
        # Current deployment URLs (from Vercel dashboard)
        "https://wealthwise-six-gamma.vercel.app",
        "https://wealthwise-git-main-stephen-vincents-projects.vercel.app", 
        "https://wealthwise-1uf20iu4j-stephen-vincents-projects.vercel.app",
        "https://wealthwise-6hl28l023-stephen-vincents-projects.vercel.app",  # NEW DOMAIN
        
        # Wildcard pattern for all Vercel deployments
        "https://*.vercel.app"
    ]
    
    # Combine all origins
    all_origins = list(set(cors_origins + dev_origins + prod_origins))
    
    # Get environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Add environment-specific URLs
    if environment == "production":
        # Add common production patterns
        vercel_url = os.getenv("VERCEL_URL")
        if vercel_url and f"https://{vercel_url}" not in all_origins:
            all_origins.append(f"https://{vercel_url}")
        
        # Add Railway frontend URL pattern (if using Railway for both)
        railway_frontend = os.getenv("RAILWAY_STATIC_URL") 
        if railway_frontend and railway_frontend not in all_origins:
            all_origins.append(railway_frontend)
    
    # For development/testing, you might want to be more permissive
    if environment == "development":
        # Add localhost variants
        localhost_variants = [
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:8080"
        ]
        all_origins.extend(localhost_variants)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=all_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    logger.info(f"🌐 CORS configured for origins: {all_origins}")

# Setup CORS
setup_cors()

# Add this after setup_cors() call
@app.middleware("http")
async def add_cors_to_errors(request, call_next):
    """Ensure CORS headers are present on all responses, including errors"""
    response = await call_next(request)
    
    # Get origin from request
    origin = request.headers.get("origin")
    
    # Add CORS headers to all responses (including errors)
    if origin:
        # Updated allowed origins list to match current deployments
        allowed_origins = [
            # Development
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            
            # All current Vercel deployments
            "https://wealthwise-qfjdrpesk-stephen-vincents-projects.vercel.app",
            "https://wealthwise-c3jjtfc2i-stephen-vincents-projects.vercel.app",
            "https://wealthwise-six-gamma.vercel.app",
            "https://wealthwise-git-main-stephen-vincents-projects.vercel.app",
            "https://wealthwise-1uf20iu4j-stephen-vincents-projects.vercel.app"
        ]
        
        if origin in allowed_origins or origin.endswith(".vercel.app"):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# Health check endpoints (important for Railway deployment)
@app.get("/")
async def root():
    """Root endpoint with API information"""
    environment = os.getenv("ENVIRONMENT", "development")
    database_type = "PostgreSQL" if os.getenv("DATABASE_URL") else "SQLite"
    
    return {
        "message": "Welcome to WealthWise API",
        "version": settings.APP_VERSION,
        "environment": environment,
        "database": database_type,
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "auth": "/auth",
            "onboarding": "/onboarding", 
            "simulations": "/simulations",
            "ai_analysis": "/api/ai",
            "instruments": "/api/instruments"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway and monitoring"""
    try:
        # Test database connection
        from database.database import SessionLocal
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"
    
    # Check AI service availability
    groq_api_key = os.getenv("GROQ_API_KEY")
    ai_status = "configured" if groq_api_key else "missing_api_key"
    
    return {
        "status": "healthy",
        "message": "WealthWise API is running",
        "timestamp": os.times(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": db_status,
        "ai_service": ai_status,
        "version": settings.APP_VERSION
    }

@app.get("/api/health")
async def api_health_check():
    """Alternative health endpoint for API monitoring"""
    return await health_check()

# Include API routers with proper error handling
def include_routers():
    """Include all API routers with error handling"""
    routers_config = [
        (auth.router, "/auth", ["auth"]),
        (onboarding.router, "/onboarding", ["onboarding"]),
        (simulations.router, "/simulations", ["simulations"]),
        (instruments.router, "/api/instruments", ["instruments"]),
        (ai_analysis.router, "/api/ai", ["ai-analysis"])
    ]
    
    for router, prefix, tags in routers_config:
        try:
            app.include_router(router, prefix=prefix, tags=tags)
            logger.info(f"✅ Included router: {prefix}")
        except Exception as e:
            logger.error(f"❌ Failed to include router {prefix}: {e}")
            # Continue with other routers even if one fails

# Include all routers
include_routers()

# Error handlers for better debugging
@app.exception_handler(500)
async def internal_server_error(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "Something went wrong. Please try again later.",
        "status_code": 500
    }

# Middleware to log requests in development
if os.getenv("DEBUG", "true").lower() == "true":
    @app.middleware("http")
    async def log_requests(request, call_next):
        origin = request.headers.get("origin", "No origin")
        logger.info(f"📨 {request.method} {request.url} from {origin}")
        response = await call_next(request)
        logger.info(f"📤 Response: {response.status_code}")
        return response

# Railway deployment specific configurations
def configure_for_railway():
    """Configure app for Railway deployment"""
    port = os.getenv("PORT")
    if port:
        logger.info(f"🚂 Railway deployment detected. Port: {port}")
    
    # Railway provides DATABASE_URL automatically
    database_url = os.getenv("DATABASE_URL")
    if database_url and "railway" in database_url:
        logger.info("🚂 Using Railway PostgreSQL database")

# Configure for Railway if deployed there
configure_for_railway()

# Additional endpoints for university project demonstration
@app.get("/api/demo")
async def demo_info():
    """Demo endpoint for university project showcase"""
    return {
        "project": "WealthWise - AI Investment Portfolio Simulator",
        "description": "University project demonstrating AI-powered financial technology",
        "features": [
            "AI portfolio analysis using free Groq API",
            "Risk assessment and recommendations", 
            "Portfolio simulation and tracking",
            "User authentication and data persistence",
            "Cross-database compatibility (SQLite/PostgreSQL)"
        ],
        "technology_stack": {
            "backend": "FastAPI + SQLAlchemy",
            "frontend": "React + Vite", 
            "database": "SQLite (dev) / PostgreSQL (prod)",
            "ai": "Groq API (free tier)",
            "hosting": "Railway (backend) + Vercel (frontend)"
        },
        "deployment": {
            "cost": "$0/month (free tier services)",
            "performance": "Production-ready",
            "scalability": "Handles concurrent users"
        }
    }

# Development-only endpoints
if os.getenv("ENVIRONMENT") == "development":
    @app.get("/api/dev/db-info")
    async def dev_database_info():
        """Development endpoint to check database status"""
        try:
            from database.database import SessionLocal
            from database.models import User, Simulation
            
            db = SessionLocal()
            user_count = db.query(User).count()
            simulation_count = db.query(Simulation).count()
            db.close()
            
            return {
                "database_type": "PostgreSQL" if os.getenv("DATABASE_URL") else "SQLite",
                "database_url": os.getenv("DATABASE_URL", "sqlite:///./wealthwise.db"),
                "tables": {
                    "users": user_count,
                    "simulations": simulation_count
                },
                "status": "connected"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

# Startup message
logger.info("🎓 WealthWise API configured for university project deployment")
logger.info("💰 Using free tier services: Groq AI + Railway + Vercel")
logger.info("🔗 Health check available at /health and /api/health")