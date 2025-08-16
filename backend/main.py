# backend/main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Updated imports for new database structure
from core.config import settings
# 🔧 CORRECTED: Removed shap_visualization since it doesn't exist yet
from api.routers import auth, onboarding, simulations, instruments, ai_analysis, password_reset
from database.db import engine, Base  # Updated import path
from database.models import User, Simulation, PasswordResetToken  # ADD PASSWORD RESET TOKEN

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
        "https://wealthwise-6hl28l023-stephen-vincents-projects.vercel.app",
        "https://wealthwise-gnayglrqo-stephen-vincents-projects.vercel.app",  # Latest domain
        
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
            "https://wealthwise-1uf20iu4j-stephen-vincents-projects.vercel.app",
            "https://wealthwise-gnayglrqo-stephen-vincents-projects.vercel.app"  # Add latest
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
            "password_reset": "/auth",
            "onboarding": "/onboarding", 
            "simulations": "/simulations",
            "ai_analysis": "/api/ai",
            "instruments": "/api/instruments",
            # 🎯 NEW: Enhanced features endpoints
            "enhanced_features": "/onboarding/health/enhanced-features",
            "crash_analysis": "/onboarding/{simulation_id}/crash-analysis",
            "shap_visualization": "/onboarding/{simulation_id}/shap-visualization"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway and monitoring"""
    try:
        # Test database connection
        from database.db import SessionLocal
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
    
    # 🎯 NEW: Check enhanced features availability
    enhanced_features_status = _check_enhanced_features()
    
    return {
        "status": "healthy",
        "message": "WealthWise API is running",
        "timestamp": os.times(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": db_status,
        "ai_service": ai_status,
        "enhanced_features": enhanced_features_status,
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
        (password_reset.router, "/auth", ["password-reset"]),
        (onboarding.router, "/onboarding", ["onboarding"]),
        (simulations.router, "/simulations", ["simulations"]),
        (instruments.router, "/api/instruments", ["instruments"]),
        (ai_analysis.router, "/api/ai", ["ai-analysis"]),
        # 🔧 REMOVED: shap_visualization router (will be added when modular simulator is ready)
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
    enhanced_features = _check_enhanced_features()
    
    return {
        "project": "WealthWise - AI Investment Portfolio Simulator",
        "description": "University project demonstrating AI-powered financial technology",
        "features": [
            "AI portfolio analysis using free Groq API",
            "Risk assessment and recommendations", 
            "Portfolio simulation and tracking",
            "User authentication and data persistence",
            "Password reset functionality",
            "Cross-database compatibility (SQLite/PostgreSQL)",
            # 🎯 NEW: Enhanced features status
            "Enhanced portfolio simulation (ready for deployment)",
            "Market crash detection and analysis",
            "SHAP explainable AI integration",
            "Smart goal calculation (fixes 0% return issue)"
        ],
        "technology_stack": {
            "backend": "FastAPI + SQLAlchemy",
            "frontend": "React + Vite", 
            "database": "SQLite (dev) / PostgreSQL (prod)",
            "ai": "Groq API (free tier)",
            "email": "SMTP (configurable)",
            "hosting": "Railway (backend) + Vercel (frontend)",
            # 🎯 NEW: Enhanced tech stack
            "enhanced_ai": "WealthWise SHAP system (optional)",
            "news_analysis": "Finnhub API integration (optional)",
            "portfolio_optimization": "Multi-algorithm optimization"
        },
        "deployment": {
            "cost": "$0/month (free tier services)",
            "performance": "Production-ready",
            "scalability": "Handles concurrent users",
            "enhanced_features": enhanced_features
        }
    }

# Development-only endpoints
if os.getenv("ENVIRONMENT") == "development":
    @app.get("/api/dev/db-info")
    async def dev_database_info():
        """Development endpoint to check database status"""
        try:
            from database.db import SessionLocal
            from database.models import User, Simulation, PasswordResetToken
            
            db = SessionLocal()
            user_count = db.query(User).count()
            simulation_count = db.query(Simulation).count()
            token_count = db.query(PasswordResetToken).count()
            db.close()
            
            return {
                "database_type": "PostgreSQL" if os.getenv("DATABASE_URL") else "SQLite",
                "database_url": os.getenv("DATABASE_URL", "sqlite:///./wealthwise.db"),
                "tables": {
                    "users": user_count,
                    "simulations": simulation_count,
                    "password_reset_tokens": token_count
                },
                "status": "connected",
                "enhanced_features": _check_enhanced_features()
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

# 🛠️ HELPER FUNCTIONS

def _check_enhanced_features():
    """Check availability of enhanced features"""
    try:
        # Check modular simulator
        modular_simulator = False
        try:
            from portfolio_simulator.main import simulate_portfolio
            modular_simulator = True
        except ImportError:
            pass
        
        # Check WealthWise
        wealthwise_available = False
        try:
            from ai_models.stock_model.core.recommender import EnhancedStockRecommender
            wealthwise_available = True
        except ImportError:
            pass
        
        # Check news analysis
        news_analysis = False
        try:
            from services.news_analysis import NewsAnalysisService
            if os.getenv("FINNHUB_API_KEY"):
                news_analysis = True
        except ImportError:
            pass
        
        # Check AI analysis
        ai_analysis = False
        try:
            from services.ai_analysis import AIAnalysisService
            ai_analysis = True
        except Exception:
            pass
        
        return {
            "modular_portfolio_simulator": modular_simulator,
            "wealthwise_shap_system": wealthwise_available,
            "news_analysis_service": news_analysis,
            "ai_analysis_service": ai_analysis,
            "smart_goal_calculation": True,  # Always available in updated onboarding
            "status": "ready_for_deployment" if modular_simulator else "standard_mode"
        }
    except Exception as e:
        logger.error(f"Error checking enhanced features: {e}")
        return {"status": "error", "error": str(e)}

# 🎯 NEW: Enhanced features test endpoint
@app.get("/api/enhanced-features/test")
async def test_enhanced_features():
    """Test endpoint for enhanced features"""
    return {
        "message": "Enhanced features test endpoint",
        "features_status": _check_enhanced_features(),
        "endpoints_available": {
            "crash_analysis": "/onboarding/{simulation_id}/crash-analysis",
            "shap_visualization": "/onboarding/{simulation_id}/shap-visualization", 
            "enhanced_health": "/onboarding/health/enhanced-features"
        },
        "note": "Enhanced features will be fully activated when modular simulator is deployed"
    }

# Startup message
logger.info("🎓 WealthWise API configured for university project deployment")
logger.info("💰 Using free tier services: Groq AI + Railway + Vercel")
logger.info("🔗 Health check available at /health and /api/health")
logger.info("🔑 Password reset functionality enabled")
logger.info("🎯 Enhanced portfolio features ready for deployment")

# Add this at the very end of main.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)