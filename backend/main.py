# backend/main.py

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from datetime import datetime
from pathlib import Path

# Updated imports for new database structure
from core.config import settings
from api.routers import auth, onboarding, simulations, instruments, ai_analysis, password_reset, shap_visualization
from database.db import engine, Base  
from database.models import User, Simulation, PasswordResetToken  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting WealthWise API...")
    
    # Create static directories if they don't exist
    static_dirs = ["static", "static/visualizations"]
    for dir_path in static_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    # Create all database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Log database type being used
        database_url = os.getenv("DATABASE_URL")
        if database_url is None:
            logger.info("Using SQLite for local development")
        else:
            logger.info("Using PostgreSQL for production")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Log environment info
    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Environment: {environment}")
    
    # Check SHAP visualization availability
    try:
        from services.portfolio_simulator import get_visualization_engine
        viz_engine = get_visualization_engine()
        if viz_engine:
            logger.info("SHAP VisualizationEngine available")
        else:
            logger.warning("SHAP VisualizationEngine not available - using fallbacks")
    except ImportError:
        logger.warning("Portfolio simulator service not available")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WealthWise API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered investment portfolio simulation and analysis with SHAP explainable AI",
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
        "https://wealthwise-gnayglrqo-stephen-vincents-projects.vercel.app",
        
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
    
    logger.info(f"CORS configured for origins: {all_origins}")

# Setup CORS
setup_cors()

# CRITICAL: Mount static files BEFORE other routes to serve visualization images
app.mount("/static", StaticFiles(directory="static"), name="static")
logger.info("Static file serving enabled for /static directory")

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
            "https://wealthwise-gnayglrqo-stephen-vincents-projects.vercel.app"
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
            "shap": "/api/shap",
            "static_files": "/static"
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
    
    # Check SHAP visualization availability
    try:
        from services.portfolio_simulator import get_visualization_engine
        viz_engine = get_visualization_engine()
        shap_status = "available" if viz_engine else "fallback_only"
    except:
        shap_status = "unavailable"
    
    return {
       "status": "healthy",
        "message": "WealthWise API is running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": db_status,
        "ai_service": ai_status,
        "shap_visualization": shap_status,
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
        # These routers already have /api prefix in their own files
        (instruments.router, "", ["instruments"]),
        (ai_analysis.router, "", ["ai-analysis"]),
        (shap_visualization.router, "/api/shap", ["shap-visualization"]),
    ]
    
    for router, prefix, tags in routers_config:
        try:
            app.include_router(router, prefix=prefix, tags=tags)
            logger.info(f"Included router: {prefix if prefix else 'root'}")
        except Exception as e:
            logger.error(f"Failed to include router {prefix}: {e}")
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
        logger.info(f"{request.method} {request.url} from {origin}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response

# Railway deployment specific configurations
def configure_for_railway():
    """Configure app for Railway deployment"""
    port = os.getenv("PORT")
    if port:
        logger.info(f"Railway deployment detected. Port: {port}")
    
    # Railway provides DATABASE_URL automatically
    database_url = os.getenv("DATABASE_URL")
    if database_url and "railway" in database_url:
        logger.info("Using Railway PostgreSQL database")

# Configure for Railway if deployed there
configure_for_railway()

# Additional endpoints for university project demonstration
@app.get("/api/demo")
async def demo_info():
    """Demo endpoint for university project showcase"""
    return {
        "project": "WealthWise - AI Investment Portfolio Simulator",
        "description": "University project demonstrating AI-powered financial technology with explainable AI",
        "features": [
            "AI portfolio analysis using free Groq API",
            "Risk assessment and recommendations", 
            "Portfolio simulation and tracking",
            "User authentication and data persistence",
            "Password reset functionality",
            "SHAP explainable AI visualizations with interactive charts",
            "Factor analysis and market regime detection",
            "Cross-database compatibility (SQLite/PostgreSQL)"
        ],
        "technology_stack": {
            "backend": "FastAPI + SQLAlchemy",
            "frontend": "React + Vite", 
            "database": "SQLite (dev) / PostgreSQL (prod)",
            "ai": "Groq API (free tier) + SHAP + Custom ML models",
            "visualization": "Custom VisualizationEngine with matplotlib fallbacks",
            "email": "SMTP (configurable)",
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
                "status": "connected"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    @app.get("/api/dev/shap-info")
    async def dev_shap_info():
        """Development endpoint to check SHAP functionality"""
        try:
            from database.db import SessionLocal
            from database.models import Simulation
            
            db = SessionLocal()
            
            # Count simulations with SHAP data
            total_simulations = db.query(Simulation).count()
            simulations_with_shap = db.query(Simulation).filter(
                Simulation.results.op('->>')('shap_explanation').isnot(None)
            ).count()
            
            # Get latest simulation with SHAP data
            latest_shap_sim = db.query(Simulation).filter(
                Simulation.results.op('->>')('shap_explanation').isnot(None)
            ).order_by(Simulation.created_at.desc()).first()
            
            # Check visualization files
            viz_dir = Path("static/visualizations")
            viz_files = list(viz_dir.glob("*.png")) if viz_dir.exists() else []
            
            db.close()
            
            return {
                "total_simulations": total_simulations,
                "simulations_with_shap": simulations_with_shap,
                "latest_shap_simulation": {
                    "id": latest_shap_sim.id if latest_shap_sim else None,
                    "name": latest_shap_sim.name if latest_shap_sim else None,
                    "created_at": latest_shap_sim.created_at.isoformat() if latest_shap_sim else None,
                    "has_visualizations": bool(latest_shap_sim.results.get("visualization_paths")) if latest_shap_sim else False
                },
                "visualization_files": len(viz_files),
                "shap_endpoints": [
                    "/api/shap/simulation/{id}/explanation",
                    "/api/shap/simulation/{id}/visualizations",
                    "/api/shap/simulation/{id}/chart/{chart_type}",
                    "/api/shap/simulation/{id}/chart-data",
                    "/api/shap/simulation/{id}/regenerate-shap"
                ],
                "static_endpoint": "/static/visualizations/{filename}"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    @app.get("/api/dev/test-visualization/{simulation_id}")
    async def test_visualization_endpoint(simulation_id: int):
        """Test endpoint to verify visualization functionality"""
        try:
            from database.db import SessionLocal
            from services.portfolio_simulator import get_simulation_visualizations
            
            db = SessionLocal()
            result = await get_simulation_visualizations(simulation_id, db)
            db.close()
            
            return {
                "simulation_id": simulation_id,
                "visualization_result": result,
                "test_endpoints": {
                    "explanation": f"/api/shap/simulation/{simulation_id}/explanation",
                    "chart_data": f"/api/shap/simulation/{simulation_id}/chart-data",
                    "visualizations": f"/api/shap/simulation/{simulation_id}/visualizations"
                }
            }
        except Exception as e:
            return {"error": str(e), "test_failed": True}

# Error handlers for SHAP-specific issues
@app.exception_handler(404)
async def not_found_handler(request, exc):
    # Check if it's a visualization file request
    if "/static/visualizations/" in str(request.url):
        logger.warning(f"Visualization file not found: {request.url}")
        return {
            "error": "Visualization not found",
            "message": "The requested visualization file does not exist or has not been generated yet.",
            "suggestion": "Try regenerating the visualization or check if the simulation has SHAP data.",
            "status_code": 404
        }
    
    return {
        "error": "Not found",
        "message": "The requested resource was not found.",
        "status_code": 404
    }

# Startup message
logger.info("WealthWise API configured for university project deployment")
logger.info("Using free tier services: Groq AI + Railway + Vercel")
logger.info("Health check available at /health and /api/health")
logger.info("Password reset functionality enabled")
logger.info("SHAP explainable AI endpoints enabled at /api/shap")
logger.info("Static file serving enabled at /static")

# Add this at the very end of main.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)