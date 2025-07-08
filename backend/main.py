# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.routers import auth, onboarding, simulations, instruments
from database.session import engine
from database.models import Base
from api.routers import ai_analysis

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Create all tables on startup
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
# Add this import with your other router imports
from api.routers import ai_analysis


app.include_router(ai_analysis.router, prefix="/api/ai", tags=["ai-analysis"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
app.include_router(simulations.router, prefix="/simulations", tags=["simulations"])
app.include_router(instruments.router, prefix="/api/instruments", tags=["instruments"])


@app.get("/")
async def root():
    return {"message": "Welcome to Wealthwise API"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Wealthwise API is running"}