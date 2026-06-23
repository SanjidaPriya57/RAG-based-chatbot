import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from db import init_db
from routes.upload import router as upload_router
from routes.ask import router as ask_router
from routes.voice import router as voice_router  # ← add with other imports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context for startup and shutdown events


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up AI Finance Assistant...")
    await init_db()
    yield
    logger.info("Shutting down AI Finance Assistant...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Finance Assistant",
    description="RAG-based finance chatbot with document ingestion",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(upload_router, prefix="/api", tags=["documents"])
app.include_router(ask_router, prefix="/api", tags=["chat"])
app.include_router(voice_router, prefix="/api",
                   tags=["voice"])  # ← add with other routers

# Health check endpoint


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {
        "message": "AI Finance Assistant API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
