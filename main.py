from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from routes import router
from config import settings


# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("🚀 Face Recognition API starting up...")
    logger.info("Model: InsightFace (buffalo_sc)")
    logger.info(f"Threshold: {settings.MATCH_THRESHOLD}")

    yield

    logger.info("🛑 Face Recognition API shutting down...")


# ── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Face Recognition API",
    description="Dating app face verification using InsightFace ArcFace.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ── Health Routes ────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Face Recognition API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy"
    }


# ── Entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )