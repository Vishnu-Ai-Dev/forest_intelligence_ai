import logging
import os
from fastapi import FastAPI
from backend.routers import router as api_v1_router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("forest_intelligence")

app = FastAPI(
    title="Forest Intelligence & Early-Warning AI",
    version="0.1.0"
)

app.include_router(api_v1_router)

@app.get("/")
def read_root():
    logger.debug("Root endpoint called")
    return {
        "message": "Forest Intelligence & Early-Warning AI backend is running",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    logger.debug("Health endpoint called")
    return {
        "status": "ok",
        "project": "Forest Intelligence & Early-Warning AI",
        "version": "0.1.0"
    }