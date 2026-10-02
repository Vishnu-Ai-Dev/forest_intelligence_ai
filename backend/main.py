from fastapi import FastAPI
from backend.routers import router as api_v1_router

app = FastAPI(
    title="Forest Intelligence & Early-Warning AI",
    version="0.1.0"
)

app.include_router(api_v1_router)

@app.get("/")
def read_root():
    return {
        "message": "Forest Intelligence & Early-Warning AI backend is running",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "Forest Intelligence & Early-Warning AI",
        "version": "0.1.0"
    }