"""
main.py — FastAPI application entry point.

Loads the classifier model once at startup via lifespan context manager,
then serves the API and static frontend files.

Start with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.classifier import Classifier

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load classifier ───────────────────────────────────────────────
    print("[Startup] Loading commit classifier...")
    classifier = Classifier()
    classifier.load()
    app.state.classifier = classifier
    print(f"[Startup] Classifier ready — mode: {classifier.mode}")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Smart Changelog Generator",
    description="NLP pipeline that generates structured changelogs from GitHub commits.",
    version="1.0.0",
    lifespan=lifespan,
)

# API routes
app.include_router(router, prefix="/api")

# Static files (frontend)
static_dir = Path("app/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = Path("app/static/index.html")
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Smart Changelog Generator API", "docs": "/docs"}
