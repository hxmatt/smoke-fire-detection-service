"""FastAPI entrypoint for the smoke & fire detection service."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

app = FastAPI(title="Smoke & Fire Detection Service")
app.include_router(router)

# The UI is a static page served at "/"; the API routes above take precedence.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True))
