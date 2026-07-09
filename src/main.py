"""FastAPI entrypoint for the smoke & fire detection service."""

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="Smoke & Fire Detection Service")
app.include_router(router)
