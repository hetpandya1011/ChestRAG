"""ChestRAG API — Weekend 1 empty-stack skeleton.

Only /health exists right now. Retrieval, re-ranking, and generation arrive
once the ingest pipeline is in place; we'll split those into separate modules
(routes.py, retrieval.py, generate.py) as the logic grows.
"""
from __future__ import annotations

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

# Structured JSON logs from day one (a JD requirement: "production logging").
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()

app = FastAPI(title="ChestRAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. The web container pings this to prove connectivity."""
    log.info("health_check")
    return {"status": "ok"}
