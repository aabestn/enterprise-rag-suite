import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import ingest, query, telemetry
from src.ingestion.db_tracker import AsyncDocTracker

db_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_tracker
    dsn = os.getenv("POSTGRES_DSN", "postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db")
    db_tracker = AsyncDocTracker(db_dsn=dsn)
    await db_tracker.connect()
    yield
    await db_tracker.close()

app = FastAPI(
    title="Enterprise RAG Pipeline & Evaluation Suite",
    version="1.0.0",
    lifespan=lifespan
)

# Register endpoints
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(telemetry.router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Enterprise RAG API"}