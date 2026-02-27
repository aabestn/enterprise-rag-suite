from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram

router = APIRouter(tags=["Telemetry"])

REQUEST_COUNT = Counter("rag_api_requests_total", "Total API Requests", ["method", "endpoint"])
LATENCY_HISTOGRAM = Histogram("rag_api_latency_seconds", "API Request Latency", ["endpoint"])

@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)