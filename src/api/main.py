"""FastAPI service for churn scoring.

Run locally:  uvicorn src.api.main:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from src.api.model_service import service
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from src.config import DEMO_HTML

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

# Minimal in-process counters. Enough to expose a /metrics endpoint that
# Prometheus or a CloudWatch agent can scrape without pulling in a dependency.
_STATS = {"requests": 0, "predictions": 0, "errors": 0, "latency_ms_total": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at start-up -- never per request."""
    try:
        service.load()
        log.info("Model ready: %s", service.version)
    except Exception:  # noqa: BLE001
        # Boot anyway so /health can report the problem and the orchestrator
        # can restart us, instead of crash-looping before logs are readable.
        log.exception("Model failed to load at start-up")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Telco Churn API",
    description=(
        "Scores a telecom subscriber's probability of churning and flags whether "
        "they clear the retention-campaign threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's origin before going public
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_latency(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _STATS["requests"] += 1
    _STATS["latency_ms_total"] += elapsed_ms
    if response.status_code >= 500:
        _STATS["errors"] += 1
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/", include_in_schema=False)
async def root():
    """The project's one public URL: the scoring UI, if this image bundles it.

    Deploy targets built from the project's main Dockerfile (Render, the AWS
    EC2 path, local `docker-compose`) copy `demo/` into the image, so this
    serves the interactive client directly -- no separate static site or
    second URL to keep track of. The Hugging Face Spaces variant builds from
    its own smaller Dockerfile, which doesn't; on that target this falls back
    to a small JSON pointer instead of a 404, so the same code runs everywhere.
    """
    if DEMO_HTML.exists():
        return FileResponse(DEMO_HTML)
    return {"service": "telco-churn-api", "docs": "/docs", "health": "/health"}
@app.get("/app.js", include_in_schema=False)
async def app_js():
    return FileResponse(DEMO_HTML.parent / "app.js")

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness + readiness. Returns 200 either way; read `status`."""
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_version=service.version,
        model_family=service.metadata.get("model_family"),
        threshold=service.threshold if service.is_ready else None,
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics() -> str:
    requests = max(_STATS["requests"], 1)
    lines = [
        "# HELP churn_api_requests_total Total HTTP requests served.",
        "# TYPE churn_api_requests_total counter",
        f"churn_api_requests_total {_STATS['requests']}",
        "# HELP churn_api_predictions_total Total customer rows scored.",
        "# TYPE churn_api_predictions_total counter",
        f"churn_api_predictions_total {_STATS['predictions']}",
        "# HELP churn_api_errors_total Total 5xx responses.",
        "# TYPE churn_api_errors_total counter",
        f"churn_api_errors_total {_STATS['errors']}",
        "# HELP churn_api_latency_ms_avg Mean request latency in milliseconds.",
        "# TYPE churn_api_latency_ms_avg gauge",
        f"churn_api_latency_ms_avg {_STATS['latency_ms_total'] / requests:.3f}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/model-info", tags=["ops"])
async def model_info() -> dict:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    meta = dict(service.metadata)
    meta.pop("top_drivers", None)
    return {"version": service.version, **meta}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["scoring"],
)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    result = service.predict([customer.model_dump()])[0]
    _STATS["predictions"] += 1
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["scoring"],
)
async def predict_batch(payload: BatchRequest) -> BatchResponse:
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    rows = [c.model_dump() for c in payload.customers]
    results = service.predict(rows)
    _STATS["predictions"] += len(results)
    return BatchResponse(
        predictions=[PredictionResponse(**r) for r in results], count=len(results)
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s", request.url.path)
    _STATS["errors"] += 1
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
