# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------- #
# Stage 1: build wheels so the runtime image carries no compilers.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt is runtime-only by design; mlflow, pytest, ruff and the
# notebook stack live in requirements-dev.txt and never reach this image.
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    WORKERS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY demo/ ./demo/

# The trained pipeline is baked in so the container boots without the tracking
# server being reachable. `make train` must run before `docker build`; if these
# two files are missing the build fails here, which is the intended behaviour.
COPY artifacts/model.joblib artifacts/model_metadata.json ./artifacts/

USER appuser

EXPOSE 8000

# Shell form (no brackets): reads $PORT at container-start time, the same
# way the CMD below does. A hardcoded port here would silently check the
# wrong address whenever $PORT is overridden -- e.g. by Render, which sets
# it via render.yaml -- leaving `docker ps` reporting a false "unhealthy"
# even though the service is fine. This never gates Render's own routing
# (Render's healthCheckPath does its own independent HTTP check), so the
# bug this guards against is local/registry tooling reading a stale status,
# not a production outage -- worth fixing regardless.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Each worker holds its own copy of the model, so scale workers with the box.
# Two fits comfortably in 1 GB alongside scikit-learn.
CMD ["sh", "-c", "exec gunicorn src.api.main:app \
     --worker-class uvicorn.workers.UvicornWorker \
     --workers ${WORKERS} \
     --bind 0.0.0.0:${PORT} \
     --timeout 60 \
     --graceful-timeout 30 \
     --access-logfile -"]
