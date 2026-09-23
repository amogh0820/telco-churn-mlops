# syntax=docker/dockerfile:1
#
# Dockerfile for Hugging Face Spaces' free CPU-basic tier (no credit card).
# Nearly identical to the project's main Dockerfile -- the only Spaces-specific
# requirements are: listen on the port declared as `app_port` in README.md
# (7860, HF's convention), and run as a non-root user with a writable home
# directory, which HF's build environment expects.
#
# This file is not built from the repo root. `deploy/huggingface/deploy_to_spaces.py`
# assembles a build context containing this Dockerfile, this README, src/,
# requirements.txt and artifacts/, then pushes that context as the Space.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    WORKERS=2 \
    HOME=/home/appuser

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY artifacts/model.joblib artifacts/model_metadata.json ./artifacts/

# HF's build environment sometimes runs an initial ownership pass as root
# before switching users; chown defensively so appuser can still read its
# own files regardless of build order.
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["sh", "-c", "exec gunicorn src.api.main:app \
     --worker-class uvicorn.workers.UvicornWorker \
     --workers ${WORKERS} \
     --bind 0.0.0.0:${PORT} \
     --timeout 60 \
     --graceful-timeout 30 \
     --access-logfile -"]
