# Local MLflow tracking server for docker-compose.
#
# Built from python:3.11-slim rather than a pinned ghcr.io/mlflow/mlflow tag so
# the version comes from requirements-dev.txt and cannot drift away from the
# version that writes the runs.
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.16,<4.0"

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-db/mlflow.db", \
     "--default-artifact-root", "/mlruns"]
