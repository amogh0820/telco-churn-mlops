"""Loads the model once at startup and scores requests.

Load order, first hit wins:
  1. `MLFLOW_MODEL_URI` (e.g. models:/telco-churn-classifier@champion) if set
  2. the local artifacts/model.joblib baked into the image

Pulling from the registry at boot is the "proper" MLOps answer, but it makes
the container depend on the tracking server being reachable at start-up. Baking
a joblib into the image keeps the deployed service self-contained. Supporting
both means you can demo either story.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, RAW_FEATURES

log = logging.getLogger(__name__)

HIGH_RISK = 0.70
MEDIUM_RISK = 0.40


def _risk_band(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    if probability >= MEDIUM_RISK:
        return "medium"
    return "low"


class ChurnModelService:
    """Holds the fitted pipeline and its metadata."""

    def __init__(self) -> None:
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.threshold: float = 0.5
        self.version: str = "unloaded"

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
        if uri:
            try:
                self._load_from_mlflow(uri)
                return
            except Exception:  # noqa: BLE001 - fall back rather than fail to boot
                log.exception("Could not load %s; falling back to local artifact", uri)
        self._load_from_disk()

    def _load_from_mlflow(self, uri: str) -> None:
        import mlflow.sklearn

        log.info("Loading model from MLflow: %s", uri)
        self.model = mlflow.sklearn.load_model(uri)
        self.version = uri
        self._load_metadata(METADATA_PATH)

    def _load_from_disk(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run `python -m src.models.train` first."
            )
        log.info("Loading model from %s", MODEL_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.version = os.getenv("MODEL_VERSION", "local")
        self._load_metadata(METADATA_PATH)

    def _load_metadata(self, path: Path) -> None:
        if path.exists():
            self.metadata = json.loads(path.read_text())
            self.threshold = float(self.metadata.get("threshold", 0.5))
            trained_at = self.metadata.get("trained_at", "")
            if self.version in {"local", ""} and trained_at:
                self.version = f"{self.metadata.get('model_family', 'model')}@{trained_at}"
        else:
            log.warning("No metadata at %s; defaulting threshold to 0.5", path)

    def predict(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        frame = pd.DataFrame(payloads)
        # Reindex rather than select: guarantees column order matches training
        # and fills any optional field the caller omitted with NaN, which the
        # pipeline's imputer handles.
        frame = frame.reindex(columns=RAW_FEATURES)

        probabilities = self.model.predict_proba(frame)[:, 1]
        return [
            {
                "churn_probability": round(float(p), 6),
                "will_churn": bool(p >= self.threshold),
                "threshold": self.threshold,
                "risk_band": _risk_band(float(p)),
                "model_version": self.version,
            }
            for p in probabilities
        ]


service = ChurnModelService()
