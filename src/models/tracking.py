"""Small MLflow helpers.

MLflow renamed `log_model(artifact_path=...)` to `log_model(name=...)` in 3.0
and kept the old keyword working with a deprecation warning. Rather than pin
you to one major version, these helpers inspect the signature and call
whichever one exists, so the project runs on 2.x and 3.x unchanged.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)


def init_tracking(tracking_uri: str, experiment: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    log.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, experiment)


def log_sklearn_model(model: Any, artifact_name: str = "model", **kwargs: Any):
    """Log a sklearn model, using whichever keyword this MLflow version wants."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    key = "name" if "name" in params else "artifact_path"
    return mlflow.sklearn.log_model(
    sk_model=model,
    serialization_format="cloudpickle",
    **{key: artifact_name},
    **kwargs,
)


def promote(model_uri: str, registered_name: str, alias: str) -> int:
    """Register `model_uri` and point `alias` at the new version.

    Aliases replaced the old Staging/Production stages in MLflow 2.9. Loading
    `models:/<name>@<alias>` always resolves to whatever version currently holds
    the alias, which is exactly what a serving container wants.
    """
    version = mlflow.register_model(model_uri=model_uri, name=registered_name)
    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_name, alias=alias, version=version.version
    )
    log.info("Registered %s v%s and set alias @%s", registered_name, version.version, alias)
    return int(version.version)
