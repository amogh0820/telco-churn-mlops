"""Single source of truth for paths, column groups and constants.

Everything else in the project imports from here so that a column name or a
path is only ever written down once.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = RAW_DIR / "telco_churn.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# The static demo page. Served at GET / by src/api/main.py when present in the
# image; deploy targets that don't bundle demo/ (e.g. the Hugging Face Spaces
# variant, which uses its own trimmed Dockerfile) fall back to a small JSON
# response instead of a 404, so main.py never has to guess which target it's
# running on.
DEMO_HTML = PROJECT_ROOT / "demo" / "index.html"

# The IBM "Telco customer churn" sample: 7,043 rows, 21 columns.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "Churn"
ID_COL = "customerID"

# Raw columns the caller must supply. Order matters for the training frame.
NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns produced by src/features/engineering.py, not supplied by the caller.
ENGINEERED_NUMERIC: list[str] = [
    "avg_monthly_spend",
    "charge_drift",
    "num_addons",
    "tenure_years",
]
ENGINEERED_CATEGORICAL: list[str] = ["tenure_bucket"]

MODEL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC
MODEL_CATEGORICAL = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

# The six optional add-on services, used to build `num_addons`.
ADDON_COLUMNS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Business assumptions used to pick the decision threshold.
# These are deliberately explicit: the "best" threshold depends on what a
# retention call costs and what a saved customer is worth.
# --------------------------------------------------------------------------- #
RETENTION_OFFER_COST = 60.0  # cost of contacting + discounting one customer
CUSTOMER_LIFETIME_VALUE = 500.0  # margin lost when a customer churns
OFFER_ACCEPTANCE_RATE = 0.35  # share of targeted churners we actually save

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
# A bare local path, not a file:// URI: MLflow accepts both, but an f-string
# file:// URI built from a Windows path (file://C:\...) is malformed.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "telco-churn")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "telco-churn-classifier")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "champion")


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for directory in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
