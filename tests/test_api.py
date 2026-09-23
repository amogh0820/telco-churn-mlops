"""API contract tests.

These run against a real trained model, so `make train` (or the CI step that
runs it) must have produced artifacts/model.joblib first. If it hasn't, the
module skips rather than failing -- a missing artifact is a pipeline problem,
not an API bug.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import MODEL_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="no trained model; run `make train` first"
)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 190.0,
}


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0 < body["threshold"] < 1


def test_predict_returns_a_calibrated_shape(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])


def test_month_to_month_scores_higher_than_two_year(client):
    """A sanity check on direction, not on a specific number."""
    risky = client.post("/predict", json=VALID_CUSTOMER).json()
    safe_payload = {**VALID_CUSTOMER, "Contract": "Two year", "tenure": 60,
                    "TotalCharges": 5730.0}
    safe = client.post("/predict", json=safe_payload).json()
    assert risky["churn_probability"] > safe["churn_probability"]


def test_unknown_category_is_rejected(client):
    bad = {**VALID_CUSTOMER, "Contract": "Three year"}
    assert client.post("/predict", json=bad).status_code == 422


def test_extra_field_is_rejected(client):
    bad = {**VALID_CUSTOMER, "favourite_colour": "blue"}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_tenure_is_rejected(client):
    bad = {**VALID_CUSTOMER, "tenure": -3}
    assert client.post("/predict", json=bad).status_code == 422


def test_total_charges_may_be_null(client):
    payload = {**VALID_CUSTOMER, "tenure": 0, "TotalCharges": None}
    assert client.post("/predict", json=payload).status_code == 200


def test_batch_matches_single_predictions(client):
    batch = client.post(
        "/predict/batch", json={"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
    ).json()
    assert batch["count"] == 2
    single = client.post("/predict", json=VALID_CUSTOMER).json()
    assert batch["predictions"][0]["churn_probability"] == pytest.approx(
        single["churn_probability"]
    )


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_metrics_endpoint_is_prometheus_shaped(client):
    body = client.get("/metrics").text
    assert "churn_api_requests_total" in body
    assert "# TYPE" in body


def test_root_serves_the_demo_page(client):
    """GET / is the project's one public URL: the scoring UI, not raw JSON.

    This only holds for images that bundle demo/ (the main Dockerfile does;
    the Hugging Face Spaces variant deliberately doesn't). Skip rather than
    fail on a checkout where demo/index.html isn't present, since that's a
    valid configuration too and main.py falls back to a JSON pointer for it.
    """
    from src.config import DEMO_HTML

    response = client.get("/")
    assert response.status_code == 200
    if DEMO_HTML.exists():
        assert "text/html" in response.headers["content-type"]
    else:
        assert response.json()["docs"] == "/docs"
