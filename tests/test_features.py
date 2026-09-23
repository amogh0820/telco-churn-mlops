"""Tests for feature engineering and threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURES
from src.features.engineering import add_engineered_features
from src.models.evaluate import campaign_value, choose_threshold, metrics_at_threshold

BASE_ROW = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "MonthlyCharges": 80.0,
    "PaymentMethod": "Electronic check",
    "TotalCharges": 1920.0,
}


def frame(**overrides) -> pd.DataFrame:
    row = {**BASE_ROW, **overrides}
    return pd.DataFrame([row])[RAW_FEATURES]


def test_counts_addons():
    out = add_engineered_features(frame())
    assert out["num_addons"].iloc[0] == 3  # OnlineSecurity, OnlineBackup, StreamingTV


def test_zero_tenure_does_not_divide_by_zero():
    out = add_engineered_features(frame(tenure=0, TotalCharges=None, MonthlyCharges=70.0))
    assert np.isfinite(out["avg_monthly_spend"].iloc[0])
    assert out["TotalCharges"].iloc[0] == 0.0
    assert out["tenure_bucket"].iloc[0] == "0-6m"


def test_charge_drift_flags_a_price_rise():
    # Paid 50/mo historically, now billed 100 -> drift ~2.0
    out = add_engineered_features(frame(tenure=10, TotalCharges=500.0, MonthlyCharges=100.0))
    assert out["charge_drift"].iloc[0] == pytest.approx(2.0, rel=1e-6)


def test_engineering_is_pure():
    original = frame()
    snapshot = original.copy()
    add_engineered_features(original)
    pd.testing.assert_frame_equal(original, snapshot)


def test_tenure_buckets_cover_the_range():
    out = add_engineered_features(
        pd.concat([frame(tenure=t) for t in (0, 8, 18, 36, 54, 72)], ignore_index=True)
    )
    assert out["tenure_bucket"].tolist() == ["0-6m", "6-12m", "1-2y", "2-4y", "4-5y", "5y+"]


def test_threshold_search_beats_the_default():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.27, 2000)
    noise = rng.normal(0, 0.25, 2000)
    y_proba = np.clip(0.2 + 0.5 * y_true + noise, 0.001, 0.999)

    choice = choose_threshold(y_true, y_proba)
    assert 0.05 <= choice.threshold <= 0.95
    assert choice.expected_value >= campaign_value(y_true, y_proba, 0.5)


def test_confusion_counts_add_up():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = metrics_at_threshold(y_true, y_proba, 0.5)
    total = (
        m["true_negatives"] + m["false_positives"] + m["false_negatives"] + m["true_positives"]
    )
    assert total == 4
    assert m["recall"] == pytest.approx(0.5)
