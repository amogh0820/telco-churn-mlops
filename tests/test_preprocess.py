"""Tests for cleaning and for the fitted pipeline as a whole."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ID_COL, MODEL_CATEGORICAL, MODEL_NUMERIC, RAW_FEATURES, TARGET
from src.data.preprocess import clean
from src.features.pipeline import build_pipeline, feature_names

RAW_ROW = {
    "customerID": "0001-AAAAA",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 50.0,
    "TotalCharges": "600.0",
    "Churn": "No",
}


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_blank_total_charges_becomes_nan_not_a_carried_over_value():
    """The bug this guards against: pad-filling a blank with the row above.

    A blank TotalCharges must become NaN so the pipeline can reconstruct it as
    MonthlyCharges * tenure. If it silently inherited the previous customer's
    lifetime spend, a brand-new subscriber would look like a long-tenured one.
    """
    frame = clean(
        raw_frame(
            [
                {**RAW_ROW, "customerID": "A", "TotalCharges": "9999.0"},
                {**RAW_ROW, "customerID": "B", "tenure": 0, "TotalCharges": " "},
            ]
        )
    )
    assert frame.loc[0, "TotalCharges"] == 9999.0
    assert np.isnan(frame.loc[1, "TotalCharges"])


def test_senior_citizen_becomes_categorical():
    frame = clean(
        raw_frame(
            [
                {**RAW_ROW, "customerID": "A", "SeniorCitizen": 0},
                {**RAW_ROW, "customerID": "B", "SeniorCitizen": 1},
            ]
        )
    )
    assert frame["SeniorCitizen"].tolist() == ["No", "Yes"]


def test_target_is_encoded_as_int():
    frame = clean(
        raw_frame(
            [
                {**RAW_ROW, "customerID": "A", "Churn": "Yes"},
                {**RAW_ROW, "customerID": "B", "Churn": "No"},
            ]
        )
    )
    assert frame[TARGET].tolist() == [1, 0]
    assert frame[TARGET].dtype.kind == "i"


def test_unmappable_target_raises():
    with pytest.raises(ValueError):
        clean(raw_frame([{**RAW_ROW, "Churn": "Maybe"}]))


def test_clean_keeps_exactly_the_expected_columns():
    frame = clean(raw_frame([RAW_ROW]))
    assert list(frame.columns) == [ID_COL, *RAW_FEATURES, TARGET]


def test_clean_does_not_mutate_its_input():
    original = raw_frame([RAW_ROW])
    snapshot = original.copy()
    clean(original)
    pd.testing.assert_frame_equal(original, snapshot)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def synthetic_training_frame(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    frame = pd.DataFrame(
        {
            **{k: v for k, v in RAW_ROW.items() if k in RAW_FEATURES},
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": monthly * tenure,
            "Contract": contract,
        }
    )[RAW_FEATURES]
    y = pd.Series(((contract == "Month-to-month") & (tenure < 12)).astype(int))
    return frame, y


def test_pipeline_fits_and_predicts_from_raw_columns():
    X, y = synthetic_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X.head(5))[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_survives_an_unseen_category():
    """handle_unknown='ignore' must keep the API up when a new value appears."""
    X, y = synthetic_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    novel = X.head(1).copy()
    novel.loc[:, "PaymentMethod"] = "Crypto wallet"
    assert 0.0 <= float(pipeline.predict_proba(novel)[0, 1]) <= 1.0


def test_pipeline_handles_null_total_charges():
    X, y = synthetic_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.head(1).copy()
    row.loc[:, "tenure"] = 0
    row.loc[:, "TotalCharges"] = np.nan
    assert np.isfinite(pipeline.predict_proba(row)[0, 1])


def test_pipeline_ignores_column_order():
    """The API reindexes, but the pipeline should not depend on ordering."""
    X, y = synthetic_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    straight = pipeline.predict_proba(X.head(3))[:, 1]
    shuffled = pipeline.predict_proba(X.head(3)[list(reversed(RAW_FEATURES))])[:, 1]
    np.testing.assert_allclose(straight, shuffled)


def test_engineered_columns_reach_the_encoder():
    X, y = synthetic_training_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    names = feature_names(pipeline)
    for numeric in MODEL_NUMERIC:
        assert numeric in names, f"{numeric} missing from the encoded matrix"
    for categorical in MODEL_CATEGORICAL:
        assert any(n.startswith(categorical) for n in names), f"{categorical} missing"
