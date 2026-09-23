"""Assembles the end-to-end scikit-learn pipeline.

Raw DataFrame -> engineered features -> encode/scale -> classifier.

Shipping one fitted `Pipeline` object (rather than a model plus a pile of
encoders) is what makes the serving layer trivial: `pipeline.predict_proba(df)`
where `df` holds the raw request fields.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import MODEL_CATEGORICAL, MODEL_NUMERIC
from src.features.engineering import add_engineered_features


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" means a category never seen in training
            # (a new payment method, say) encodes to all zeros instead of
            # crashing the API at 3am.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, MODEL_NUMERIC),
            ("cat", categorical, MODEL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap `estimator` in the full feature pipeline."""
    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Column names after encoding, for coefficient / importance reports."""
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())
