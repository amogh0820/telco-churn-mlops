"""Guards the contract between the training config and the API schema.

These two definitions live in different files and are edited at different
times. If someone adds a feature to `src/config.py` and forgets the Pydantic
model, the API will happily accept requests missing that field and the imputer
will quietly fill it -- a silent accuracy regression with no error anywhere.
This test turns that into a red build.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from src.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures  # noqa: E402
from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
)


def test_schema_fields_match_the_training_features_exactly():
    schema_fields = set(CustomerFeatures.model_fields)
    assert schema_fields == set(RAW_FEATURES), (
        f"only in schema: {sorted(schema_fields - set(RAW_FEATURES))}; "
        f"only in config: {sorted(set(RAW_FEATURES) - schema_fields)}"
    )


def test_raw_features_has_no_duplicates():
    assert len(RAW_FEATURES) == len(set(RAW_FEATURES))


def test_numeric_and_categorical_do_not_overlap():
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_documented_example_validates():
    """The example shown in /docs must be a payload the API actually accepts."""
    customer = CustomerFeatures(**EXAMPLE_CUSTOMER)
    assert customer.tenure == EXAMPLE_CUSTOMER["tenure"]


def test_example_covers_every_field():
    assert set(EXAMPLE_CUSTOMER) == set(RAW_FEATURES)


def test_optional_total_charges_defaults_to_none():
    payload = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "TotalCharges"}
    assert CustomerFeatures(**payload).TotalCharges is None
