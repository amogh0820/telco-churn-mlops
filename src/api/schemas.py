"""Request and response schemas.

Every categorical field is a `Literal`, not a bare `str`. That means a typo in
a caller's payload comes back as a 422 naming the offending field and its legal
values, instead of silently one-hot encoding to all zeros and returning a
confident, wrong probability. Schema enforcement at the edge is the cheapest
data-quality control you get.

Two Pydantic v2 details worth knowing, both of which bite silently:

* A default belongs on the assignment (``x: T = Field(default=...)``), not
  inside ``Annotated[...]``. Some 2.x versions reject a default declared inside
  Annotated, so every optional field below uses the assignment form.
* Fields beginning with ``model_`` collide with Pydantic's protected namespace
  and emit a UserWarning on import. ``protected_namespaces=()`` silences that
  for the two response models that legitimately need ``model_version`` and
  ``model_family``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoInternet = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
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


class CustomerFeatures(BaseModel):
    """The 19 fields a billing system knows about a subscriber."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_CUSTOMER},
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0, le=120, description="Months on book")]
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(ge=0, le=1000)]
    # Null is legal: a customer in their first billing cycle has not been billed
    # yet. The pipeline reconstructs the value as MonthlyCharges * tenure.
    TotalCharges: float | None = Field(default=None, ge=0, le=100_000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: Annotated[float, Field(ge=0, le=1)]
    will_churn: bool = Field(description="probability >= the tuned threshold")
    threshold: Annotated[float, Field(ge=0, le=1)]
    risk_band: Literal["low", "medium", "high"]
    model_version: str


class BatchRequest(BaseModel):
    customers: Annotated[list[CustomerFeatures], Field(min_length=1, max_length=1000)]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    model_family: str | None = None
    threshold: float | None = None


class ErrorResponse(BaseModel):
    detail: str
