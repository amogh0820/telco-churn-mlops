"""Feature engineering.

This lives *inside* the scikit-learn pipeline rather than in a separate
preprocessing script. That matters: the API only ever receives the 19 raw
fields a billing system knows about, and the same code that ran at training
time derives everything else. There is no way for training and serving to
drift apart, because there is only one code path.

The function must stay module-level and stateless so that joblib can pickle
the surrounding `FunctionTransformer` by reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ADDON_COLUMNS

_TENURE_BINS = [-0.1, 6, 12, 24, 48, 60, np.inf]
_TENURE_LABELS = ["0-6m", "6-12m", "1-2y", "2-4y", "4-5y", "5y+"]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `frame` with derived columns appended.

    Derived columns
    ---------------
    avg_monthly_spend : lifetime spend divided by months on book. Differs from
        MonthlyCharges whenever the customer's plan has changed.
    charge_drift : current monthly charge relative to historical average. Values
        above 1 mean the customer is paying more now than they used to, which is
        a classic churn trigger.
    num_addons : how many of the six optional services the customer has. Sticky
        customers hold more add-ons.
    tenure_years : tenure on a scale that tree splits read more naturally.
    tenure_bucket : tenure binned into lifecycle stages, because churn hazard is
        not linear in tenure -- it spikes in the first year and flattens after.
    """
    out = frame.copy()

    tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0.0)
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # A blank TotalCharges only occurs for tenure == 0 customers, who have not
    # been billed yet. Their true lifetime spend is 0, not the column median.
    total = total.fillna(monthly * tenure)

    months = tenure.clip(lower=1)
    avg_monthly_spend = total / months

    out["tenure"] = tenure
    out["MonthlyCharges"] = monthly
    out["TotalCharges"] = total
    out["avg_monthly_spend"] = avg_monthly_spend
    out["charge_drift"] = monthly / avg_monthly_spend.replace(0.0, np.nan)
    out["charge_drift"] = out["charge_drift"].fillna(1.0)
    out["tenure_years"] = tenure / 12.0

    present = [c for c in ADDON_COLUMNS if c in out.columns]
    if present:
        out["num_addons"] = (out[present] == "Yes").sum(axis=1).astype(float)
    else:  # pragma: no cover - defensive
        out["num_addons"] = 0.0

    out["tenure_bucket"] = pd.cut(
        tenure, bins=_TENURE_BINS, labels=_TENURE_LABELS
    ).astype(str)

    return out
