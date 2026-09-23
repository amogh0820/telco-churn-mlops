"""Clean the raw CSV and write a stratified train/test split.

Deliberately minimal: the only work done here is what cannot live inside the
model pipeline -- type coercion, target encoding, and the split. Everything
that transforms a *feature* belongs in src/features/ so it also runs at
serving time.

Run:  python -m src.data.preprocess
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COL,
    RANDOM_STATE,
    RAW_CSV,
    RAW_FEATURES,
    TARGET,
    TEST_CSV,
    TEST_SIZE,
    TRAIN_CSV,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    # TotalCharges ships as text and holds 11 blank strings, all for customers
    # with tenure == 0. Coerce to float and leave the NaNs -- the engineering
    # step fills them with monthly * tenure (i.e. 0), which is the true value.
    # `errors="coerce"` already turns "" and " " into NaN, so no replace() is
    # needed; `Series.replace("", None)` would pad-fill on some pandas versions.
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )

    # SeniorCitizen is 0/1 but semantically categorical. Make that explicit so
    # the one-hot encoder picks it up instead of the scaler.
    senior = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    out["SeniorCitizen"] = senior.map({0: "No", 1: "Yes"})

    out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0})
    if out[TARGET].isna().any():
        raise ValueError("Unmapped values found in the Churn column")
    out[TARGET] = out[TARGET].astype(int)

    keep = [c for c in (ID_COL, *RAW_FEATURES, TARGET) if c in out.columns]
    return out[keep]


def main() -> None:
    ensure_dirs()
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found. Run `python -m src.data.download` first.")

    raw = pd.read_csv(RAW_CSV)
    frame = clean(raw)
    log.info("Cleaned frame: %d rows, %d columns", *frame.shape)
    log.info("Missing TotalCharges after coercion: %d", frame["TotalCharges"].isna().sum())

    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET],
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    log.info("Train: %d rows (%.1f%% churn)", len(train), train[TARGET].mean() * 100)
    log.info("Test:  %d rows (%.1f%% churn)", len(test), test[TARGET].mean() * 100)


if __name__ == "__main__":
    main()
