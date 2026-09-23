"""Download the IBM Telco customer-churn dataset.

Run:  python -m src.data.download
"""

from __future__ import annotations

import logging
import urllib.request

import pandas as pd

from src.config import DATA_URL, RAW_CSV, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPECTED_ROWS = 7043
EXPECTED_COLS = 21


def download(force: bool = False) -> None:
    ensure_dirs()
    if RAW_CSV.exists() and not force:
        log.info("%s already exists, skipping download", RAW_CSV)
    else:
        log.info("Downloading %s", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
        log.info("Saved to %s", RAW_CSV)

    frame = pd.read_csv(RAW_CSV)
    rows, cols = frame.shape
    log.info("Loaded %d rows x %d columns", rows, cols)
    if (rows, cols) != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Unexpected shape {(rows, cols)}; expected "
            f"{(EXPECTED_ROWS, EXPECTED_COLS)}. The upstream file may have changed."
        )
    log.info("Churn rate: %.1f%%", (frame["Churn"] == "Yes").mean() * 100)


if __name__ == "__main__":
    download()
