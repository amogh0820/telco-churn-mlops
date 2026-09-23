"""Metrics, plus the thing most churn projects skip: choosing a threshold.

A churn model's default 0.5 cut-off is almost never the right one. The team
running the retention campaign has a budget per call and a value per saved
customer, so the question is not "which threshold maximises F1" but "which
threshold maximises money". This module answers that, and reports the
expected campaign value at the chosen cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CUSTOMER_LIFETIME_VALUE,
    OFFER_ACCEPTANCE_RATE,
    RETENTION_OFFER_COST,
)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    expected_value: float
    customers_targeted: int
    precision: float
    recall: float

    def as_dict(self) -> dict:
        return asdict(self)


def threshold_free_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Metrics that do not depend on where the cut-off sits.

    ROC-AUC is the headline number people expect. PR-AUC (average precision) is
    the more honest one on a ~27% positive class. Brier score checks whether the
    predicted probabilities are actually calibrated, which matters because the
    threshold search below treats them as real probabilities.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def metrics_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def campaign_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    offer_cost: float = RETENTION_OFFER_COST,
    clv: float = CUSTOMER_LIFETIME_VALUE,
    acceptance_rate: float = OFFER_ACCEPTANCE_RATE,
) -> float:
    """Net value of running a retention campaign at this threshold.

    Every customer we contact costs `offer_cost`, whether or not they were
    going to leave. Each true positive is saved with probability
    `acceptance_rate`, and a save is worth `clv`. False negatives cost nothing
    extra here because the lost margin is the baseline we are measuring against.
    """
    y_pred = (y_proba >= threshold).astype(int)
    targeted = int(y_pred.sum())
    true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
    return true_positives * acceptance_rate * clv - targeted * offer_cost


def choose_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    grid: np.ndarray | None = None,
    **cost_kwargs: float,
) -> ThresholdChoice:
    """Grid-search the threshold that maximises expected campaign value."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    best: ThresholdChoice | None = None
    for raw in grid:
        threshold = round(float(raw), 4)
        value = campaign_value(y_true, y_proba, threshold, **cost_kwargs)
        if best is None or value > best.expected_value:
            scored = metrics_at_threshold(y_true, y_proba, threshold)
            best = ThresholdChoice(
                threshold=threshold,
                expected_value=float(value),
                customers_targeted=int((y_proba >= threshold).sum()),
                precision=scored["precision"],
                recall=scored["recall"],
            )
    assert best is not None
    return best


def full_report(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    report = threshold_free_metrics(y_true, y_proba)
    report.update(metrics_at_threshold(y_true, y_proba, threshold))
    report["campaign_value"] = float(campaign_value(y_true, y_proba, threshold))
    report["campaign_value_at_0.5"] = float(campaign_value(y_true, y_proba, 0.5))
    return report
