"""Train, compare, and register churn models.

Run:  python -m src.models.train
      python -m src.models.train --no-mlflow      # skip tracking entirely
      python -m src.models.train --quick          # tiny search, for CI

What happens:
  1. Load the split written by src/data/preprocess.py
  2. For each candidate estimator, randomised-search its hyper-parameters with
     stratified CV, scoring on PR-AUC (the right metric for an imbalanced
     target)
  3. Log every candidate to MLflow as a nested run
  4. Pick the winner on cross-validated PR-AUC -- not on test-set performance,
     which stays untouched until the very end
  5. Choose a decision threshold that maximises expected campaign value
  6. Score once on the held-out test set, log it, register the model, and write
     a self-contained artifacts/model.joblib for the API to load
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.config import (
    CUSTOMER_LIFETIME_VALUE,
    CV_FOLDS,
    ID_COL,
    METADATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    OFFER_ACCEPTANCE_RATE,
    PRODUCTION_ALIAS,
    RANDOM_STATE,
    RAW_FEATURES,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    RETENTION_OFFER_COST,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    ensure_dirs,
)
from src.features.pipeline import build_pipeline, feature_names
from src.models.evaluate import choose_threshold, full_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCORING = "average_precision"  # PR-AUC


def candidate_models(quick: bool = False) -> dict[str, dict[str, Any]]:
    """Estimators plus the distributions to sample hyper-parameters from.

    `class_weight="balanced"` on the linear and forest models, rather than
    resampling: it costs nothing, keeps the probability outputs interpretable,
    and avoids leaking synthetic rows into cross-validation folds the way a
    naively-applied SMOTE does.
    """
    n_iter = 3 if quick else 25
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "n_iter": min(n_iter, 10),
            "param_distributions": {
                "classifier__C": loguniform(1e-3, 1e2),
                "classifier__solver": ["liblinear", "lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "n_iter": n_iter,
            "param_distributions": {
                "classifier__n_estimators": randint(200, 600),
                "classifier__max_depth": randint(3, 16),
                "classifier__min_samples_leaf": randint(2, 40),
                "classifier__max_features": uniform(0.2, 0.7),
            },
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15
            ),
            "n_iter": n_iter,
            "param_distributions": {
                "classifier__learning_rate": loguniform(1e-2, 3e-1),
                "classifier__max_leaf_nodes": randint(8, 64),
                "classifier__min_samples_leaf": randint(10, 60),
                "classifier__l2_regularization": loguniform(1e-4, 1e1),
            },
        },
    }


def load_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    for path in (TRAIN_CSV, TEST_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.preprocess` first."
            )
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    drop = [c for c in (ID_COL, TARGET) if c in train.columns]
    return (
        train.drop(columns=drop)[RAW_FEATURES],
        train[TARGET],
        test.drop(columns=drop)[RAW_FEATURES],
        test[TARGET],
    )


def search_one(
    name: str, spec: dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=build_pipeline(spec["estimator"]),
        param_distributions=spec["param_distributions"],
        n_iter=spec["n_iter"],
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise",
    )
    log.info("Tuning %s (%d candidates x %d folds)", name, spec["n_iter"], CV_FOLDS)
    search.fit(X, y)
    log.info("  best CV %s = %.4f", SCORING, search.best_score_)
    return search


def top_drivers(pipeline, k: int = 15) -> dict[str, float] | None:
    """Global feature importances, for the report. None if unavailable."""
    clf = pipeline.named_steps["classifier"]
    names = feature_names(pipeline)
    if hasattr(clf, "feature_importances_"):
        weights = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        weights = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
    else:
        return None
    if len(weights) != len(names):
        return None
    order = np.argsort(weights)[::-1][:k]
    return {names[i]: float(weights[i]) for i in order}


def main(use_mlflow: bool = True, quick: bool = False) -> dict[str, Any]:
    ensure_dirs()
    X_train, y_train, X_test, y_test = load_split()
    log.info("Train %s | Test %s", X_train.shape, X_test.shape)

    if use_mlflow:
        from src.models.tracking import init_tracking, log_sklearn_model, promote

        init_tracking(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)
        import mlflow
        from mlflow.models import infer_signature

    results: dict[str, RandomizedSearchCV] = {}
    parent_ctx = (
        mlflow.start_run(run_name=f"sweep-{datetime.now(UTC):%Y%m%d-%H%M%S}")
        if use_mlflow
        else _NullRun()
    )

    with parent_ctx:
        for name, spec in candidate_models(quick=quick).items():
            child = mlflow.start_run(run_name=name, nested=True) if use_mlflow else _NullRun()
            with child:
                search = search_one(name, spec, X_train, y_train)
                results[name] = search
                if use_mlflow:
                    mlflow.log_param("model_family", name)
                    mlflow.log_params(
                        {k: str(v) for k, v in search.best_params_.items()}
                    )
                    mlflow.log_metric("cv_pr_auc", float(search.best_score_))
                    mlflow.log_metric(
                        "cv_pr_auc_std",
                        float(
                            search.cv_results_["std_test_score"][search.best_index_]
                        ),
                    )

        winner_name = max(results, key=lambda k: results[k].best_score_)
        winner = results[winner_name].best_estimator_
        log.info(
            "Winner: %s (CV PR-AUC %.4f)", winner_name, results[winner_name].best_score_
        )

        # Threshold is tuned on out-of-fold training predictions, never on test.
        oof_proba = cross_val_predict(
            winner,
            X_train,
            y_train,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        choice = choose_threshold(y_train.to_numpy(), oof_proba)
        log.info(
            "Chosen threshold %.2f -> expected value %.0f on the training fold",
            choice.threshold,
            choice.expected_value,
        )

        # Single, final look at the held-out set.
        test_proba = winner.predict_proba(X_test)[:, 1]
        report = full_report(y_test.to_numpy(), test_proba, choice.threshold)
        log.info(
            "TEST  roc_auc=%.4f pr_auc=%.4f precision=%.3f recall=%.3f",
            report["roc_auc"],
            report["pr_auc"],
            report["precision"],
            report["recall"],
        )

        drivers = top_drivers(winner)
        comparison = {
            name: {
                "cv_pr_auc": float(s.best_score_),
                "cv_pr_auc_std": float(s.cv_results_["std_test_score"][s.best_index_]),
                "best_params": {k: str(v) for k, v in s.best_params_.items()},
            }
            for name, s in sorted(
                results.items(), key=lambda kv: kv[1].best_score_, reverse=True
            )
        }
        metadata = {
            "model_family": winner_name,
            "model_comparison": comparison,
            "threshold": choice.threshold,
            # The threshold above isn't 0.5 -- it's whatever maximised expected
            # campaign value under these assumptions. Recording them alongside
            # the number they produced is what makes that number checkable
            # later, and lets anything downstream (including the demo page)
            # explain the threshold from the artifact instead of a copy of it.
            "cost_assumptions": {
                "retention_offer_cost": RETENTION_OFFER_COST,
                "customer_lifetime_value": CUSTOMER_LIFETIME_VALUE,
                "offer_acceptance_rate": OFFER_ACCEPTANCE_RATE,
            },
            "cv_pr_auc": float(results[winner_name].best_score_),
            "test_metrics": report,
            "best_params": {k: str(v) for k, v in results[winner_name].best_params_.items()},
            "raw_features": RAW_FEATURES,
            "trained_at": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "top_drivers": drivers,
        }

        joblib.dump(winner, MODEL_PATH)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        (REPORTS_DIR / "test_report.json").write_text(
            json.dumps(
                {
                    "model_family": winner_name,
                    "threshold": choice.threshold,
                    "cv_pr_auc": float(results[winner_name].best_score_),
                    "model_comparison": comparison,
                    "test_metrics": report,
                    "trained_at": metadata["trained_at"],
                },
                indent=2,
            )
        )
        log.info("Wrote %s and %s", MODEL_PATH, METADATA_PATH)

        if use_mlflow:
            mlflow.log_param("winner", winner_name)
            mlflow.log_param("decision_threshold", choice.threshold)
            mlflow.log_metrics(
                {k: v for k, v in report.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(METADATA_PATH))
            signature = infer_signature(X_train.head(50), test_proba[:50])
            info = log_sklearn_model(
                winner,
                artifact_name="model",
                signature=signature,
                input_example=X_train.head(3),
            )
            promote(info.model_uri, REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)

    return metadata


class _NullRun:
    """Stand-in for an MLflow run when tracking is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    parser.add_argument("--quick", action="store_true", help="tiny search, for CI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(use_mlflow=not args.no_mlflow, quick=args.quick)
