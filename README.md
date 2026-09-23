# Telco churn prediction — training to production

A churn model for a telecom subscriber base: a trained scikit-learn pipeline
served behind FastAPI, tracked with MLflow, containerised with Docker, tested
end to end, and deployed publicly on Render's free tier.

**Live demo:** <a href="https://telecom-churn-predictor-5ptx.onrender.com/" target="_blank">Telecom Customer Churn Predictor</a>  
**Repository:** <a href="https://github.com/amogh0820/telco-churn-mlops" target="_blank">GitHub Repository</a>

> The first request after a period of inactivity takes 30-60 seconds to
> respond -- Render's free tier sleeps an idle service and wakes it on the
> next request. That's expected, not a bug; see "Deploying to Render" below.

---

## What this does

A telecom operator loses about a quarter of its subscriber base. The
retention team wants to know who is about to leave, early enough to
intervene. The model scores each subscriber and the API returns a probability
plus a recommendation, and a small web client lets anyone try it without
writing a request by hand.

The recommendation is the interesting part. Most churn projects report
accuracy at a 0.5 cut-off, which is arbitrary and usually wrong: a retention
call costs money whether or not the customer was leaving, and a saved
customer is worth far more than the call. This project picks the decision
threshold that maximises expected campaign value under stated cost
assumptions, and reports the gap against the naive 0.5 baseline.

Those assumptions live in `src/config.py`, are recorded in the trained
model's own metadata, and are surfaced by the `/model-info` endpoint -- the
frontend reads them from there rather than hardcoding a copy:

| Assumption | Value |
|---|---|
| Cost of one retention contact | $60 |
| Margin lost when a customer churns | $500 |
| Share of targeted churners actually saved | 35% |

**Data:** the IBM Telco customer-churn sample -- 7,043 customers, 21 columns,
pulled straight from the [IBM repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
by `make data`. No Kaggle account needed, no manual download, and the
download step asserts the shape so a silent upstream change fails loudly.

---

## Results

<!-- RESULTS:START -->

_Results from the latest training run._

### Model comparison

Cross-validated PR-AUC on the training split, 5 folds. The winner is
chosen here, before the test set is touched.

| Model | CV PR-AUC | Std | |
|---|---|---|---|
| Random forest | 0.6641 | +/- 0.0219 | **selected** |
| Logistic regression | 0.6629 | +/- 0.0142 | |
| Hist gradient boosting | 0.6623 | +/- 0.0214 | |

### Held-out test performance

Single evaluation of **Random forest** on the 20% test split, at the tuned threshold of **0.58**.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8466 |
| PR-AUC (average precision) | 0.6652 |
| Brier score (lower is better) | 0.1587 |
| Precision | 0.5742 |
| Recall | 0.7139 |
| F1 | 0.6365 |
| Accuracy | 0.7835 |
| Decision threshold | 0.58 |

Confusion matrix at that threshold:

| | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 837 | 198 |
| **Actually churned** | 107 | 267 |

### Campaign value

Net value of a retention campaign on the test cohort under the cost assumptions in `src/config.py`.

| Threshold | Net campaign value |
|---|---|
| Naive 0.50 | $18,495 |
| Tuned 0.58 | $18,825 |
| **Difference** | **+$330** |

On a test cohort of 1,409 customers.

<!-- RESULTS:END -->

Report PR-AUC as the headline, not accuracy. The target is ~27% positive, so
a model that predicts "nobody churns" scores 73% accuracy and is not useful.

---

## Architecture

```
  IBM Telco CSV (downloaded, shape-asserted)
        |
        v
  src/data/preprocess.py --> stratified train/test split
        |
        v
  src/models/train.py
    - 3 model families x randomised hyper-parameter search (5-fold CV)
    - every candidate logged to MLflow as a nested run
    - threshold chosen on out-of-fold predictions, never on test
    - winner -> MLflow registry (@champion) + artifacts/model.joblib
        |
        v
  src/api/main.py (FastAPI)
    GET  /            -> the scoring UI (demo/index.html), same origin as the API
    POST /predict, /predict/batch
    GET  /health, /model-info, /metrics
        |
        v
  one Docker image -> one Render web service -> one public URL
```

One Docker image, one Render service, one URL: `demo/index.html` is a plain
static file, but `main.py` serves it directly at `GET /` when the image
bundles it. That means the frontend and the API are always same-origin -- no
CORS configuration to get right, no second Render service to create, and no
separate URL to keep track of for a resume link.

### Four decisions worth defending in an interview

**Feature engineering lives inside the scikit-learn pipeline.** The API
receives only the 19 raw fields a billing system knows. Derived features
(`avg_monthly_spend`, `charge_drift`, `num_addons`, `tenure_years`,
`tenure_bucket`) are computed by a `FunctionTransformer` that is part of
the fitted pipeline, so the exact code that ran during training runs at
serving time.

**The threshold is tuned on out-of-fold training predictions.** Tuning it on
the test set would leak. The test set is touched exactly once, at the end.

**Blank `TotalCharges` is reconstructed, not median-imputed.** All 11 blanks
belong to `tenure == 0` customers who have not been billed yet, so their true
lifetime spend is 0. Median imputation would invent history for those new
customers. There is a regression test for this behaviour.

**The trained model is committed to the repository.** Every other generated
file (raw data, MLflow runs, coverage reports) is gitignored, but
`artifacts/model.joblib` and `artifacts/model_metadata.json` are deliberate
exceptions.

---

## Project layout

```
src/
  config.py               paths, column groups, cost assumptions
  data/download.py        fetch the IBM CSV, assert its shape
  data/preprocess.py      type coercion, target encoding, stratified split
  features/engineering.py derived features (runs inside the pipeline)
  features/pipeline.py    ColumnTransformer + estimator assembly
  models/train.py         sweep, select, threshold, register
  models/evaluate.py      metrics + cost-based threshold search
  models/tracking.py      MLflow helpers (2.x/3.x compatible)
  api/schemas.py          Pydantic request/response contracts
  api/model_service.py    model loading and scoring
  api/main.py             FastAPI app; serves the UI at GET /
demo/
  index.html              the scoring UI's markup and styling
  app.js                  its logic -- request building, response rendering
  app.test.js             Node tests for app.js
tests/
  test_features.py        feature engineering + threshold search
  test_preprocess.py      cleaning + end-to-end pipeline behaviour
  test_contract.py        config <-> API schema drift guard
  test_api.py             endpoint contracts via TestClient
scripts/
  smoke_test.sh           verifies all endpoints against a running API
  update_readme_results.py writes real metrics into this README
  build_notebook.py       .py percent-cells -> .ipynb
deploy/                    alternative deployment paths -- see "Deploying"
  push_to_ecr.sh, ec2_user_data.sh, run_on_ec2.sh, iam/   AWS EC2 (optional)
  huggingface/              Hugging Face Spaces (optional)
notebooks/01_eda.ipynb     exploratory analysis
render.yaml                Render Blueprint
Dockerfile                 production container
.github/workflows/         CI/CD workflows
```

## The frontend

`demo/index.html` and `demo/app.js` are a small, dependency-free client:
no build step, no framework, one HTML file and one script. The form builds a
`POST /predict` body from `CustomerFeatures`; the panel shows the prediction
and a plain-language reading of it.

- **Risk and threshold values come from `/model-info`.** The frontend does
  not maintain a second hardcoded copy.
- **The endpoint defaults to the page's own origin** when served over http(s),
  so Render, Docker, and local serving work without configuration.
- **The application logic is testable without a browser.** `app.test.js`
  exercises the request and response logic with Node.
- **Frontend fixtures follow the API contract**, so schema changes can surface
  as test failures rather than silent UI bugs.

## MLflow

`src/models/train.py` opens one parent run per training sweep and nested runs
per model family, logging hyperparameters and cross-validated PR-AUC. The
winner is registered under `REGISTERED_MODEL_NAME` and pointed at by the
`PRODUCTION_ALIAS` alias (`@champion`).

Locally this runs against the file-backed `./mlruns` store. The deployed API
loads the joblib pipeline baked into its image and does not require a tracking
server.

## Docker

The `Dockerfile` builds the runtime image used by local Docker, Render, and
the optional deployment paths. It copies `src/`, `demo/`, and the committed
model artifacts, installs runtime requirements, and runs the application as a
non-root user.

```bash
make docker-build
make docker-run
make smoke
```

## Testing

```bash
make test
make test-frontend
make lint
```

The backend test suite covers preprocessing, feature engineering, API
contracts, and threshold behaviour. The frontend tests cover request building
and response handling.

---

## Deploying

### Render (actual deployment)

The live application is deployed as a Docker web service on Render.

The trained model is committed because the Render Docker build needs
`artifacts/model.joblib` and `artifacts/model_metadata.json` to exist in
the repository when the image is built. Other generated training and MLflow
files remain gitignored.

1. Connect the GitHub repository to a Render Web Service.
2. Select **Docker** as the runtime.
3. Set the health check path to `/health`.
4. Set `PORT=8000` and `WORKERS=1`.
5. Deploy and open the generated HTTPS URL.

### Alternative deployment paths

AWS EC2/ECR and Hugging Face Spaces configurations are retained as optional
demonstration paths from the original project. They are not used by the
current live demo.

---

## GitHub readiness

- No secrets or credentials are committed.
- `.env` and generated MLflow/training data are ignored.
- The two trained model artifacts are intentionally committed for deployment.
- CI/CD, tests, linting, and formatting are included.

## Known limitations

- No temporal validation: the dataset is a static snapshot with no timestamps.
- `/metrics` counters are per-worker.
- No input-drift monitoring.
- Cost assumptions are illustrative and would need to come from a real
  retention team in production.

## Licence

MIT.
