# Setup & deployment guide (Windows, beginner-friendly)

This guide takes you from the project folder on Windows to a working local
application and the public Render deployment. Commands are written for
**PowerShell** unless stated otherwise.

---

## 0. What you'll end up with

- The project running locally with the trained model and web interface.
- The project on GitHub.
- The project deployed on Render at its public HTTPS URL.

The current repository already contains the trained model artifacts required
by the Docker deployment:

- `artifacts\model.joblib`
- `artifacts\model_metadata.json`

---

## 1. Install the software

| Software | Get it from | Notes |
|---|---|---|
| **Python 3.11** | <https://www.python.org/downloads/> | This project's dependencies are tested against Python 3.11. |
| **Git for Windows** | <https://git-scm.com/download/win> | Default options are fine. |
| **VS Code** | <https://code.visualstudio.com/> | Default options are fine. |
| **Node.js LTS** | <https://nodejs.org/> | Optional; needed only for frontend tests. |
| **Docker Desktop** | <https://www.docker.com/products/docker-desktop/> | Optional; Render builds the production container for you. |

You'll also need:

- **GitHub:** <https://github.com/join>
- **Render:** <https://dashboard.render.com/register>

Verify Python and Git:

```powershell
python --version
git --version
```

---

## 2. Open the project

Unzip the project and open the `churn-mlops` folder in VS Code.

Open **Terminal → New Terminal** and make sure the terminal is inside the
project directory.

---

## 3. Create and activate the Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

---

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

---

## 5. Get the dataset and train the model

The repository contains the trained model used by the current deployment, so
you do **not** need to retrain it just to run the application.

To reproduce the training pipeline from scratch, download the dataset and
train:

```powershell
python -m src.data.download
python -m src.data.preprocess
python -m src.models.train
```

Training evaluates Logistic Regression, Random Forest, and HistGradientBoosting
using cross-validation and produces:

```text
artifacts\model.joblib
artifacts\model_metadata.json
```

It also creates local MLflow tracking data under `mlruns\`. That generated
tracking data is intentionally not committed to GitHub.

---

## 6. Update the README results after retraining

If you retrain the model:

```powershell
python -m scripts.update_readme_results
```

This updates the Results section of `README.md` from the latest training
report.

---

## 7. Run the tests

Backend tests:

```powershell
pytest tests/ -v --cov=src --cov-report=term-missing
```

Frontend tests, if Node.js is installed:

```powershell
node demo\app.test.js
```

Lint and formatting:

```powershell
ruff check src tests scripts
ruff format --check src tests scripts
```

---

## 8. Run the application locally

```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- **http://localhost:8000/** — scoring interface
- **http://localhost:8000/docs** — FastAPI documentation
- **http://localhost:8000/health** — health check

Press `Ctrl+C` to stop the server.

---

## 9. View MLflow locally (optional)

MLflow tracking data is local and generated during training.

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open **http://localhost:5000**.

The local `mlflow.db` and `mlruns\` data are development artifacts and are
ignored by Git.

---

## 10. Docker (optional)

You do not need Docker Desktop to deploy to Render. Render builds the Docker
image remotely.

To test the production container locally:

```powershell
docker build -t telco-churn-api:latest .
docker run --rm -p 8000:8000 --name churn-api telco-churn-api:latest
```

Open http://localhost:8000/.

The Docker build requires the committed model artifact.

---

## 11. GitHub

The current repository is:

<https://github.com/amogh0820/telco-churn-mlops>

For a fresh copy of the project, the normal Git workflow is:

```powershell
git add .
git commit -m "Describe the change"
git push
```

The repository intentionally commits:

```text
artifacts/model.joblib
artifacts/model_metadata.json
```

Generated datasets, MLflow tracking data, caches, coverage files, and local
environment files are ignored by `.gitignore`.

---

## 12. Render deployment

The current live application is:

<https://telecom-churn-predictor-5ptx.onrender.com/>

For a new Render deployment:

1. Open <https://dashboard.render.com>.
2. Create a **Web Service**.
3. Connect the GitHub repository.
4. Select **Docker** as the runtime.
5. Set the health check path to `/health`.
6. Set:
   - `PORT=8000`
   - `WORKERS=1`
7. Create the service.

Render builds the Docker image from the repository and deploys it as the web
service.

The first request after a period of inactivity can take longer because the
free Render service may sleep while idle.

---

## 13. Verify the live deployment

Open the current Render URL and verify:

- [ ] The scoring page loads.
- [ ] A prediction can be submitted.
- [ ] `/health` returns successfully.
- [ ] `/docs` loads the FastAPI documentation.
- [ ] The GitHub repository contains the two trained model artifacts.
- [ ] The README links to the live demo and repository.

Current live demo:

<https://telecom-churn-predictor-5ptx.onrender.com/>

---

## 14. Updating the project

For normal code changes:

```powershell
git add .
git commit -m "Describe the change"
git push
```

Render automatically redeploys when the connected GitHub branch receives a
new commit.

If you retrain the model, make sure the updated files are committed:

```powershell
git add artifacts/model.joblib artifacts/model_metadata.json
git commit -m "Update trained model"
git push
```

---

## 15. Troubleshooting

### Python or pip is not recognized

Reinstall Python 3.11 and make sure Python is added to PATH.

### PowerShell blocks Activate.ps1

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate the environment again.

### Docker says the model file is missing

Make sure these files exist:

```text
artifacts\model.joblib
artifacts\model_metadata.json
```

If you are reproducing the project from scratch, run the training pipeline
before building Docker.

### Render reports that the model cannot be loaded

Check the Render logs. The committed model should be trained with the same
dependency versions specified by the repository's requirements files.

### The Render site is slow on the first request

The free service may sleep while idle. The first request after sleeping can
take longer while the service wakes up.

---

## Project documentation

For the project architecture, modeling decisions, results, testing, and
deployment details, see the main `README.md`.
