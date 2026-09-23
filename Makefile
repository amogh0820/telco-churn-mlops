.PHONY: help setup data train train-quick report test lint fmt serve mlflow smoke docker-build docker-run compose-up compose-down clean all

IMAGE ?= telco-churn-api:latest
BASE_URL ?= http://localhost:8000

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a virtualenv and install dev dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	@echo "Now run: source .venv/bin/activate"

data:  ## Download and preprocess the real IBM Telco dataset
	python -m src.data.download
	python -m src.data.preprocess

train:  ## Full hyper-parameter sweep with MLflow tracking
	python -m src.models.train

train-quick:  ## Fast sweep, no MLflow -- for CI and smoke tests
	python -m src.models.train --no-mlflow --quick

report:  ## Write the real training results into the README
	python -m scripts.update_readme_results

test:  ## Run the Python test suite with coverage, then the frontend tests
	pytest tests/ -v --cov=src --cov-report=term-missing
	node demo/app.test.js

test-frontend:  ## Run only the frontend (demo/app.js) tests -- needs Node
	node demo/app.test.js

lint:  ## Lint and format-check
	ruff check src tests scripts
	ruff format --check src tests scripts

fmt:  ## Auto-format
	ruff format src tests scripts
	ruff check --fix src tests scripts

serve:  ## Run the API locally with hot reload
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:  ## Start a local MLflow UI on :5000
	mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

smoke:  ## Hit every endpoint of a running API (BASE_URL=... to target a deploy)
	./scripts/smoke_test.sh $(BASE_URL)

deploy-hf:  ## Push to a Hugging Face Space (free, no card) -- needs HF_TOKEN env var
	@if [ -z "$(SPACE_ID)" ]; then echo "Usage: make deploy-hf SPACE_ID=yourname/telco-churn-api"; exit 1; fi
	@if [ -z "$$HF_TOKEN" ]; then echo "Set HF_TOKEN first: export HF_TOKEN=hf_..."; exit 1; fi
	python -m deploy.huggingface.deploy_to_spaces $(SPACE_ID)

docker-build:  ## Build the API image (requires artifacts/model.joblib)
	docker build -t $(IMAGE) .

docker-run:  ## Run the API image on :8000
	docker run --rm -p 8000:8000 --name churn-api $(IMAGE)

compose-up:  ## Start MLflow + API together
	docker compose up --build -d

compose-down:  ## Stop everything
	docker compose down

clean:  ## Remove generated artifacts and caches
	rm -rf artifacts/*.joblib artifacts/*.json reports/*.json .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: data train report test  ## Dataset -> model -> README -> tests
