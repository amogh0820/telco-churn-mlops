#!/usr/bin/env bash
# Verify every endpoint of a running API. Works against localhost or a
# deployed instance; used by CI after a deploy and by `make smoke`.
#
#   ./scripts/smoke_test.sh http://localhost:8000
#   ./scripts/smoke_test.sh http://<ec2-public-ip>
set -euo pipefail

BASE="${1:-http://localhost:8000}"
BASE="${BASE%/}"
PASS=0
FAIL=0

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }

check() {   # check <name> <expected-status> <curl args...>
  local name="$1" expected="$2"; shift 2
  local status
  status=$(curl -s -o /tmp/smoke_body -w '%{http_code}' "$@")
  if [ "$status" = "$expected" ]; then
    green "  PASS  $name (HTTP $status)"; PASS=$((PASS + 1))
  else
    red   "  FAIL  $name (got HTTP $status, expected $expected)"
    sed -n '1,5p' /tmp/smoke_body; FAIL=$((FAIL + 1))
  fi
}

contains() {  # contains <name> <needle>
  if grep -q "$2" /tmp/smoke_body; then
    green "  PASS  $1"; PASS=$((PASS + 1))
  else
    red   "  FAIL  $1 (missing '$2')"; FAIL=$((FAIL + 1))
  fi
}

CUSTOMER='{
  "gender":"Female","SeniorCitizen":"No","Partner":"Yes","Dependents":"No",
  "tenure":2,"PhoneService":"Yes","MultipleLines":"No",
  "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
  "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
  "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190.0
}'

echo "Smoke-testing $BASE"

echo "GET /health"
check "health responds" 200 "$BASE/health"
contains "model is loaded" '"model_loaded":true'
contains "status is ok" '"status":"ok"'

echo "GET /model-info"
check "model-info responds" 200 "$BASE/model-info"
contains "reports a threshold" '"threshold"'

echo "GET /metrics"
check "metrics responds" 200 "$BASE/metrics"
contains "prometheus format" 'churn_api_requests_total'

echo "POST /predict"
check "scores a customer" 200 -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d "$CUSTOMER"
contains "returns a probability" '"churn_probability"'
contains "returns a risk band" '"risk_band"'

echo "POST /predict with a bad category"
check "rejects an illegal Contract value" 422 -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' \
  -d "$(echo "$CUSTOMER" | sed 's/Month-to-month/Three year/')"

echo "POST /predict with an unexpected field"
check "rejects an unknown field" 422 -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' \
  -d "$(echo "$CUSTOMER" | sed 's/{/{"surprise":1,/')"

echo "POST /predict/batch"
check "scores a batch" 200 -X POST "$BASE/predict/batch" \
  -H 'Content-Type: application/json' \
  -d "{\"customers\":[$CUSTOMER,$CUSTOMER]}"
contains "batch count is 2" '"count":2'

echo "POST /predict/batch with an empty list"
check "rejects an empty batch" 422 -X POST "$BASE/predict/batch" \
  -H 'Content-Type: application/json' -d '{"customers":[]}'

echo "GET /docs"
check "openapi docs render" 200 "$BASE/docs"

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
