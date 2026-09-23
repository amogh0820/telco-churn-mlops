#!/usr/bin/env bash
# Build the image and push it to ECR. Run from the repo root.
#
#   AWS_REGION=ap-south-1 ECR_REPOSITORY=telco-churn-api ./deploy/push_to_ecr.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-telco-churn-api}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [[ ! -f artifacts/model.joblib ]]; then
  echo "artifacts/model.joblib is missing. Run 'make train' first." >&2
  exit 1
fi

aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "$ECR_REPOSITORY" \
       --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# Build for the instance's architecture, not your laptop's. An Apple Silicon
# Mac defaults to arm64 and the image will not start on an x86 EC2 instance.
docker build --platform linux/amd64 \
  -t "${REGISTRY}/${ECR_REPOSITORY}:${TAG}" \
  -t "${REGISTRY}/${ECR_REPOSITORY}:latest" .

docker push "${REGISTRY}/${ECR_REPOSITORY}:${TAG}"
docker push "${REGISTRY}/${ECR_REPOSITORY}:latest"

echo "Pushed ${REGISTRY}/${ECR_REPOSITORY}:${TAG}"
