#!/usr/bin/env bash
# Run ON the EC2 instance (after SSH or Session Manager) to start the API.
#
#   AWS_REGION=ap-south-1 ECR_REPOSITORY=telco-churn-api ./run_on_ec2.sh
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-telco-churn-api}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker pull "${REGISTRY}/${ECR_REPOSITORY}:latest"
docker stop churn-api 2>/dev/null || true
docker rm churn-api 2>/dev/null || true

docker run -d \
  --name churn-api \
  --restart unless-stopped \
  -p 80:8000 \
  --memory 700m \
  "${REGISTRY}/${ECR_REPOSITORY}:latest"

sleep 8
curl -fsS http://localhost/health && echo " <- healthy"
