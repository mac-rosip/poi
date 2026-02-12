#!/usr/bin/env bash
# =============================================================================
# deploy-serverless.sh — Deploy Hyperfanity to RunPod Serverless
# =============================================================================
#
# Prerequisites:
#   1. Docker installed and logged into your registry
#   2. RunPod account with API key
#
# Usage:
#   ./scripts/deploy-serverless.sh <docker-username>
#
# Example:
#   ./scripts/deploy-serverless.sh myuser
#
# This will:
#   1. Build the serverless Docker image
#   2. Push to Docker Hub as <username>/hyperfanity-serverless:latest
#   3. Print instructions for creating the RunPod endpoint
#
# =============================================================================

set -euo pipefail

DOCKER_USER="${1:?Usage: $0 <docker-username>}"
IMAGE_NAME="${DOCKER_USER}/hyperfanity-serverless"
IMAGE_TAG="latest"

echo "========================================"
echo "  Hyperfanity Serverless Deploy"
echo "========================================"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Check for Docker
if ! command -v docker &>/dev/null; then
    echo "[!] Docker not found. Please install Docker."
    exit 1
fi

# Build the image
echo "[*] Building Docker image..."
docker build -f Dockerfile.serverless -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# Push to registry
echo ""
echo "[*] Pushing to Docker Hub..."
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "========================================"
echo "  Image pushed successfully!"
echo "========================================"
echo ""
echo "Next steps to create RunPod Serverless endpoint:"
echo ""
echo "1. Go to https://www.runpod.io/console/serverless"
echo ""
echo "2. Click 'New Endpoint'"
echo ""
echo "3. Configure:"
echo "   - Name: hyperfanity-worker"
echo "   - Container Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - GPU Type: RTX 4090, A100, or similar"
echo "   - Max Workers: 1-10 (as needed)"
echo "   - Idle Timeout: 60 seconds"
echo "   - Environment Variables:"
echo "       PANEL_ADDR=178.128.157.147:50051"
echo ""
echo "4. Click 'Deploy'"
echo ""
echo "5. Test with:"
echo "   curl -X POST https://api.runpod.ai/v2/<endpoint-id>/run \\"
echo "     -H 'Authorization: Bearer \$RUNPOD_API_KEY' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"input\": {\"max_runtime\": 3600}}'"
echo ""
