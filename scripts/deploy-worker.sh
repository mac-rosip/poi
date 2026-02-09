#!/usr/bin/env bash
# =============================================================================
# deploy-worker.sh — Build and deploy the Hyperfanity CUDA worker
# =============================================================================
#
# Usage:
#   ./scripts/deploy-worker.sh                    # Build only
#   ./scripts/deploy-worker.sh --test             # Build + run tests
#   ./scripts/deploy-worker.sh --run btc dead     # Build + mine btc prefix "dead"
#   ./scripts/deploy-worker.sh --push registry    # Build + push to registry
#
# =============================================================================

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-hyperfanity-worker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Hyperfanity Worker Deploy"
echo "========================================"
echo "  Image: ${FULL_IMAGE}"
echo ""

# ---- Build ----
echo "[*] Building worker image..."
docker build \
    -f Dockerfile.worker \
    -t "${FULL_IMAGE}" \
    .

echo "[+] Build complete: ${FULL_IMAGE}"

# ---- Handle flags ----
case "${1:-}" in
    --test)
        echo ""
        echo "[*] Running tests..."
        docker build \
            -f Dockerfile.worker \
            --target test \
            -t "${IMAGE_NAME}-test:${IMAGE_TAG}" \
            .
        docker run --gpus all --rm "${IMAGE_NAME}-test:${IMAGE_TAG}"
        echo "[+] All tests passed."
        ;;
    --run)
        CHAIN="${2:-eth}"
        PATTERN="${3:-dead}"
        echo ""
        echo "[*] Starting miner: chain=${CHAIN} prefix=${PATTERN}"
        docker run --gpus all --rm "${FULL_IMAGE}" \
            --chain "${CHAIN}" --prefix "${PATTERN}"
        ;;
    --push)
        REGISTRY="${2:?Usage: $0 --push <registry>}"
        REMOTE_TAG="${REGISTRY}/${FULL_IMAGE}"
        echo ""
        echo "[*] Pushing to ${REMOTE_TAG}..."
        docker tag "${FULL_IMAGE}" "${REMOTE_TAG}"
        docker push "${REMOTE_TAG}"
        echo "[+] Pushed: ${REMOTE_TAG}"
        ;;
    "")
        echo ""
        echo "Done. Run with:"
        echo "  docker run --gpus all ${FULL_IMAGE} --chain btc --prefix dead"
        echo ""
        echo "Or run tests:"
        echo "  $0 --test"
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: $0 [--test|--run <chain> <pattern>|--push <registry>]"
        exit 1
        ;;
esac
