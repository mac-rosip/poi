#!/usr/bin/env bash
# =============================================================================
# deploy-runpod.sh — Deploy Hyperfanity worker to RunPod GPU cloud
# =============================================================================
#
# Prerequisites:
#   1. RunPod CLI: pip install runpodctl
#   2. RunPod API key: export RUNPOD_API_KEY=...
#   3. Docker image pushed to a registry accessible by RunPod
#
# Usage:
#   ./scripts/deploy-runpod.sh <registry/image:tag> <chain> <pattern>
#
# Example:
#   ./scripts/deploy-runpod.sh ghcr.io/user/hyperfanity-worker:latest btc dead
#
# =============================================================================

set -euo pipefail

IMAGE="${1:?Usage: $0 <image> <chain> <pattern> [gpu_type]}"
CHAIN="${2:?Usage: $0 <image> <chain> <pattern> [gpu_type]}"
PATTERN="${3:?Usage: $0 <image> <chain> <pattern> [gpu_type]}"
GPU_TYPE="${4:-NVIDIA RTX A4000}"

echo "========================================"
echo "  Hyperfanity RunPod Deploy"
echo "========================================"
echo "  Image:    ${IMAGE}"
echo "  Chain:    ${CHAIN}"
echo "  Pattern:  ${PATTERN}"
echo "  GPU:      ${GPU_TYPE}"
echo ""

# Check for runpodctl
if ! command -v runpodctl &>/dev/null; then
    echo "[!] runpodctl not found. Install with: pip install runpodctl"
    echo "    Then set: export RUNPOD_API_KEY=<your-key>"
    exit 1
fi

# Check for API key
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "[!] RUNPOD_API_KEY not set."
    echo "    export RUNPOD_API_KEY=<your-key>"
    exit 1
fi

echo "[*] Creating RunPod pod..."

# Create a GPU pod with the worker image
runpodctl create pod \
    --name "hyperfanity-${CHAIN}-$(date +%s)" \
    --gpuType "${GPU_TYPE}" \
    --imageName "${IMAGE}" \
    --args "--chain ${CHAIN} --prefix ${PATTERN}" \
    --volumeSize 0 \
    --containerDiskSize 20

echo ""
echo "[+] Pod created. Monitor at: https://www.runpod.io/console/pods"
echo ""
echo "To list pods:   runpodctl get pod"
echo "To stop:        runpodctl stop pod <pod-id>"
echo "To remove:      runpodctl remove pod <pod-id>"
