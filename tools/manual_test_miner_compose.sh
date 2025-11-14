#!/usr/bin/env bash
# Build local miner image and run the full Docker Compose integration test.

set -euo pipefail

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")

cd "${REPO_ROOT}"

echo "Building infinitehash-miner:local image..."
docker build --platform linux/amd64 \
    -t infinitehash-miner:local \
    -f app/envs/deployed/Dockerfile .

echo "Starting manual Docker Compose integration test..."
MINER_IMAGE=infinitehash-miner:local uv run python app/src/manual_test_miner_compose.py
