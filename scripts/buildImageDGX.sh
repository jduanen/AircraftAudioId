#!/usr/bin/env bash
# Build (or rebuild) the aircraft-audio-training Docker image on the DGX Spark.
# Run this ON the DGX Spark (via SSH), after syncing with syncToDGX.sh.
#
# Useful standalone when you need the image ready before precomputing PANNs
# embeddings or specs (those wrappers don't build it themselves) — e.g. right
# after adding a new dependency to Dockerfile.training.
#
# Usage:
#   bash scripts/buildImageDGX.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.training.yml"

echo "Building training image..."
docker compose \
    --file "${COMPOSE_FILE}" \
    --project-name aircraft-training \
    build
