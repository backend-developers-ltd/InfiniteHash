#!/usr/bin/env bash
# Update script for APS miner docker-compose deployment.

set -euo pipefail

ENV_NAME="${1:-prod}"
WORKING_DIRECTORY="${2:-$HOME/InfiniteHash-miner/}"

mkdir -p "${WORKING_DIRECTORY}"

cd "${WORKING_DIRECTORY}"

mkdir -p logs
mkdir -p brainsproxy/config

GITHUB_URL="https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads"

TEMP_FILE="/tmp/InfiniteHash_miner_compose_update.yml"
curl -s "${GITHUB_URL}/deploy-config-${ENV_NAME}/envs/miner/docker-compose.yml" > "${TEMP_FILE}"

LOCAL_FILE="${WORKING_DIRECTORY}/docker-compose.yml"

if [ ! -f "${LOCAL_FILE}" ]; then
    echo "Local docker-compose.yml does not exist. Creating it."
    cat "${TEMP_FILE}" > "${LOCAL_FILE}"
    UPDATED=true
else
    if diff -q "${TEMP_FILE}" "${LOCAL_FILE}" > /dev/null; then
        echo "No changes detected in docker-compose.yml"
        UPDATED=false
    else
        echo "Changes detected in docker-compose.yml. Updating..."
        cat "${TEMP_FILE}" > "${LOCAL_FILE}"
        UPDATED=true
    fi
fi

if [ "${UPDATED}" = true ]; then
    echo "Updating services..."

    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose up -d --remove-orphans
    elif command -v docker-compose &> /dev/null; then
        docker-compose up -d --remove-orphans
    else
        echo "Error: Neither docker compose nor docker-compose is available."
        exit 1
    fi

    echo "Services updated successfully."
fi

echo "Update process completed."
