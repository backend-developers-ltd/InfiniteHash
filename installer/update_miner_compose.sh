#!/usr/bin/env bash
# Update script for APS miner docker-compose deployment.

set -euo pipefail

ENV_NAME="${1:-prod}"
WORKING_DIRECTORY="${2:-$HOME/InfiniteHash-miner/}"

mkdir -p "${WORKING_DIRECTORY}"

cd "${WORKING_DIRECTORY}"

mkdir -p logs

GITHUB_URL="https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads"
PROXY_MODE_FILE="${WORKING_DIRECTORY}/proxy_mode"

PROXY_MODE=""
if [ -f "${PROXY_MODE_FILE}" ]; then
    PROXY_MODE=$(tr -d '[:space:]' < "${PROXY_MODE_FILE}" | tr '[:upper:]' '[:lower:]')
fi

if [ "${PROXY_MODE}" != "ihp" ] && [ "${PROXY_MODE}" != "braiins" ]; then
    if [ -f "${WORKING_DIRECTORY}/brainsproxy/config/active_profile.toml" ]; then
        PROXY_MODE="braiins"
    elif [ -f "${WORKING_DIRECTORY}/proxy/pools.toml" ]; then
        PROXY_MODE="ihp"
    else
        PROXY_MODE="ihp"
    fi
fi

if [ "${PROXY_MODE}" = "ihp" ]; then
    COMPOSE_REMOTE_PATH="envs/miner-ihp/docker-compose.yml"
    mkdir -p proxy
else
    COMPOSE_REMOTE_PATH="envs/miner/docker-compose.yml"
    mkdir -p brainsproxy/config
fi

TEMP_FILE="/tmp/InfiniteHash_miner_compose_update.yml"
curl -s "${GITHUB_URL}/deploy-config-${ENV_NAME}/${COMPOSE_REMOTE_PATH}" > "${TEMP_FILE}"

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
echo "Proxy mode: ${PROXY_MODE}"
