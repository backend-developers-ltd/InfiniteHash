#!/usr/bin/env bash
# Script to update docker-compose.yml and restart services if needed

set -euo pipefail

# Default values for arguments
ENV_NAME="${1:-prod}"
WORKING_DIRECTORY="${2:-$HOME/luxor-validator/}"

mkdir -p "${WORKING_DIRECTORY}"

# Ensure we're in the working directory
cd "${WORKING_DIRECTORY}"

# Hardcoded GitHub raw URL
GITHUB_URL="https://raw.githubusercontent.com/backend-developers-ltd/luxor-subnet/refs/heads"

# Use a fixed temporary file for the remote docker-compose.yml
TEMP_FILE="/tmp/luxor_compose_update.yml"
curl -s "${GITHUB_URL}/deploy-config-${ENV_NAME}/envs/deployed/docker-compose.yml" > "${TEMP_FILE}"

# Path to the local docker-compose.yml file
LOCAL_FILE="${WORKING_DIRECTORY}/docker-compose.yml"

# Check if the local file exists
if [ ! -f "${LOCAL_FILE}" ]; then
    echo "Local docker-compose.yml does not exist. Creating it."
    cat "${TEMP_FILE}" > "${LOCAL_FILE}"
    UPDATED=true
else
    # Compare the files
    if diff -q "${TEMP_FILE}" "${LOCAL_FILE}" > /dev/null; then
        echo "No changes detected in docker-compose.yml"
        UPDATED=false
    else
        echo "Changes detected in docker-compose.yml. Updating..."
        cat "${TEMP_FILE}" > "${LOCAL_FILE}"
        UPDATED=true
    fi
fi

# We're using a fixed temporary file, so we don't need to clean it up

# If the file was updated, restart the services
if [ "${UPDATED}" = true ]; then
    echo "Updating services..."

    # Check if docker compose or docker-compose should be used
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        # Docker Compose V2 (docker compose)
        docker compose up -d --remove-orphans
    elif command -v docker-compose &> /dev/null; then
        # Docker Compose V1 (docker-compose)
        docker-compose up -d --remove-orphans
    else
        echo "Error: Neither docker compose nor docker-compose is available."
        exit 1
    fi

    echo "Services updated successfully."
fi

echo "Update process completed."
