#!/usr/bin/env bash
# Installer for APS miner deployment.

set -euo pipefail

ENV_NAME="${1:-miner}"
WORKING_DIRECTORY=${2:-~/InfiniteHash-miner/}

mkdir -p "${WORKING_DIRECTORY}"
WORKING_DIRECTORY=$(realpath "${WORKING_DIRECTORY}")

CONFIG_FILE="${WORKING_DIRECTORY}/config.toml"
BRAIINSPROXY_DIR="${WORKING_DIRECTORY}/brainsproxy"
BRAIINSPROXY_CONFIG_DIR="${BRAIINSPROXY_DIR}/config"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Creating config.toml file..."

    read -r -p "Enter BITTENSOR_NETWORK [finney]: " BITTENSOR_NETWORK </dev/tty
    BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-finney}

    read -r -p "Enter BITTENSOR_NETUID [42]: " BITTENSOR_NETUID </dev/tty
    BITTENSOR_NETUID=${BITTENSOR_NETUID:-42}
    BITTENSOR_NETUID=$(echo "${BITTENSOR_NETUID}" | tr -d '[:space:]')

    read -r -p "Enter BITTENSOR_WALLET_DIRECTORY [~/.bittensor/wallets]: " BITTENSOR_WALLET_DIRECTORY </dev/tty
    BITTENSOR_WALLET_DIRECTORY=${BITTENSOR_WALLET_DIRECTORY:-~/.bittensor/wallets}

    read -r -p "Enter BITTENSOR_WALLET_NAME [default]: " BITTENSOR_WALLET_NAME </dev/tty
    BITTENSOR_WALLET_NAME=${BITTENSOR_WALLET_NAME:-default}

    read -r -p "Enter BITTENSOR_WALLET_HOTKEY_NAME [default]: " BITTENSOR_WALLET_HOTKEY_NAME </dev/tty
    BITTENSOR_WALLET_HOTKEY_NAME=${BITTENSOR_WALLET_HOTKEY_NAME:-default}

    read -r -p "Enter price multiplier for workers [1.0]: " PRICE_MULTIPLIER </dev/tty
    PRICE_MULTIPLIER=${PRICE_MULTIPLIER:-1.0}
    PRICE_MULTIPLIER=$(echo "${PRICE_MULTIPLIER}" | tr -d '[:space:]')

    read -r -p "Enter worker hashrates (comma separated) [100,200,150]: " HASHRATES_INPUT </dev/tty
    HASHRATES_INPUT=${HASHRATES_INPUT:-100,200,150}

    IFS=',' read -ra HASHRATE_ITEMS <<< "${HASHRATES_INPUT}"
    HASHRATE_VALUES=()
    for item in "${HASHRATE_ITEMS[@]}"; do
        trimmed=$(echo "${item}" | tr -d '[:space:]')
        if [ -n "${trimmed}" ]; then
            HASHRATE_VALUES+=("${trimmed}")
        fi
    done
    if [ "${#HASHRATE_VALUES[@]}" -eq 0 ]; then
        HASHRATE_VALUES=("100" "200" "150")
    fi

    HASHRATE_LINES=""
    for idx in "${!HASHRATE_VALUES[@]}"; do
        value=${HASHRATE_VALUES[$idx]}
        if [ "${idx}" -eq 0 ]; then
            HASHRATE_LINES="    \"${value}\""
        else
            HASHRATE_LINES="${HASHRATE_LINES},
    \"${value}\""
        fi
    done

    cat > "${CONFIG_FILE}" <<EOL
[bittensor]
network = "${BITTENSOR_NETWORK}"
netuid = ${BITTENSOR_NETUID}

[wallet]
name = "${BITTENSOR_WALLET_NAME}"
hotkey_name = "${BITTENSOR_WALLET_HOTKEY_NAME}"
directory = "${BITTENSOR_WALLET_DIRECTORY}"

[workers]
price_multiplier = "${PRICE_MULTIPLIER}"
hashrates = [
${HASHRATE_LINES}
]
EOL

    echo "config.toml file created successfully at ${CONFIG_FILE}."
fi

mkdir -p "${WORKING_DIRECTORY}/logs"
mkdir -p "${BRAIINSPROXY_CONFIG_DIR}"

BRAIINSPROXY_ACTIVE_PROFILE="${BRAIINSPROXY_CONFIG_DIR}/active_profile.toml"

if [ ! -f "${BRAIINSPROXY_ACTIVE_PROFILE}" ]; then
    echo "Creating Braiins Farm Proxy profile at ${BRAIINSPROXY_ACTIVE_PROFILE}..."
    cat > "${BRAIINSPROXY_ACTIVE_PROFILE}" <<'EOL'
[[server]]
name = "InfiniteHash"
port = 3333

[[target]]
name = "InfiniteHashLuxorTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "InfiniteHashLuxor"
identity_pass_through = true

[[target]]
name = "MinerDefaultTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "MinerDefault"
identity_pass_through = true

[[target]]
name = "MinerBackupTarget"
url = "stratum+tcp://btc.viabtc.io:3333"
user_identity = "MinerBackup"
identity_pass_through = true

[[routing]]
name = "RD"
from = ["InfiniteHash"]

[[routing.goal]]
name = "InfiniteHashLuxorGoal"
hr_weight = 9

[[routing.goal.level]]
targets = ["InfiniteHashLuxorTarget"]

[[routing.goal]]
name = "MinerDefaultGoal"
hr_weight = 10

[[routing.goal.level]]
targets = ["MinerDefaultTarget"]

[[routing.goal.level]]
targets = ["MinerBackupTarget"]
EOL
fi

GITHUB_URL="https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads"

echo "Running update_miner_compose.sh once to ensure it works..."
curl -s "${GITHUB_URL}/deploy-config-${ENV_NAME}/installer/update_miner_compose.sh" > /tmp/update_miner_compose.sh
chmod +x /tmp/update_miner_compose.sh
if ! /tmp/update_miner_compose.sh "${ENV_NAME}" "${WORKING_DIRECTORY}"; then
    echo "Error: update_miner_compose.sh failed. Not adding cron line."
    exit 1
fi
echo "update_miner_compose.sh ran successfully."

CRON_CMD="*/15 * * * * cd ${WORKING_DIRECTORY} && curl -s ${GITHUB_URL}/deploy-config-${ENV_NAME}/installer/update_miner_compose.sh > /tmp/update_miner_compose.sh && chmod +x /tmp/update_miner_compose.sh && /tmp/update_miner_compose.sh ${ENV_NAME} ${WORKING_DIRECTORY} # INFINITE_HASH_APS_MINER_UPDATE"

(crontab -l 2>/dev/null || echo "") | grep -v "INFINITE_HASH_APS_MINER_UPDATE" | { cat; echo "${CRON_CMD}"; } | crontab -

echo "Cron job installed successfully. It will run every 15 minutes."
echo "Environment: ${ENV_NAME}"
echo "Working directory: ${WORKING_DIRECTORY}"
