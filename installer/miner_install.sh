#!/usr/bin/env bash
# Installer for APS miner deployment.

set -euo pipefail

ENV_NAME="${1:-prod}"
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

    read -r -p "Enter BITTENSOR_NETUID [89]: " BITTENSOR_NETUID </dev/tty
    BITTENSOR_NETUID=${BITTENSOR_NETUID:-89}
    BITTENSOR_NETUID=$(echo "${BITTENSOR_NETUID}" | tr -d '[:space:]')

    read -r -p "Enter BITTENSOR_WALLET_DIRECTORY [~/.bittensor/wallets]: " BITTENSOR_WALLET_DIRECTORY </dev/tty
    BITTENSOR_WALLET_DIRECTORY=${BITTENSOR_WALLET_DIRECTORY:-~/.bittensor/wallets}

    read -r -p "Enter BITTENSOR_WALLET_NAME [default]: " BITTENSOR_WALLET_NAME </dev/tty
    BITTENSOR_WALLET_NAME=${BITTENSOR_WALLET_NAME:-default}

    read -r -p "Enter BITTENSOR_WALLET_HOTKEY_NAME [default]: " BITTENSOR_WALLET_HOTKEY_NAME </dev/tty
    BITTENSOR_WALLET_HOTKEY_NAME=${BITTENSOR_WALLET_HOTKEY_NAME:-default}

    read -r -p "Enter price multiplier for workers [1.05]: " PRICE_MULTIPLIER </dev/tty
    PRICE_MULTIPLIER=${PRICE_MULTIPLIER:-1.05}
    PRICE_MULTIPLIER=$(echo "${PRICE_MULTIPLIER}" | tr -d '[:space:]')

    read -r -p "Enter worker hashrates in PH (comma separated, e.g. 5.5,8.2) []: " HASHRATES_INPUT </dev/tty
    HASHRATES_INPUT=${HASHRATES_INPUT:-}

    IFS=',' read -ra HASHRATE_ITEMS <<< "${HASHRATES_INPUT}"
    HASHRATE_VALUES=()
    for item in "${HASHRATE_ITEMS[@]}"; do
        trimmed=$(echo "${item}" | tr -d '[:space:]')
        if [ -n "${trimmed}" ]; then
            HASHRATE_VALUES+=("${trimmed}")
        fi
    done

    HASHRATE_LINES=""
    if [ "${#HASHRATE_VALUES[@]}" -gt 0 ]; then
        for idx in "${!HASHRATE_VALUES[@]}"; do
            value=${HASHRATE_VALUES[$idx]}
            if [ "${idx}" -eq 0 ]; then
                HASHRATE_LINES="    \"${value}\""
            else
                HASHRATE_LINES="${HASHRATE_LINES},
    \"${value}\""
            fi
        done
    fi

    DEFAULT_LUXOR_IDENTITY="sn${BITTENSOR_NETUID}auction.${BITTENSOR_WALLET_HOTKEY_NAME}.worker1"
    read -r -p "Enter Luxor user identity for InfiniteHash target [${DEFAULT_LUXOR_IDENTITY}]: " LUXOR_USER_ID </dev/tty
    LUXOR_USER_ID=${LUXOR_USER_ID:-${DEFAULT_LUXOR_IDENTITY}}

    read -r -p "Enable identity pass-through for Luxor target? [Y/n]: " LUXOR_PASSTHROUGH </dev/tty
    if [ -z "${LUXOR_PASSTHROUGH}" ] || [[ "${LUXOR_PASSTHROUGH}" =~ ^[Yy] ]]; then
        LUXOR_PASSTHROUGH_BOOL=true
    else
        LUXOR_PASSTHROUGH_BOOL=false
    fi

    DEFAULT_MINER_IDENTITY="infinite.${BITTENSOR_WALLET_HOTKEY_NAME}.worker1"
    read -r -p "Enter user identity for MinerDefault target [${DEFAULT_MINER_IDENTITY}]: " MINER_DEFAULT_ID </dev/tty
    MINER_DEFAULT_ID=${MINER_DEFAULT_ID:-${DEFAULT_MINER_IDENTITY}}

    DEFAULT_BACKUP_IDENTITY="backup.${BITTENSOR_WALLET_HOTKEY_NAME}.worker1"
    read -r -p "Enter user identity for MinerBackup target [${DEFAULT_BACKUP_IDENTITY}]: " MINER_BACKUP_ID </dev/tty
    MINER_BACKUP_ID=${MINER_BACKUP_ID:-${DEFAULT_BACKUP_IDENTITY}}

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
else
    echo "Using existing config at ${CONFIG_FILE}"
    parse_toml_value() {
        local section="$1"
        local key="$2"
        awk -v section="$section" -v key="$key" '
            /^\[/ {
                current = substr($0, 2, length($0) - 2)
                next
            }
            current == section {
                if ($0 ~ ("^" key " ")) {
                    sub(/^[^=]*= */, "")
                    gsub(/"/, "")
                    print
                    exit
                }
            }
        ' "${CONFIG_FILE}"
    }
    BITTENSOR_NETWORK=${BITTENSOR_NETWORK:-$(parse_toml_value "bittensor" "network")}
    BITTENSOR_NETUID=${BITTENSOR_NETUID:-$(parse_toml_value "bittensor" "netuid")}
    BITTENSOR_WALLET_NAME=${BITTENSOR_WALLET_NAME:-$(parse_toml_value "wallet" "name")}
    BITTENSOR_WALLET_HOTKEY_NAME=${BITTENSOR_WALLET_HOTKEY_NAME:-$(parse_toml_value "wallet" "hotkey_name")}
    BITTENSOR_WALLET_DIRECTORY=${BITTENSOR_WALLET_DIRECTORY:-$(parse_toml_value "wallet" "directory")}
fi

expand_path() {
    local input=$1
    if [ "${input#\~/}" != "${input}" ]; then
        printf "%s/%s" "$HOME" "${input#\~/}"
    elif [ "${input}" = "~" ]; then
        printf "%s" "$HOME"
    else
        printf "%s" "${input}"
    fi
}

WALLET_DIR_EXPANDED=$(expand_path "${BITTENSOR_WALLET_DIRECTORY}")
HOTKEY_PUB_FILE="${WALLET_DIR_EXPANDED}/${BITTENSOR_WALLET_NAME}/hotkeys/${BITTENSOR_WALLET_HOTKEY_NAME}pub.txt"
HOTKEY_SS58=""
if [ -f "${HOTKEY_PUB_FILE}" ]; then
    HOTKEY_SS58=$(sed -n 's/.*"ss58Address":"\([^"]*\)".*/\1/p' "${HOTKEY_PUB_FILE}")
fi
HOTKEY_IDENTIFIER=${HOTKEY_SS58:-${BITTENSOR_WALLET_HOTKEY_NAME}}

mkdir -p "${WORKING_DIRECTORY}/logs"
mkdir -p "${BRAIINSPROXY_CONFIG_DIR}"

BRAIINSPROXY_ACTIVE_PROFILE="${BRAIINSPROXY_CONFIG_DIR}/active_profile.toml"

if [ ! -f "${BRAIINSPROXY_ACTIVE_PROFILE}" ]; then
    echo "Creating Braiins Farm Proxy profile at ${BRAIINSPROXY_ACTIVE_PROFILE}..."
    DEFAULT_LUXOR_IDENTITY="sn${BITTENSOR_NETUID}auction.${HOTKEY_IDENTIFIER}.worker1"
    read -r -p "Enter Luxor user identity [${DEFAULT_LUXOR_IDENTITY}]: " LUXOR_USER_ID </dev/tty
    LUXOR_USER_ID=${LUXOR_USER_ID:-${DEFAULT_LUXOR_IDENTITY}}

    DEFAULT_MINER_IDENTITY="infinite.${HOTKEY_IDENTIFIER}.worker1"
    read -r -p "Enter MinerDefault target user identity [${DEFAULT_MINER_IDENTITY}]: " MINER_DEFAULT_ID </dev/tty
    MINER_DEFAULT_ID=${MINER_DEFAULT_ID:-${DEFAULT_MINER_IDENTITY}}

    DEFAULT_BACKUP_IDENTITY="backup.${HOTKEY_IDENTIFIER}.worker1"
    read -r -p "Enter MinerBackup target user identity [${DEFAULT_BACKUP_IDENTITY}]: " MINER_BACKUP_ID </dev/tty
    MINER_BACKUP_ID=${MINER_BACKUP_ID:-${DEFAULT_BACKUP_IDENTITY}}

    echo ""
    echo "Only one target can forward the ASIC identities (identity pass-through)."
    echo "You can adjust this later by editing ${BRAIINSPROXY_ACTIVE_PROFILE}."
    echo "Choose which target should use pass-through:"
    echo "  1) InfiniteHashLuxorTarget (default)"
    echo "  2) MinerDefaultTarget"
    echo "  3) MinerBackupTarget"
    echo "  4) None"
    read -r -p "Selection [1]: " PASSTHROUGH_CHOICE </dev/tty
    PASSTHROUGH_CHOICE=${PASSTHROUGH_CHOICE:-1}
    LUXOR_PASSTHROUGH_BOOL=false
    MINER_DEFAULT_PASSTHROUGH_BOOL=false
    MINER_BACKUP_PASSTHROUGH_BOOL=false
    case "${PASSTHROUGH_CHOICE}" in
        2) MINER_DEFAULT_PASSTHROUGH_BOOL=true ;;
        3) MINER_BACKUP_PASSTHROUGH_BOOL=true ;;
        4) ;;
        *) LUXOR_PASSTHROUGH_BOOL=true ;;
    esac

    cat > "${BRAIINSPROXY_ACTIVE_PROFILE}" <<EOL
[[server]]
name = "InfiniteHash"
port = 3333

[[target]]
name = "InfiniteHashLuxorTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "${LUXOR_USER_ID}"
identity_pass_through = ${LUXOR_PASSTHROUGH_BOOL}

[[target]]
name = "MinerDefaultTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "${MINER_DEFAULT_ID}"
identity_pass_through = ${MINER_DEFAULT_PASSTHROUGH_BOOL}

[[target]]
name = "MinerBackupTarget"
url = "stratum+tcp://btc.viabtc.io:3333"
user_identity = "${MINER_BACKUP_ID}"
identity_pass_through = ${MINER_BACKUP_PASSTHROUGH_BOOL}

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
