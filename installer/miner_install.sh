#!/usr/bin/env bash
# Installer for APS miner deployment.

set -euo pipefail

ENV_NAME="${1:-prod}"
DEFAULT_WORKING_DIRECTORY=~/InfiniteHash-miner/
read -r -p "Enter WORKING_DIRECTORY [${DEFAULT_WORKING_DIRECTORY}]: " WORKING_DIRECTORY </dev/tty
WORKING_DIRECTORY=${WORKING_DIRECTORY:-${DEFAULT_WORKING_DIRECTORY}}
if [[ "${WORKING_DIRECTORY}" == ~* ]]; then
    WORKING_DIRECTORY="${WORKING_DIRECTORY/#\~/${HOME}}"
fi
GITHUB_URL="https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads"
MAX_V2_WORKER_SIZE_PH="0.45"
MAX_V2_TOTAL_WORKERS=1000

mkdir -p "${WORKING_DIRECTORY}"
WORKING_DIRECTORY=$(realpath "${WORKING_DIRECTORY}")

CONFIG_FILE="${WORKING_DIRECTORY}/config.toml"
BRAIINSPROXY_DIR="${WORKING_DIRECTORY}/brainsproxy"
BRAIINSPROXY_CONFIG_DIR="${BRAIINSPROXY_DIR}/config"
BRAIINSPROXY_ACTIVE_PROFILE="${BRAIINSPROXY_CONFIG_DIR}/active_profile.toml"
NEW_PROXY_DIR="${WORKING_DIRECTORY}/proxy"
NEW_PROXY_ENV_FILE="${NEW_PROXY_DIR}/.env"
NEW_PROXY_POOLS_FILE="${NEW_PROXY_DIR}/pools.toml"
PROXY_MODE_FILE="${WORKING_DIRECTORY}/proxy_mode"

write_default_ihp_env() {
    local destination="$1"
    cat > "${destination}" <<'EOL'
IHP_POOL_CONFIG_PATH=/etc/infinite-hash-proxy/pools.toml
IHP_LISTEN_ADDR=0.0.0.0:3333
IHP_METRICS_ADDR=0.0.0.0:9090

IHP_DATABASE_URL=postgresql+asyncpg://ihp:ihp@ihp-postgres:5432/ihp
IHP_COMPACTION_ENABLED=true
IHP_COMPACTION_INTERVAL_SECONDS=3600
IHP_COMPACTION_RETENTION_HOURS=72

IHP_API_HOST=0.0.0.0
IHP_API_PORT=8000
IHP_API_LOG_LEVEL=INFO
IHP_API_DATABASE_URL=postgresql+asyncpg://ihp:ihp@ihp-postgres:5432/ihp
EOL
}

write_default_ihp_pools() {
    local destination="$1"
    local backup_pool_host="$2"
    local backup_pool_port="$3"
    local backup_pool_worker_id="${4:-}"
    local main_pool_worker_id="${5:-}"

    cat > "${destination}" <<EOL
[pools]

[pools.backup]
name = "private-backup"
host = "${backup_pool_host}"
port = ${backup_pool_port}
EOL

    if [ -n "${backup_pool_worker_id}" ]; then
        cat >> "${destination}" <<EOL
worker_id = "${backup_pool_worker_id}"
EOL
    fi

    cat >> "${destination}" <<EOL
[[pools.main]]
name = "central-proxy"
host = "stratum.infinitehash.xyz"
port = 9332
weight = 1
EOL

    if [ -n "${main_pool_worker_id}" ]; then
        cat >> "${destination}" <<EOL
worker_id = "${main_pool_worker_id}"
EOL
    fi

    cat >> "${destination}" <<EOL

[extranonce]
extranonce2_size = 2

[routing]
rebalance_interval_seconds = 10
grace_period_seconds = 60
min_shares_for_estimate = 20
rebalance_threshold_percent = 10
min_reassign_interval_seconds = 45
pool_connect_timeout_seconds = 10
pool_read_timeout_seconds = 300
pool_unhealthy_cooldown_seconds = 10
worker_assignment_stale_threshold_seconds = 30
disconnected_worker_retention_seconds = 900

[filters]
ban_nicehash = true
EOL
}

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

    echo "Configure workers using v2 grouped format."
    echo "Format: hashratePHxcount pairs separated by commas (example: 0.45x10,0.25x4)."
    echo "Limits: hashrate > 0 and <= ${MAX_V2_WORKER_SIZE_PH} PH, total workers <= ${MAX_V2_TOTAL_WORKERS}."
    echo ""

    WORKER_SIZE_TOML_LINES=""
    while true; do
        read -r -p "Enter worker sizes (v2) []: " WORKER_SIZES_INPUT </dev/tty
        WORKER_SIZES_INPUT=$(echo "${WORKER_SIZES_INPUT}" | tr -d '[:space:]')

        if [ -z "${WORKER_SIZES_INPUT}" ]; then
            echo "Worker sizes input cannot be empty."
            continue
        fi

        IFS=',' read -ra WORKER_SIZE_ITEMS <<< "${WORKER_SIZES_INPUT}"
        if [ "${#WORKER_SIZE_ITEMS[@]}" -eq 0 ]; then
            echo "No worker sizes provided."
            continue
        fi

        declare -A SEEN_HASHRATES=()
        WORKER_SIZE_TOML_LINES=""
        TOTAL_WORKERS=0
        TOTAL_HASHRATE_PH="0"
        PARSE_ERROR=""

        for item in "${WORKER_SIZE_ITEMS[@]}"; do
            if [[ ! "${item}" =~ ^([0-9]+([.][0-9]+)?)x([0-9]+)$ ]]; then
                PARSE_ERROR="Invalid item '${item}'. Expected format hashratePHxcount (e.g. 0.45x10)."
                break
            fi

            HASHRATE_PH="${BASH_REMATCH[1]}"
            WORKER_COUNT="${BASH_REMATCH[3]}"

            if ! awk -v h="${HASHRATE_PH}" -v max="${MAX_V2_WORKER_SIZE_PH}" 'BEGIN { exit !(h > 0 && h <= max) }'; then
                PARSE_ERROR="Invalid hashrate '${HASHRATE_PH}' in '${item}'. Allowed range is (0, ${MAX_V2_WORKER_SIZE_PH}] PH."
                break
            fi

            if [ "${WORKER_COUNT}" -le 0 ]; then
                PARSE_ERROR="Invalid worker count '${WORKER_COUNT}' in '${item}'. Count must be > 0."
                break
            fi

            if [ -n "${SEEN_HASHRATES[${HASHRATE_PH}]+x}" ]; then
                PARSE_ERROR="Duplicate hashrate '${HASHRATE_PH}' found. Use each hashrate only once."
                break
            fi
            SEEN_HASHRATES["${HASHRATE_PH}"]=1

            TOTAL_WORKERS=$((TOTAL_WORKERS + WORKER_COUNT))
            if [ "${TOTAL_WORKERS}" -gt "${MAX_V2_TOTAL_WORKERS}" ]; then
                PARSE_ERROR="Total workers ${TOTAL_WORKERS} exceeds limit ${MAX_V2_TOTAL_WORKERS}."
                break
            fi

            TOTAL_HASHRATE_PH=$(awk -v total="${TOTAL_HASHRATE_PH}" -v h="${HASHRATE_PH}" -v c="${WORKER_COUNT}" 'BEGIN { printf "%.8f", total + (h * c) }')

            if [ -z "${WORKER_SIZE_TOML_LINES}" ]; then
                WORKER_SIZE_TOML_LINES="\"${HASHRATE_PH}\" = ${WORKER_COUNT}"
            else
                WORKER_SIZE_TOML_LINES="${WORKER_SIZE_TOML_LINES}
\"${HASHRATE_PH}\" = ${WORKER_COUNT}"
            fi
        done

        if [ -n "${PARSE_ERROR}" ]; then
            echo "${PARSE_ERROR}"
            continue
        fi

        echo "Parsed workers: ${TOTAL_WORKERS}"
        echo "Total hashrate: ${TOTAL_HASHRATE_PH} PH"
        echo "Entries:"
        printf "%s\n" "${WORKER_SIZE_TOML_LINES}"
        read -r -p "Confirm worker sizes? [Y/n]: " WORKER_SIZES_CONFIRM </dev/tty
        WORKER_SIZES_CONFIRM=${WORKER_SIZES_CONFIRM:-Y}
        case "${WORKER_SIZES_CONFIRM}" in
            [Yy]|[Yy][Ee][Ss]) break ;;
            *) echo "Re-enter worker sizes." ;;
        esac
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

[workers.worker_sizes]
${WORKER_SIZE_TOML_LINES}
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

PROXY_MODE=""
if [ -f "${PROXY_MODE_FILE}" ]; then
    PROXY_MODE=$(tr -d '[:space:]' < "${PROXY_MODE_FILE}" | tr '[:upper:]' '[:lower:]')
fi

if [ "${PROXY_MODE}" != "ihp" ] && [ "${PROXY_MODE}" != "braiins" ]; then
    DEFAULT_MODE_SELECTION=1
    if [ -f "${BRAIINSPROXY_ACTIVE_PROFILE}" ]; then
        DEFAULT_MODE_SELECTION=2
        echo "Detected existing Braiins profile. Defaulting selection to Braiins to avoid migration."
    fi

    echo ""
    echo "Choose proxy mode:"
    echo "  1) InfiniteHash Proxy (ihp, default)"
    echo "  2) Braiins Farm Proxy (braiins)"
    read -r -p "Selection [${DEFAULT_MODE_SELECTION}]: " PROXY_MODE_SELECTION </dev/tty
    PROXY_MODE_SELECTION=${PROXY_MODE_SELECTION:-${DEFAULT_MODE_SELECTION}}

    case "${PROXY_MODE_SELECTION}" in
        2) PROXY_MODE="braiins" ;;
        *) PROXY_MODE="ihp" ;;
    esac

    printf "%s\n" "${PROXY_MODE}" > "${PROXY_MODE_FILE}"
    echo "Proxy mode selected: ${PROXY_MODE}"
else
    echo "Using existing proxy mode from ${PROXY_MODE_FILE}: ${PROXY_MODE}"
fi

mkdir -p "${WORKING_DIRECTORY}/logs"

if [ "${PROXY_MODE}" = "ihp" ]; then
    mkdir -p "${NEW_PROXY_DIR}"

    if [ ! -f "${NEW_PROXY_ENV_FILE}" ]; then
        echo "Creating InfiniteHash Proxy environment file at ${NEW_PROXY_ENV_FILE}..."
        write_default_ihp_env "${NEW_PROXY_ENV_FILE}"
    else
        echo "Using existing InfiniteHash Proxy environment file at ${NEW_PROXY_ENV_FILE}"
    fi

    if [ ! -f "${NEW_PROXY_POOLS_FILE}" ]; then
        echo "Creating InfiniteHash Proxy pools file at ${NEW_PROXY_POOLS_FILE}..."
        DEFAULT_PRIVATE_POOL_HOST="btc.global.luxor.tech"
        read -r -p "Enter IHP backup/private pool host [${DEFAULT_PRIVATE_POOL_HOST}]: " PRIVATE_POOL_HOST </dev/tty
        PRIVATE_POOL_HOST=${PRIVATE_POOL_HOST:-${DEFAULT_PRIVATE_POOL_HOST}}

        DEFAULT_PRIVATE_POOL_PORT="700"
        read -r -p "Enter IHP backup/private pool port [${DEFAULT_PRIVATE_POOL_PORT}]: " PRIVATE_POOL_PORT </dev/tty
        PRIVATE_POOL_PORT=${PRIVATE_POOL_PORT:-${DEFAULT_PRIVATE_POOL_PORT}}
        PRIVATE_POOL_PORT=$(echo "${PRIVATE_POOL_PORT}" | tr -d '[:space:]')

        read -r -p "Enter IHP backup/private pool worker ID override (optional) []: " PRIVATE_POOL_WORKER_ID </dev/tty
        PRIVATE_POOL_WORKER_ID=$(echo "${PRIVATE_POOL_WORKER_ID}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        DEFAULT_MAIN_POOL_WORKER_ID="sn${BITTENSOR_NETUID}auction.${HOTKEY_IDENTIFIER}.worker1"
        echo ""
        echo "Configure IHP central-proxy worker ID override:"
        echo "  1) Use suggested value (${DEFAULT_MAIN_POOL_WORKER_ID})"
        echo "  2) Set manually (leave empty for no override)"
        read -r -p "Selection [1]: " MAIN_POOL_WORKER_ID_MODE </dev/tty
        MAIN_POOL_WORKER_ID_MODE=${MAIN_POOL_WORKER_ID_MODE:-1}

        case "${MAIN_POOL_WORKER_ID_MODE}" in
            2)
                read -r -p "Enter IHP central-proxy worker ID override (optional) []: " MAIN_POOL_WORKER_ID </dev/tty
                MAIN_POOL_WORKER_ID=$(echo "${MAIN_POOL_WORKER_ID}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                ;;
            *)
                MAIN_POOL_WORKER_ID="${DEFAULT_MAIN_POOL_WORKER_ID}"
                ;;
        esac

        write_default_ihp_pools "${NEW_PROXY_POOLS_FILE}" "${PRIVATE_POOL_HOST}" "${PRIVATE_POOL_PORT}" "${PRIVATE_POOL_WORKER_ID}" "${MAIN_POOL_WORKER_ID}"
    else
        echo "Using existing InfiniteHash Proxy pools file at ${NEW_PROXY_POOLS_FILE}"
    fi
else
    mkdir -p "${BRAIINSPROXY_CONFIG_DIR}"
fi

if [ "${PROXY_MODE}" = "braiins" ] && [ ! -f "${BRAIINSPROXY_ACTIVE_PROFILE}" ]; then
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

echo "Running update_miner_compose.sh once to ensure it works..."
curl -s "${GITHUB_URL}/deploy-config-${ENV_NAME}/installer/update_miner_compose.sh" > /tmp/update_miner_compose.sh
chmod +x /tmp/update_miner_compose.sh
if ! /tmp/update_miner_compose.sh "${ENV_NAME}" "${WORKING_DIRECTORY}"; then
    echo "Error: update_miner_compose.sh failed. Not adding cron line."
    exit 1
fi
echo "update_miner_compose.sh ran successfully."

echo "Pulling latest images and recreating services..."
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    if ! (cd "${WORKING_DIRECTORY}" && docker compose pull && docker compose up -d --remove-orphans); then
        echo "Error: docker compose pull/up failed."
        exit 1
    fi
elif command -v docker-compose &> /dev/null; then
    if ! (cd "${WORKING_DIRECTORY}" && docker-compose pull && docker-compose up -d --remove-orphans); then
        echo "Error: docker-compose pull/up failed."
        exit 1
    fi
else
    echo "Error: Neither docker compose nor docker-compose is available."
    exit 1
fi
echo "Latest images applied successfully."

CRON_CMD="*/15 * * * * cd ${WORKING_DIRECTORY} && curl -s ${GITHUB_URL}/deploy-config-${ENV_NAME}/installer/update_miner_compose.sh > /tmp/update_miner_compose.sh && chmod +x /tmp/update_miner_compose.sh && /tmp/update_miner_compose.sh ${ENV_NAME} ${WORKING_DIRECTORY} # INFINITE_HASH_APS_MINER_UPDATE"

(crontab -l 2>/dev/null || echo "") | grep -v "INFINITE_HASH_APS_MINER_UPDATE" | { cat; echo "${CRON_CMD}"; } | crontab -

echo "Cron job installed successfully. It will run every 15 minutes."
echo "Environment: ${ENV_NAME}"
echo "Working directory: ${WORKING_DIRECTORY}"
echo "Proxy mode: ${PROXY_MODE}"
