# InfiniteHash Validator Installer

This directory contains scripts to install and maintain a InfiniteHash validator node.

## Quick Installation

You can install the InfiniteHash validator with a single command:

```bash
curl -s https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads/deploy-config-prod/installer/install.sh | bash
```

This will:
1. Create a working directory at `~/InfiniteHash-validator/` (default)
2. Prompt you for configuration values if needed
3. Set up a cron job to automatically update your validator when changes are published
4. Download and start the validator services using Docker Compose

## Custom Installation

If you want to customize the installation, you can pass arguments to the script:

```bash
curl -s https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads/deploy-config-prod/installer/install.sh | bash -s -- [ENV_NAME] [WORKING_DIRECTORY]
```

Where:
- `ENV_NAME`: The environment to use (defaults to "prod")
- `WORKING_DIRECTORY`: The directory where the validator will be installed (defaults to `~/InfiniteHash-validator/`)

Example with custom working directory:

```bash
curl -s https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads/deploy-config-prod/installer/install.sh | bash -s -- prod /opt/InfiniteHash-validator
```

## Prerequisites

- Docker and Docker Compose installed
- cron running
- curl installed
- bash shell
- Internet access to GitHub

## What the Installer Does

1. Creates a working directory
2. Sets up a `.env` file with your configuration
3. Downloads the latest `docker-compose.yml` file
4. Sets up a cron job to check for updates every 15 minutes
5. Starts the validator services

## Manual Update

If you want to manually update your validator, you can run:

```bash
curl -s https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads/deploy-config-prod/installer/update_compose.sh | bash -s -- [ENV_NAME] [WORKING_DIRECTORY]
```

## APS Miner Installation

An installer for the APS miner (APScheduler-based miner) is also available:

```bash
curl -s https://raw.githubusercontent.com/backend-developers-ltd/InfiniteHash/refs/heads/deploy-config-prod/installer/miner_install.sh | bash
```

This script prompts for TOML configuration values and proxy mode, generates `config.toml`, deploys the miner stack with Docker Compose, and installs a cron job that keeps the compose file up to date by calling `installer/update_miner_compose.sh`.

For `ihp` mode, default proxy template files (`proxy/.env` and `proxy/pools.toml`) are generated directly by the installer script (inline templates), not downloaded from external template files.

Proxy mode options:
- `InfiniteHash Proxy` (`ihp`, default on Enter)
- `Braiins Farm Proxy` (`braiins`, optional)

In `braiins` mode, the installer provisions Braiins Farm Proxy (`farm-proxy` and configurator sidecar) and keeps `brainsproxy/active_profile.toml` in sync with APS miner auction results.

### Routing identifiers by mode

`ihp` mode:
- APS miner updates subnet allocation by pool `name` in `proxy/pools.toml` (`[[pools.main]]` entry), not by host/port.
- The miner container uses `APS_MINER_SUBNET_POOL_NAME` (default: `central-proxy`) to select which pool gets absolute `target_hashrate`.
- The selected pool name must exist in `proxy/pools.toml`; otherwise no subnet target update is applied.
- During installation, if `proxy/pools.toml` does not exist yet, the script asks for backup/private pool host/port and writes those values to `pools.backup`.
- Installer default sets `[extranonce].extranonce2_size = 2`.
- After updating `proxy/pools.toml`, APS miner touches reload sentinel `APS_MINER_IHP_RELOAD_SENTINEL` (default: `/root/src/proxy/.reload-ihp`); sidecar `ihp-proxy-reloader` then runs `kill -HUP 1` in `ihp-proxy` PID namespace.

`braiins` mode:
- APS miner updates routing by Braiins goal names in `brainsproxy/active_profile.toml`.
- Expected goal names are `InfiniteHashLuxorGoal` (subnet side) and `MinerDefaultGoal` (non-subnet side).
- If you rename these goals manually, APS miner will not match them until code/config is aligned.
