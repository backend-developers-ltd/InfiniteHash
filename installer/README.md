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
