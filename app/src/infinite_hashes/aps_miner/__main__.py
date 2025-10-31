"""
Main entry point for running the miner.

Usage:
    python -m infinite_hashes.aps_miner config.toml
"""

import sys
from pathlib import Path

import structlog

from .config import MinerConfig
from .scheduler import run_scheduler

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging_level=20),  # INFO
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger(__name__)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m infinite_hashes.aps_miner <config.toml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Validate config by loading it once
    logger.info("Loading configuration", config_path=str(config_path))
    config = MinerConfig.load(config_path)

    logger.info(
        "Starting miner",
        network=config.bittensor.network,
        netuid=config.bittensor.netuid,
        workers=len(config.workers.hashrates),
        price_multiplier=config.workers.price_multiplier,
    )

    # Pass config path, not config object - allows hot reload
    run_scheduler(str(config_path))


if __name__ == "__main__":
    main()
