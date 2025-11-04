"""
APScheduler setup for miner tasks.

Configures and runs the scheduler with periodic jobs for commitment and auction.
"""

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import MinerConfig
from .constants import AUCTION_INTERVAL, COMMITMENT_INTERVAL, JOB_TIMEOUT
from .executor import HardTimeoutExecutor
from .tasks import compute_current_auction, ensure_worker_commitment

logger = structlog.get_logger(__name__)


def create_scheduler(config_path: str) -> BackgroundScheduler:
    """
    Create and configure the APScheduler instance.

    Args:
        config_path: Path to TOML config file

    Returns:
        Configured BackgroundScheduler instance
    """
    executors = {"default": HardTimeoutExecutor()}

    job_defaults = {
        "max_instances": 1,  # Skip if previous instance still running
        "coalesce": True,  # Merge missed runs
        "misfire_grace_time": 10,  # Tolerate up to 10s late execution
    }

    scheduler = BackgroundScheduler(executors=executors, job_defaults=job_defaults)

    # Add commitment job - pass config_path, not config object
    scheduler.add_job(
        ensure_worker_commitment,
        trigger=IntervalTrigger(seconds=COMMITMENT_INTERVAL),
        kwargs={
            "config_path": config_path,
            "_exec": {"timeout": JOB_TIMEOUT},
        },
        id="ensure_commitment",
        name="Ensure Worker Commitment",
    )

    # Add auction computation job - pass config_path, not config object
    scheduler.add_job(
        compute_current_auction,
        trigger=IntervalTrigger(seconds=AUCTION_INTERVAL),
        kwargs={
            "config_path": config_path,
            "_exec": {"timeout": JOB_TIMEOUT},
        },
        id="compute_auction",
        name="Compute Current Auction",
    )

    logger.info(
        "Scheduler configured",
        commitment_interval=COMMITMENT_INTERVAL,
        auction_interval=AUCTION_INTERVAL,
        job_timeout=JOB_TIMEOUT,
    )

    return scheduler


def run_scheduler(config_path: str) -> None:
    """
    Start the scheduler and run until interrupted.

    Args:
        config_path: Path to TOML config file
    """
    # Load config once for initial logging
    config = MinerConfig.load(config_path)

    scheduler = create_scheduler(config_path)
    scheduler.start()

    logger.info(
        "Miner scheduler started",
        config_path=config_path,
        workers_count=len(config.workers.hashrates),
        price_multiplier=config.workers.price_multiplier,
    )

    # Run both jobs immediately on startup
    from datetime import datetime

    logger.info("Scheduling immediate execution of commitment and auction jobs...")
    scheduler.modify_job("ensure_commitment", next_run_time=datetime.now())
    scheduler.modify_job("compute_auction", next_run_time=datetime.now())

    try:
        # Keep the main thread alive
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
        logger.info("Scheduler stopped")
