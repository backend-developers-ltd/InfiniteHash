"""
Miner tasks for APScheduler.

Refactored from Django/Celery to use TOML config and APScheduler.
"""

import asyncio
from typing import Any

import bittensor_wallet
import structlog
import turbobt

from infinite_hashes.auctions import utils as auction_utils
from infinite_hashes.consensus.bidding import BiddingCommitment, select_auction_winners_async
from infinite_hashes.consensus.price import _parse_decimal_to_fp18_int

from .callbacks import handle_auction_results
from .config import MinerConfig
from .constants import (
    AUCTION_ILP_CBC_MAX_NODES,
    AUCTION_INTERVAL,
    BLOCK_TIME,
    MAX_PRICE_MULTIPLIER,
    WINDOW_TRANSITION_THRESHOLD,
    WINDOW_TRANSITION_TIMEOUT,
)
from .models import AuctionResult, BidResult, Worker

logger = structlog.get_logger(__name__)


def _load_wallet(config: MinerConfig) -> bittensor_wallet.Wallet:
    """Load wallet from configuration."""
    return bittensor_wallet.Wallet(
        name=config.wallet.name,
        hotkey=config.wallet.hotkey_name,
        path=str(config.wallet.directory_path),
    )


def _get_hotkey(config: MinerConfig) -> str | None:
    """Get hotkey SS58 address from wallet."""
    try:
        wallet = _load_wallet(config)
        return wallet.hotkey.ss58_address
    except Exception:
        logger.exception("Unable to load wallet for miner hotkey")
        return None


def _build_workers_from_config(config: MinerConfig) -> list[Worker]:
    """Build worker list from configuration."""
    workers = []
    for hashrate in config.workers.hashrates:
        worker = Worker(
            hashrate_compact=hashrate,
            price_multiplier=config.workers.price_multiplier,
            is_active=True,
        )
        workers.append(worker)
    return workers


def _build_commitment(workers: list[Worker]) -> tuple[str, list[tuple[str, int]]]:
    """
    Build commitment from workers.

    Copied from original tasks.py.
    """
    bids = [w.bid_tuple() for w in workers if w.is_active]
    commit = BiddingCommitment(t="b", bids=bids, v=1)
    compact = commit.to_compact()
    return compact, bids


def _hashrate_to_fp18(value: Any) -> int | None:
    """Convert a hashrate representation to FP18 int, returning None on failure."""
    if value is None:
        return None
    try:
        return _parse_decimal_to_fp18_int(str(value))
    except Exception:
        logger.warning("Failed to normalize hashrate value", value=str(value))
        return None


async def _get_current_commitment(bittensor: turbobt.Bittensor, netuid: int, hotkey: str) -> str | None:
    """Fetch current on-chain commitment for this miner."""
    try:
        subnet = await bittensor.subnet(netuid).get()
        # Fetch commitment data from chain
        commitment_data = await subnet.commitments.get(hotkey)

        logger.info(
            "Fetched commitment from chain",
            hotkey=hotkey[:16],
            data_type=type(commitment_data).__name__ if commitment_data else "None",
            data_len=len(commitment_data) if commitment_data else 0,
            has_data=bool(commitment_data),
        )

        if commitment_data:
            # Decode bytes to string
            decoded = commitment_data.decode("utf-8")
            logger.info("Decoded commitment", value=decoded[:50])
            return decoded
        logger.info("No existing commitment found on-chain", hotkey=hotkey[:16])
        return None
    except Exception as e:
        logger.warning(
            "Failed to fetch current commitment from chain",
            error=str(e),
            error_type=type(e).__name__,
            hotkey=hotkey[:16],
        )
        return None


async def _publish_commitment(config: MinerConfig, payload: bytes) -> None:
    """
    Publish commitment to chain.

    Copied from original tasks.py.
    """
    async with turbobt.Bittensor(config.bittensor.network) as bittensor:
        subnet = bittensor.subnet(config.bittensor.netuid)
        wallet = _load_wallet(config)
        extrinsic = await subnet.commitments.set(
            data=payload,
            wallet=wallet,
        )
        await extrinsic.wait_for_finalization()
        logger.info("Published miner bidding commitment", bytes_len=len(payload))


async def _ensure_worker_commitment_async(config: MinerConfig, force: bool = False) -> bool:
    """
    Ensure worker commitment is published on-chain.

    Args:
        config: Miner configuration
        force: Force commitment even if unchanged

    Returns:
        True if commitment was updated, False otherwise
    """
    hotkey = _get_hotkey(config)
    if not hotkey:
        logger.error("Cannot commit without hotkey")
        return False

    workers = _build_workers_from_config(config)
    if not workers and not force:
        logger.debug("No workers configured; nothing to commit")
        return False

    # Build what we would commit
    compact, bids = _build_commitment(workers)

    # Check if we need to update by comparing with on-chain commitment
    if not force:
        async with turbobt.Bittensor(config.bittensor.network) as bittensor:
            current_commitment = await _get_current_commitment(bittensor, config.bittensor.netuid, hotkey)

            logger.info(
                "Checking commitment change",
                current=current_commitment[:50] if current_commitment else None,
                new=compact[:50],
                changed=(current_commitment != compact),
            )

            if current_commitment == compact:
                logger.info("Commitment unchanged; skipping update")
                return False
            else:
                logger.info(
                    "Commitment changed, will update",
                    current_len=len(current_commitment) if current_commitment else 0,
                    new_len=len(compact),
                )

    # Publish the new commitment
    payload = compact.encode("utf-8")
    await _publish_commitment(config, payload)

    logger.info(
        "Miner commitment updated",
        bids=bids,
        force=force,
        worker_count=len(workers),
    )
    return True


def ensure_worker_commitment(config_path: str, force: bool = False, *, event_loop: Any = None, **kwargs) -> bool:
    """
    Sync wrapper for ensure_worker_commitment_async.

    Called by APScheduler job.

    Args:
        config_path: Path to TOML config file (loaded on each run)
        force: Force commitment even if unchanged
        event_loop: Optional event loop for test speedup with cached connection
        **kwargs: Additional kwargs (e.g., _exec for executor config) - ignored

    Returns:
        True if commitment was updated, False otherwise
    """
    from infinite_hashes.utils import run_async

    config = MinerConfig.load(config_path)
    return run_async(_ensure_worker_commitment_async, config, force, event_loop=event_loop)


async def _process_window(
    *,
    bittensor: turbobt.Bittensor,
    subnet: Any,
    config: MinerConfig,
    hotkey: str,
    start_block: int,
    end_block: int,
    epoch_start: int,
) -> AuctionResult:
    """
    Process auction window and determine winners.

    Refactored from original to return AuctionResult instead of saving to DB.
    """
    start_blk, bids_by_hotkey = await auction_utils.fetch_bids_for_start_block(
        bittensor, subnet, start_block, config.bittensor.netuid
    )
    cbc_seed = auction_utils.cbc_seed_from_hash(start_blk.hash)
    winners = await select_auction_winners_async(
        bittensor,
        config.bittensor.netuid,
        start_block,
        end_block,
        bids_by_hotkey,
        cbc_max_nodes=AUCTION_ILP_CBC_MAX_NODES,
        cbc_seed=cbc_seed,
        max_price_multiplier=MAX_PRICE_MULTIPLIER,
    )
    start_ts, end_ts = await auction_utils.window_timestamps(bittensor, start_block, end_block)

    # Build our workers for result reconciliation
    workers = _build_workers_from_config(config)

    # Determine which of our bids won
    my_bids: list[BidResult] = []
    winner_keys: set[tuple[str, int]] = set()
    for item in winners:
        winner_hashrate_fp = _hashrate_to_fp18(item.get("hashrate"))
        if winner_hashrate_fp is None:
            continue
        winner_keys.add((item.get("hotkey"), winner_hashrate_fp))

    for worker in workers:
        hashrate = worker.hashrate_compact
        worker_hashrate_fp = _hashrate_to_fp18(hashrate)
        won = (hotkey, worker_hashrate_fp) in winner_keys if worker_hashrate_fp is not None else False

        price_fp18 = worker.bid_tuple()[1]
        my_bids.append(BidResult(hashrate=hashrate, price_fp18=price_fp18, won=won))

    result = AuctionResult(
        epoch_start=epoch_start,
        start_block=start_block,
        end_block=end_block,
        window_start_time=start_ts,
        window_end_time=end_ts,
        commitments_count=len(bids_by_hotkey),
        all_winners=winners,
        my_bids=my_bids,
    )

    logger.info(
        "Processed auction window",
        start_block=start_block,
        end_block=end_block,
        winning_count=sum(1 for b in my_bids if b.won),
        commitments_count=len(bids_by_hotkey),
    )

    return result


async def _compute_current_auction_async(config: MinerConfig) -> dict[str, Any]:
    """
    Compute auction results for the current validation window.

    Implements smart window transition detection:
    - If close to next window (within 75% of schedule time), waits for window change
    - Polls for new window up to 80% of schedule time
    - Processes new window if detected, otherwise processes current window

    Refactored from original to use callbacks instead of DB.
    """
    hotkey = _get_hotkey(config)
    if not hotkey:
        logger.error("Cannot compute auction without hotkey")
        return {"error": "No hotkey available"}

    async with turbobt.Bittensor(config.bittensor.network) as bittensor:
        head, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(config.bittensor.netuid).get(),
        )

        current_block = head.number
        subnet_epoch_start = subnet.epoch(current_block).start
        blocks_per_window = auction_utils.blocks_per_window_default()

        # Calculate current window
        window_index, (window_start, window_end) = auction_utils.current_validation_window(
            current_block, subnet_epoch_start, blocks_per_window
        )

        # Calculate blocks until next window starts (window_end + 1)
        next_window_start = window_end + 1
        blocks_until_next_window = next_window_start - current_block
        time_until_next_window = blocks_until_next_window * BLOCK_TIME

        # Check if we're close to the next window transition
        threshold_time = AUCTION_INTERVAL * WINDOW_TRANSITION_THRESHOLD
        timeout_time = AUCTION_INTERVAL * WINDOW_TRANSITION_TIMEOUT

        if time_until_next_window <= threshold_time:
            # We're close to next window - actively wait for it
            logger.info(
                "Close to window transition, waiting for new window",
                current_block=current_block,
                blocks_until_next=blocks_until_next_window,
                time_until_next_s=time_until_next_window,
                threshold_s=threshold_time,
                timeout_s=timeout_time,
            )

            poll_interval = 1.0  # Poll every second
            wait_time = 0.0

            while wait_time < timeout_time:
                await asyncio.sleep(poll_interval)
                wait_time += poll_interval

                # Check current block again
                head = await bittensor.head.get()
                new_block = head.number

                # Recalculate window for new block
                new_window_index, (new_window_start, new_window_end) = auction_utils.current_validation_window(
                    new_block, subnet_epoch_start, blocks_per_window
                )

                # Check if we've transitioned to a new window
                if new_window_index != window_index:
                    logger.info(
                        "Window transition detected",
                        old_window=window_index,
                        new_window=new_window_index,
                        wait_time_s=wait_time,
                        new_block=new_block,
                    )
                    # Update to new window
                    window_index = new_window_index
                    window_start = new_window_start
                    window_end = new_window_end
                    current_block = new_block
                    break

            if wait_time >= timeout_time:
                logger.info(
                    "Window transition timeout, processing current window",
                    window_index=window_index,
                    wait_time_s=wait_time,
                )

        # Process the window (either current or newly detected)
        result = await _process_window(
            bittensor=bittensor,
            subnet=subnet,
            config=config,
            hotkey=hotkey,
            start_block=window_start,
            end_block=window_end,
            epoch_start=subnet_epoch_start,
        )

        # Call callback to handle results
        handle_auction_results(result)

        won_bids = [b for b in result.my_bids if b.won]
        logger.info(
            "Computed current window results",
            window_index=window_index,
            window_range=(window_start, window_end),
            winning_workers_count=len(won_bids),
        )

        return {
            "processed": True,
            "window_info": {
                "index": window_index,
                "start": window_start,
                "end": window_end,
                "epoch_start": subnet_epoch_start,
            },
            "won_bids": [{"hashrate": b.hashrate, "price_fp18": b.price_fp18} for b in won_bids],
        }


def compute_current_auction(config_path: str, *, event_loop: Any = None, **kwargs) -> dict[str, Any]:
    """
    Sync wrapper for compute_current_auction_async.

    Called by APScheduler job.

    Args:
        config_path: Path to TOML config file (loaded on each run)
        event_loop: Optional event loop for test speedup with cached connection
        **kwargs: Additional kwargs (e.g., _exec for executor config) - ignored

    Returns:
        Dict with window info and won bids
    """
    from infinite_hashes.utils import run_async

    config = MinerConfig.load(config_path)
    return run_async(_compute_current_auction_async, config, event_loop=event_loop)
