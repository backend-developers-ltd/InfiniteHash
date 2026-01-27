"""
Miner tasks for APScheduler.

Refactored from Django/Celery to use TOML config and APScheduler.
"""

import asyncio
import os
from collections import Counter
from typing import Any

import bittensor_wallet
import structlog
import turbobt

from infinite_hashes.auctions import utils as auction_utils
from infinite_hashes.consensus.bidding import (
    MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18,
    MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18,
    BiddingCommitment,
    select_auction_winners_async,
)
from infinite_hashes.consensus.price import _fp18_to_min_decimal_str, _parse_decimal_to_fp18_int

from .callbacks import handle_auction_results
from .config import MinerConfig
from .constants import (
    AUCTION_ILP_CBC_MAX_NODES,
    AUCTION_INTERVAL,
    BLOCK_TIME,
    COMMITMENT_RENEW_AGE_SECONDS,
    MAX_PRICE_MULTIPLIER,
    WINDOW_TRANSITION_THRESHOLD,
    WINDOW_TRANSITION_TIMEOUT,
)
from .models import AuctionResult, BidResult

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


def _build_commitment_from_config(config: MinerConfig) -> tuple[str, list[Any]]:
    price_fp18 = _parse_decimal_to_fp18_int(config.workers.price_multiplier)

    # Always use v2 commitments.
    # - workers.worker_sizes dict (or hashrates dict normalized into worker_sizes) => v2
    # - workers.hashrates list => aggregate into v2 counts
    if config.workers.worker_sizes is not None:
        v2_bids = [("BTC", price_fp18, dict(config.workers.worker_sizes))]
        commit = BiddingCommitment(t="b", bids=v2_bids, v=2)
        return commit.to_compact(), v2_bids

    allow_v1 = os.getenv("APS_MINER_ALLOW_V1", "").strip().lower() in {"1", "true", "yes", "on"}
    if allow_v1:
        bids = [(hr, price_fp18) for hr in (config.workers.hashrates or [])]
        commit = BiddingCommitment(t="b", bids=bids, v=1)
        return commit.to_compact(), bids

    worker_sizes: dict[str, int] = {}
    for hr in config.workers.hashrates or []:
        hr_fp = _parse_decimal_to_fp18_int(str(hr))
        if hr_fp <= 0:
            raise ValueError(f"invalid worker hashrate: {hr}")
        full = hr_fp // MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18
        remainder = hr_fp % MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18

        if remainder != 0 and remainder < MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18:
            if full <= 0:
                raise ValueError(f"worker hashrate below v2 min size ({MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18}): {hr}")
            delta = MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18 - remainder
            adjusted = MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18 - delta
            full -= 1
            remainder = MIN_BIDDING_COMMITMENT_WORKER_SIZE_FP18
            adj_key = _fp18_to_min_decimal_str(adjusted)
            worker_sizes[adj_key] = worker_sizes.get(adj_key, 0) + 1

        if full > 0:
            full_key = _fp18_to_min_decimal_str(MAX_BIDDING_COMMITMENT_WORKER_SIZE_FP18)
            worker_sizes[full_key] = worker_sizes.get(full_key, 0) + int(full)
        if remainder > 0:
            rem_key = _fp18_to_min_decimal_str(remainder)
            worker_sizes[rem_key] = worker_sizes.get(rem_key, 0) + 1

    v2_bids = [("BTC", price_fp18, worker_sizes)]
    commit = BiddingCommitment(t="b", bids=v2_bids, v=2)
    return commit.to_compact(), v2_bids


def _hashrate_to_fp18(value: Any) -> int | None:
    """Convert a hashrate representation to FP18 int, returning None on failure."""
    if value is None:
        return None
    try:
        return _parse_decimal_to_fp18_int(str(value))
    except Exception:
        logger.warning("Failed to normalize hashrate value", value=str(value))
        return None


def _reconcile_miner_bids(
    *,
    hotkey: str,
    commitment_bids: list[Any],
    winners: list[dict[str, Any]],
) -> list[BidResult]:
    """Expand this miner's commitment bids and mark them as won/lost.

    Important: winners can include multiple identical (hotkey, hashrate, price) entries
    (e.g. v2 hashrate->count), so we must treat winners as a multiset.
    """
    winner_counts: Counter[tuple[int, int]] = Counter()
    for item in winners or []:
        if (item.get("hotkey") or "") != hotkey:
            continue
        hashrate_fp = _hashrate_to_fp18(item.get("hashrate"))
        if hashrate_fp is None:
            continue
        try:
            price_fp18 = int(item.get("price"))
        except (TypeError, ValueError):
            continue
        winner_counts[(hashrate_fp, price_fp18)] += 1

    my_bids: list[BidResult] = []
    for bid in commitment_bids or []:
        if not isinstance(bid, list | tuple):
            continue

        count = 1
        if len(bid) == 3:
            hashrate, price_fp18, count = bid
        elif len(bid) == 2:
            hashrate, price_fp18 = bid
        else:
            continue

        hashrate_str = str(hashrate)
        worker_hashrate_fp = _hashrate_to_fp18(hashrate_str)
        try:
            price_fp18_int = int(price_fp18)
        except (TypeError, ValueError):
            logger.warning("Invalid price in commitment bid", price=price_fp18, hotkey=hotkey[:16])
            continue
        try:
            expanded = int(count)
        except (TypeError, ValueError):
            logger.warning("Invalid worker count in commitment bid", count=count, hotkey=hotkey[:16])
            continue
        if expanded <= 0:
            logger.warning("Invalid worker count in commitment bid", count=count, hotkey=hotkey[:16])
            continue

        won_n = 0
        if worker_hashrate_fp is not None:
            key = (worker_hashrate_fp, price_fp18_int)
            available = winner_counts.get(key, 0)
            if available > 0:
                won_n = min(expanded, available)
                remaining = available - won_n
                if remaining > 0:
                    winner_counts[key] = remaining
                else:
                    del winner_counts[key]

        for _ in range(won_n):
            my_bids.append(BidResult(hashrate=hashrate_str, price_fp18=price_fp18_int, won=True))
        for _ in range(expanded - won_n):
            my_bids.append(BidResult(hashrate=hashrate_str, price_fp18=price_fp18_int, won=False))

    return my_bids


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s, 16) if s.startswith("0x") else int(s)
        except ValueError:
            return None
    return None


def _commitment_age_seconds(current_block: int, commitment_block: int) -> int:
    if current_block <= commitment_block:
        return 0
    return (current_block - commitment_block) * BLOCK_TIME


async def _get_commitment_block(bittensor: turbobt.Bittensor, netuid: int, hotkey: str) -> int | None:
    subtensor = getattr(bittensor, "subtensor", None)
    state = getattr(subtensor, "state", None) if subtensor is not None else None
    if state is None:
        return None
    try:
        record = await state.getStorage("Commitments.CommitmentOf", netuid, hotkey)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch commitment metadata",
            error=str(exc),
            error_type=type(exc).__name__,
            hotkey=hotkey[:16],
        )
        return None
    if not record:
        return None
    if isinstance(record, dict):
        return _coerce_int(record.get("block"))
    if isinstance(record, list | tuple) and record:
        return _coerce_int(record[0])
    return _coerce_int(getattr(record, "block", None))


async def _get_current_commitment(
    bittensor: turbobt.Bittensor,
    netuid: int,
    hotkey: str,
) -> tuple[str | None, int | None]:
    """Fetch current on-chain commitment for this miner."""
    commitment_block = await _get_commitment_block(bittensor, netuid, hotkey)
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
            commit_block=commitment_block,
        )

        if commitment_data:
            # Decode bytes to string
            decoded = commitment_data.decode("utf-8")
            logger.info("Decoded commitment", value=decoded[:50])
            return decoded, commitment_block
        logger.info("No existing commitment found on-chain", hotkey=hotkey[:16])
        return None, commitment_block
    except Exception as e:
        logger.warning(
            "Failed to fetch current commitment from chain",
            error=str(e),
            error_type=type(e).__name__,
            hotkey=hotkey[:16],
        )
        return None, commitment_block


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

    worker_count = config.workers.total_workers()
    if worker_count <= 0 and not force:
        logger.debug("No workers configured; nothing to commit")
        return False

    # Build what we would commit
    try:
        compact, bids = _build_commitment_from_config(config)
    except ValueError:
        logger.exception("Invalid miner workers config; cannot build commitment")
        return False

    # Check if we need to update by comparing with on-chain commitment
    if not force:
        async with turbobt.Bittensor(config.bittensor.network) as bittensor:
            head = await bittensor.head.get()
            current_block = getattr(head, "number", None)
            current_commitment, commitment_block = await _get_current_commitment(
                bittensor,
                config.bittensor.netuid,
                hotkey,
            )
            commitment_age_sec = None
            commitment_stale = False
            if current_block is not None and commitment_block is not None:
                commitment_age_sec = _commitment_age_seconds(current_block, commitment_block)
                commitment_stale = commitment_age_sec >= COMMITMENT_RENEW_AGE_SECONDS

            logger.info(
                "Checking commitment change",
                current=current_commitment[:50] if current_commitment else None,
                new=compact[:50],
                changed=(current_commitment != compact),
                current_block=current_block,
                commitment_block=commitment_block,
                commitment_age_sec=commitment_age_sec,
                commitment_stale=commitment_stale,
            )

            if current_commitment == compact and not commitment_stale:
                logger.info("Commitment unchanged; skipping update")
                return False
            if current_commitment == compact and commitment_stale:
                logger.info("Commitment stale; refreshing", commitment_age_sec=commitment_age_sec)
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
        worker_count=worker_count,
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
    start_blk, bids_by_hotkey, _ = await auction_utils.fetch_bids_for_start_block(
        bittensor, subnet, start_block, config.bittensor.netuid
    )
    cbc_seed = auction_utils.cbc_seed_from_hash(start_blk.hash)
    winners, budget_ph = await select_auction_winners_async(
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

    commitment_bids = bids_by_hotkey.get(hotkey) or []
    if not commitment_bids:
        logger.warning(
            "No bidding commitment found for miner hotkey during window reconciliation",
            hotkey=hotkey[:16],
            start_block=start_block,
            end_block=end_block,
        )

    my_bids = _reconcile_miner_bids(hotkey=hotkey, commitment_bids=commitment_bids, winners=winners)

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
