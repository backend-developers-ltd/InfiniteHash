import asyncio
from typing import Any

import structlog
import turbobt

from infinite_hashes.consensus.bidding import BiddingCommitment
from infinite_hashes.consensus.parser import parse_commitment

logger = structlog.get_logger(__name__)

__all__ = [
    "parse_bidding_commitments",
    "cbc_seed_from_hash",
    "fetch_bids_for_start_block",
    "window_timestamps",
    "blocks_per_window_default",
    "windows_for_epoch",
    "validation_windows_for_subnet_epoch",
    "current_validation_window",
    "validation_epoch_for_subnet_epoch",
]

COMMITMENT_MAX_AGE_MINUTES = 150
COMMITMENT_MAX_AGE_SECONDS = COMMITMENT_MAX_AGE_MINUTES * 60
COMMITMENT_BLOCK_TIME_SECONDS = 12


def parse_bidding_commitments(
    commits: dict[str, bytes | str],
) -> dict[str, list[tuple[str, int] | tuple[str, int, int]]]:
    """Parse bidding commitments, safely handling potential binary suffixes.

    Returns per-hotkey bid lists in normalized forms:
    - v1: (hashrate_str, price_fp18_int)
    - v2: (hashrate_str, price_fp18_int, count_int)
    """
    out: dict[str, list[tuple[str, int] | tuple[str, int, int]]] = {}
    for hotkey, raw in commits.items():
        if raw is None:
            continue

        # Use generic parser with type filtering (auto-discovers "b" token)
        model = parse_commitment(raw, expected_types=[BiddingCommitment])
        if model is None:
            continue

        try:
            if int(getattr(model, "v", 1) or 1) >= 2:
                bids_v2: list[tuple[str, int, int]] = []
                for algo, price, hr_map in getattr(model, "bids", []) or []:
                    if algo != "BTC":
                        raise ValueError("non-BTC bidding is not supported")
                    price_fp18 = int(price)
                    for hr, count in (hr_map or {}).items():
                        bids_v2.append((str(hr), price_fp18, int(count)))
                out[hotkey] = bids_v2
            else:
                out[hotkey] = list(model.bids or [])
        except (TypeError, ValueError):
            # Any malformed commitment is ignored as a whole.
            continue

    return out


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
    return (current_block - commitment_block) * COMMITMENT_BLOCK_TIME_SECONDS


async def _fetch_commitment_block(
    state: Any,
    *,
    netuid: int,
    hotkey: str,
    block_hash: str,
) -> int | None:
    try:
        record = await state.getStorage("Commitments.CommitmentOf", netuid, hotkey, block_hash=block_hash)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch commitment metadata",
            hotkey=hotkey[:16],
            error=str(exc),
        )
        return None

    if not record:
        return None
    if isinstance(record, dict):
        return _coerce_int(record.get("block"))
    if isinstance(record, list | tuple) and record:
        return _coerce_int(record[0])
    return _coerce_int(getattr(record, "block", None))


async def _filter_commitments_by_age(
    *,
    state: Any,
    netuid: int,
    block_hash: str,
    current_block: int,
    bids_by_hotkey: dict[str, list[tuple[str, int] | tuple[str, int, int]]],
) -> tuple[dict[str, list[tuple[str, int] | tuple[str, int, int]]], set[str]]:
    if not bids_by_hotkey:
        return bids_by_hotkey, set()

    hotkeys = list(bids_by_hotkey.keys())
    blocks = await asyncio.gather(
        *(
            _fetch_commitment_block(
                state,
                netuid=netuid,
                hotkey=hotkey,
                block_hash=block_hash,
            )
            for hotkey in hotkeys
        ),
        return_exceptions=True,
    )

    filtered: dict[str, list[tuple[str, int] | tuple[str, int, int]]] = {}
    stale: set[str] = set()
    for hotkey, block in zip(hotkeys, blocks):
        if isinstance(block, Exception):
            logger.warning(
                "Failed to fetch commitment block",
                hotkey=hotkey[:16],
                error=str(block),
            )
            filtered[hotkey] = bids_by_hotkey[hotkey]
            continue
        commitment_block = block
        if commitment_block is None:
            filtered[hotkey] = bids_by_hotkey[hotkey]
            continue
        age_sec = _commitment_age_seconds(current_block, commitment_block)
        if age_sec >= COMMITMENT_MAX_AGE_SECONDS:
            stale.add(hotkey)
            continue
        filtered[hotkey] = bids_by_hotkey[hotkey]

    return filtered, stale


def cbc_seed_from_hash(h: str) -> int:
    try:
        return (int(str(h).strip().lstrip("0x")[-16:], 16) % 2147483647) + 1
    except Exception:
        return 1


async def fetch_bids_for_start_block(
    bittensor: turbobt.Bittensor,
    subnet: Any,
    start_block: int,
    netuid: int,
) -> tuple[Any, dict[str, list[tuple[str, int] | tuple[str, int, int]]], set[str]]:
    """Fetch bids for a start block and filter out banned hotkeys based on consensus.

    Args:
        bittensor: Bittensor instance
        subnet: Subnet instance
        start_block: Block number to fetch bids for
        netuid: Network UID for ban consensus computation

    Returns:
        Tuple of (block, bids_by_hotkey, banned_hotkeys) where bids from banned hotkeys are excluded
    """
    from infinite_hashes.consensus.price import compute_ban_consensus

    start_blk = await bittensor.block(start_block).get()
    commits_raw = await subnet.commitments.fetch(block_hash=start_blk.hash)
    bids_by_hotkey = parse_bidding_commitments(commits_raw)

    state = getattr(getattr(bittensor, "subtensor", None), "state", None)
    if state is not None and bids_by_hotkey:
        bids_by_hotkey, stale_hotkeys = await _filter_commitments_by_age(
            state=state,
            netuid=netuid,
            block_hash=start_blk.hash,
            current_block=start_block,
            bids_by_hotkey=bids_by_hotkey,
        )
        if stale_hotkeys:
            logger.info(
                "Filtered stale commitments",
                block=start_block,
                total_bids=len(commits_raw),
                stale_count=len(stale_hotkeys),
                filtered_bids=len(bids_by_hotkey),
            )

    # Compute ban consensus for this block
    banned_hotkeys = await compute_ban_consensus(
        netuid=netuid,
        block_number=start_block,
        bt=bittensor,
    )

    # Filter out bids from banned hotkeys
    if banned_hotkeys:
        filtered_bids = {hotkey: bids for hotkey, bids in bids_by_hotkey.items() if hotkey not in banned_hotkeys}
        logger.info(
            "Filtered banned hotkeys from bids",
            block=start_block,
            total_bids=len(bids_by_hotkey),
            banned_count=len(banned_hotkeys),
            filtered_bids=len(filtered_bids),
        )
        return start_blk, filtered_bids, set(banned_hotkeys)

    return start_blk, bids_by_hotkey, set()


async def window_timestamps(bittensor: turbobt.Bittensor, start_block: int, end_block: int):
    return await asyncio.gather(
        (await bittensor.block(start_block).get()).get_timestamp(),
        (await bittensor.block(end_block).get()).get_timestamp(),
    )


def blocks_per_window_default() -> int:
    return 60  # Standard auction window size


def windows_for_epoch(epoch_start: int, epoch_tempo: int, blocks_per_window: int) -> list[tuple[int, int]]:
    start = epoch_start
    end_epoch = epoch_start + epoch_tempo
    windows: list[tuple[int, int]] = []
    cur = start
    while cur <= end_epoch:
        w_end = min(cur + blocks_per_window - 1, end_epoch)
        windows.append((cur, w_end))
        cur = w_end + 1

    # Merge small runt windows into the last full window
    # If the last window is very small (< 50% of normal size), extend the previous window
    if len(windows) >= 2:
        last_start, last_end = windows[-1]
        last_size = last_end - last_start + 1
        if last_size < blocks_per_window * 0.5:
            # Remove the runt and extend the previous window to include it
            windows.pop()
            prev_start, _prev_end = windows[-1]
            windows[-1] = (prev_start, last_end)

    return windows


def validation_epoch_for_subnet_epoch(subnet_epoch_start: int, blocks_per_window: int) -> tuple[int, int]:
    """Calculate validation epoch boundaries for a given subnet epoch.

    Validation epoch starts one window (60 blocks) before subnet epoch.
    Total duration: 361 blocks (6 windows: 60+60+60+60+60+61).

    Args:
        subnet_epoch_start: Start block of the subnet epoch
        blocks_per_window: Blocks per window (typically 60)

    Returns:
        Tuple of (validation_epoch_start, validation_epoch_end)
    """
    validation_start = subnet_epoch_start - blocks_per_window
    validation_end = validation_start + 360  # 361 blocks total (start + 360 = end inclusive)
    return validation_start, validation_end


def validation_windows_for_subnet_epoch(subnet_epoch_start: int, blocks_per_window: int) -> list[tuple[int, int]]:
    """Calculate validation windows for a subnet epoch.

    Returns 6 windows starting 60 blocks before subnet epoch:
    - Window 0: 60 blocks (ends 1 block before subnet epoch)
    - Window 1: 60 blocks (starts at subnet epoch)
    - Window 2-4: 60 blocks each
    - Window 5: 61 blocks (the trailing +1)

    Args:
        subnet_epoch_start: Start block of the subnet epoch
        blocks_per_window: Blocks per window (typically 60)

    Returns:
        List of (start_block, end_block) tuples for each window
    """
    validation_start, validation_end = validation_epoch_for_subnet_epoch(subnet_epoch_start, blocks_per_window)

    windows = []
    cur = validation_start

    # Windows 0-4: 60 blocks each
    for _ in range(5):
        windows.append((cur, cur + blocks_per_window - 1))
        cur += blocks_per_window

    # Window 5: remaining blocks (should be 61)
    windows.append((cur, validation_end))

    return windows


def current_validation_window(
    current_block: int,
    subnet_epoch_start: int,
    blocks_per_window: int,
) -> tuple[int, tuple[int, int]]:
    """Determine which validation window the current block is in.

    Handles blocks across multiple validation epochs by calculating which
    epoch the block belongs to, then which window within that epoch.

    Each validation epoch is 361 blocks (6 windows: 60+60+60+60+60+61).

    Args:
        current_block: Current block number
        subnet_epoch_start: Start block of ANY subnet epoch (used as reference)
        blocks_per_window: Blocks per window (typically 60)

    Returns:
        Tuple of (window_index, (start_block, end_block)) where window_index is 0-5
    """
    # Validation epoch for the reference subnet epoch
    reference_validation_start = subnet_epoch_start - blocks_per_window

    # Total blocks in a validation epoch: 60*5 + 61 = 361
    validation_epoch_length = blocks_per_window * 5 + (blocks_per_window + 1)

    # Find offset from reference validation epoch start
    offset_from_reference = current_block - reference_validation_start

    # Find which validation epoch we're in (could be before, same, or after reference)
    epoch_offset = offset_from_reference // validation_epoch_length
    if offset_from_reference < 0:
        epoch_offset = -((-offset_from_reference + validation_epoch_length - 1) // validation_epoch_length)

    # Calculate the actual validation epoch start for current block
    actual_validation_start = reference_validation_start + (epoch_offset * validation_epoch_length)

    # Now calculate window index within this validation epoch
    offset_in_epoch = current_block - actual_validation_start

    # Calculate window index (0-5)
    # Windows 0-4 are 60 blocks each, window 5 is 61 blocks
    if offset_in_epoch < blocks_per_window * 5:
        window_index = offset_in_epoch // blocks_per_window
    else:
        window_index = 5

    # Calculate window boundaries
    window_start = actual_validation_start + (window_index * blocks_per_window)

    # Window 5 is 61 blocks (the +1), all others are 60 blocks
    if window_index == 5:
        window_end = window_start + blocks_per_window  # 61 blocks
    else:
        window_end = window_start + blocks_per_window - 1  # 60 blocks

    return window_index, (window_start, window_end)
