"""
Callbacks for handling auction results.

These functions are called when we determine which workers won/lost auctions.
Future: integrate with ASIC routing logic.
"""

import structlog

from .brainsproxy import update_routing_weights
from .models import AuctionResult, BidResult

logger = structlog.get_logger(__name__)


def handle_auction_results(result: AuctionResult) -> None:
    """
    Handle auction results - called after computing which workers won/lost.

    This is the main entry point for routing integration.

    Args:
        result: Complete auction result with all bids and outcomes

    Future integration:
        - Route winning workers to mining pool
        - Route losing workers to alternative tasks or idle
        - Update ASIC configurations based on allocation
    """
    won_bids = [b for b in result.my_bids if b.won]
    lost_bids = [b for b in result.my_bids if not b.won]

    logger.info(
        "Auction results computed",
        epoch_start=result.epoch_start,
        window_range=(result.start_block, result.end_block),
        total_commitments=result.commitments_count,
        my_bids_total=len(result.my_bids),
        won_count=len(won_bids),
        lost_count=len(lost_bids),
    )

    if won_bids:
        _handle_won_bids(won_bids, result)

    if lost_bids:
        _handle_lost_bids(lost_bids, result)

    try:
        update_routing_weights(won_bids, lost_bids)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update Braiins routing weights")


def _handle_won_bids(won_bids: list[BidResult], result: AuctionResult) -> None:
    """
    Handle workers that won their bids.

    Future integration:
        - Configure these workers to mine for the subnet
        - Set up routing to direct hashrate to winning pool
        - Start monitoring and reporting for these workers
    """
    logger.info(
        "Processing winning bids",
        count=len(won_bids),
        hashrates=[b.hashrate for b in won_bids],
        window_start=result.window_start_time.isoformat(),
        window_end=result.window_end_time.isoformat(),
    )

    # TODO: Integration point for ASIC routing
    # For each won bid:
    # 1. Identify the physical ASIC/worker by hashrate
    # 2. Configure routing to subnet mining pool
    # 3. Start monitoring hashrate delivery
    # 4. Report any issues

    for bid in won_bids:
        logger.debug(
            "Won bid details",
            hashrate=bid.hashrate,
            price_fp18=bid.price_fp18,
        )


def _handle_lost_bids(lost_bids: list[BidResult], result: AuctionResult) -> None:
    """
    Handle workers that lost their bids.

    Future integration:
        - Route these workers to alternative revenue streams
        - Configure for spot market or other subnets
        - Or set to idle to save power
    """
    logger.info(
        "Processing lost bids",
        count=len(lost_bids),
        hashrates=[b.hashrate for b in lost_bids],
    )

    # TODO: Integration point for ASIC routing
    # For each lost bid:
    # 1. Identify the physical ASIC/worker by hashrate
    # 2. Configure alternative routing (spot market, other subnet, idle)
    # 3. Update monitoring

    for bid in lost_bids:
        logger.debug(
            "Lost bid details",
            hashrate=bid.hashrate,
            price_fp18=bid.price_fp18,
        )
