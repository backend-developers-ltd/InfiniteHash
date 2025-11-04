"""
Data models for APScheduler-based miner.

Simple dataclasses for workers and auction results.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Worker:
    """Represents a single worker with hashrate and price."""

    hashrate_compact: str  # Hashrate in compact string format (e.g., "100")
    price_multiplier: str  # Price multiplier (e.g., "1.0")
    is_active: bool = True

    def bid_tuple(self) -> tuple[str, int]:
        """
        Convert to bid tuple format (hashrate, price_fp18).

        Returns:
            Tuple of (hashrate_compact, price_as_fp18_int)
        """
        # Convert price_multiplier to fp18 integer
        # fp18 means fixed-point with 18 decimal places
        multiplier = float(self.price_multiplier)
        price_fp18 = int(multiplier * (10**18))
        return (self.hashrate_compact, price_fp18)


@dataclass
class BidResult:
    """Result for a single bid (won or lost)."""

    hashrate: str
    price_fp18: int
    won: bool


@dataclass
class AuctionResult:
    """Results from an auction computation."""

    epoch_start: int
    start_block: int
    end_block: int
    window_start_time: datetime
    window_end_time: datetime
    commitments_count: int
    all_winners: list[dict[str, Any]]  # All winners from the auction
    my_bids: list[BidResult]  # Our bids and whether they won
