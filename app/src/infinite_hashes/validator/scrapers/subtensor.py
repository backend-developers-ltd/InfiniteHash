"""Subtensor scraper for on-chain dTAO pool data."""

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_alpha_tao_price(bittensor: Any, netuid: int) -> dict:
    """Get ALPHA/TAO price directly from subtensor dTAO pool storage.

    Uses the EXACT formula from subtensor source code (pallets/swap/src/pallet/impls.rs):
        price = tao_reserve / alpha_reserve
    Where:
        tao_reserve = SubnetTAO + SubnetTaoProvided
        alpha_reserve = SubnetAlphaIn + SubnetAlphaInProvided

    This is the authoritative price used by the blockchain for swaps.

    Args:
        bittensor: Bittensor client instance
        netuid: Subnet UID

    Returns:
        {
            "timestamp": datetime.datetime,  # Current time
            "price": float,  # ALPHA/TAO price
        }

    Raises:
        RuntimeError: If pool data cannot be queried or parsed
    """
    subtensor = getattr(bittensor, "subtensor", None)
    if subtensor is None:
        raise RuntimeError("Subtensor not available")

    state = subtensor.state

    # Query the exact storage items used by subtensor's current_price() function
    subnet_tao = await state.getStorage("SubtensorModule.SubnetTAO", netuid)
    subnet_tao_provided = await state.getStorage("SubtensorModule.SubnetTaoProvided", netuid)
    subnet_alpha_in = await state.getStorage("SubtensorModule.SubnetAlphaIn", netuid)
    subnet_alpha_in_provided = await state.getStorage("SubtensorModule.SubnetAlphaInProvided", netuid)

    if any(v is None for v in [subnet_tao, subnet_tao_provided, subnet_alpha_in, subnet_alpha_in_provided]):
        raise RuntimeError(
            f"Failed to query pool data for netuid {netuid}. "
            f"SubnetTAO={subnet_tao}, SubnetTaoProvided={subnet_tao_provided}, "
            f"SubnetAlphaIn={subnet_alpha_in}, SubnetAlphaInProvided={subnet_alpha_in_provided}"
        )

    # Calculate reserves exactly as subtensor does
    tao_reserve = subnet_tao + subnet_tao_provided
    alpha_reserve = subnet_alpha_in + subnet_alpha_in_provided

    if alpha_reserve <= 0:
        raise RuntimeError(
            f"Invalid alpha reserve for netuid {netuid}: {alpha_reserve}. "
            f"SubnetAlphaIn={subnet_alpha_in}, SubnetAlphaInProvided={subnet_alpha_in_provided}"
        )

    # Calculate price using subtensor's formula
    price = float(tao_reserve) / float(alpha_reserve)

    logger.info(
        f"ALPHA/TAO price for netuid {netuid}: tao_reserve={tao_reserve}, alpha_reserve={alpha_reserve}, price={price}"
    )

    return {
        "timestamp": datetime.datetime.now(datetime.UTC),
        "price": price,
    }
