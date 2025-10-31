"""Utilities for reading mechanism emission split information from chain state."""

from __future__ import annotations

from typing import Any

MECHANISM_SPLIT_DENOMINATOR = 65535


async def fetch_mechanism_share_fraction(
    bittensor: Any,
    netuid: int,
    mechanism_id: int,
) -> float:
    """Return the emission share allocated to ``mechanism_id`` as a fraction in (0, 1]."""

    subtensor = getattr(bittensor, "subtensor", None)
    if subtensor is None:
        raise RuntimeError("Subtensor not available")
    state = subtensor.state

    split_raw = await state.getStorage("SubtensorModule.MechanismEmissionSplit", netuid)
    if not split_raw:
        raise RuntimeError("MechanismEmissionSplit not available from chain state")

    try:
        split_values = [int(value) for value in split_raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Unrecognized mechanism split payload") from exc

    # No split configured => even distribution across mechanisms. Unfortunately
    # we do not currently have the mechanism count in this call, so we treat the
    # missing split as an error to surface configuration issues quickly.
    if not split_values:
        raise RuntimeError("Mechanism emission split is empty")

    if not (0 <= mechanism_id < len(split_values)):
        raise RuntimeError(f"Mechanism {mechanism_id} out of range for split {split_values}")

    share_raw = int(split_values[mechanism_id])
    if share_raw <= 0:
        raise RuntimeError(f"Mechanism {mechanism_id} has zero emission share")

    return share_raw / MECHANISM_SPLIT_DENOMINATOR
