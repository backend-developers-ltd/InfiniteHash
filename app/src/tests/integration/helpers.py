from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BURN_KEY = "__owner__"
WINDOW_LENGTHS: tuple[int, ...] = (60, 60, 60, 60, 60, 61)


def compute_expected_weights(
    *,
    budget_ph: float | Sequence[float],
    window_lengths: Sequence[int] = WINDOW_LENGTHS,
    **miner_deliveries: Sequence[tuple[float, float, Sequence[float]]],
) -> dict[str, float]:
    """Compute normalized weights (including burn) for deterministic integration tests.

    Args:
        budget_ph: Either a single per-window PH budget or a sequence per window.
        window_lengths: Blocks per window (defaults to standard tempo layout).
        miner_deliveries: keyword arguments per miner mapping to sequences of tuples:
            (promised_ph, price_multiplier, success_per_window).

    Returns:
        Mapping from miner name (plus ``__owner__``) to normalized weights that sum to 1.0.
    """

    window_lengths = tuple(window_lengths)
    if not window_lengths:
        raise ValueError("window_lengths must not be empty")

    if isinstance(budget_ph, int | float):
        budgets = [float(budget_ph)] * len(window_lengths)
    else:
        budgets = [float(value) for value in budget_ph]
        if len(budgets) != len(window_lengths):
            raise ValueError("budget sequence must match number of windows")

    totals: dict[str, float] = {miner_name: 0.0 for miner_name in miner_deliveries}
    owner_total = 0.0

    for window_idx, (window_len, window_budget) in enumerate(zip(window_lengths, budgets)):
        budget_value = window_budget * window_len
        window_spent = 0.0

        for miner_name, deliveries in miner_deliveries.items():
            miner_window_value = 0.0
            for delivery in deliveries:
                promised_ph, price_multiplier, success_curve = _normalize_delivery_tuple(delivery)
                if len(success_curve) != len(window_lengths):
                    raise ValueError(
                        f"success curve for {miner_name} must match number of windows "
                        f"(expected {len(window_lengths)}, got {len(success_curve)})"
                    )
                success = success_curve[window_idx]
                if success <= 0:
                    continue
                miner_window_value += promised_ph * price_multiplier * success * window_len

            totals[miner_name] = totals.get(miner_name, 0.0) + miner_window_value
            window_spent += miner_window_value

        leftover = budget_value - window_spent
        if leftover > 1e-9:
            owner_total += leftover

    totals[BURN_KEY] = owner_total

    total_value = sum(totals.values())
    if total_value <= 0:
        return {name: 0.0 for name in totals}

    return {name: value / total_value for name, value in totals.items()}


def _normalize_delivery_tuple(
    delivery: tuple[float, float, Sequence[float]] | Sequence[Any],
) -> tuple[float, float, Sequence[float]]:
    """Ensure a delivery tuple has the canonical shape."""
    if not isinstance(delivery, tuple | list) or len(delivery) != 3:
        raise ValueError(f"invalid delivery tuple: {delivery!r}")
    promised_ph = float(delivery[0])
    price_multiplier = float(delivery[1])
    success_curve = delivery[2]
    return promised_ph, price_multiplier, success_curve
