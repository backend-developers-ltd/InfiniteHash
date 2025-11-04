"""Max price filter integration test.

Tests that bids with price_multiplier > MAX_PRICE_MULTIPLIER are filtered out
during auction winner selection.

## APS Miner Constraint: Single Price Per Miner

All workers for a given miner MUST have the same price_multiplier, so we use
two separate miners to test different prices.

Scenario:
- Single validator with 50 PH budget
- Two miners with different prices:
  - miner_0: 20 PH at 1.0 multiplier (should win - within limit)
  - miner_1: 25 PH at 1.06 multiplier (should be filtered - exceeds 1.05 limit)
- Expected: Only miner_0's 20 PH bid wins, remaining 30 PH goes to burn (__owner__)
- Note: Budget is 50 PH, so both bids would fit if price filtering didn't work
  (20 + 25*1.06 = 46.5 < 50), proving the filter is working

Run with:
    pytest app/src/tests/integration/test_max_price_filter.py -x
    pytest -m integration -v
"""

import pytest
import structlog

from infinite_hashes.testutils.integration.scenario import (
    AssertWeightsEvent,
    RegisterMiner,
    RegisterValidator,
    Scenario,
    SetCommitment,
    SetPrices,
    TimeAddress,
    perfect_delivery_hook,
)
from infinite_hashes.testutils.integration.scenario_runner import ScenarioRunner

from .helpers import compute_expected_weights

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=False)
@pytest.mark.integration
@pytest.mark.slow
async def test_max_price_filter(django_db_setup) -> None:
    """Test that bids exceeding MAX_PRICE_MULTIPLIER (1.05) are filtered out.

    Uses two separate miners due to APS miner constraint (single price per miner).

    Verifies that:
    1. Bids with price ≤ 1.05 are accepted (miner_0)
    2. Bids with price > 1.05 are filtered out during auction selection (miner_1)
    3. Unused budget goes to burn (__owner__)
    """
    logger.info("Starting max price filter integration test")

    # Create scenario with perfect delivery (1 epoch is sufficient)
    scenario = Scenario(
        num_epochs=1,
        default_delivery_hook=perfect_delivery_hook,
    )

    # Base time: initialization phase (before epoch 0)
    t0 = TimeAddress(-1, 5, 0)

    # Define scenario timeline
    scenario.add_events(
        # Register single validator
        RegisterValidator(time=t0, name="validator_0", stake=10_000.0),
        # Register miner with acceptable price (1.0 ≤ 1.05)
        RegisterMiner(
            time=t0,
            name="miner_0",
            workers=[
                {"identifier": "worker_acceptable", "hashrate_ph": "20.0", "price_multiplier": "1.0"},
            ],
        ),
        # Register miner with excessive price (1.06 > 1.05)
        # This miner's bid should be filtered out
        RegisterMiner(
            time=t0,
            name="miner_1",
            workers=[
                {"identifier": "worker_expensive", "hashrate_ph": "25.0", "price_multiplier": "1.06"},
            ],
        ),
        # Validator publishes price commitment with 50 PH budget
        # Budget is large enough for both bids if filtering didn't work
        SetPrices(time=t0.b_dt(1), validator_name="validator_0", ph_budget=50),
        # Both miners submit bidding commitments
        SetCommitment(time=t0.b_dt(2), miner_name="miner_0"),
        SetCommitment(time=t0.b_dt(2), miner_name="miner_1"),
    )

    # Compute expected weights for epoch 0
    # Only miner_0's 20 PH bid at 1.0 multiplier should win
    # miner_1's 25 PH bid at 1.06 should be filtered out (not in any window)
    # Remaining budget (50 - 20 = 30 PH * blocks) goes to __owner__ (burn)
    # Note: Budget is large enough (50 PH) that both would fit if filtering didn't work
    #       (20 + 25*1.06 = 46.5 < 50), proving the filter is active
    epoch_0_weights = compute_expected_weights(
        budget_ph=50,
        miner_0=[
            (20, 1.0, [1] * 6),  # Acceptable price - wins all windows
        ],
        # miner_1 is filtered out - no entry (same as [0]*6)
    )

    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(1, 0, 1),  # Check at start of epoch 1
            for_epoch=0,
            expected_weights={
                "validator_0": epoch_0_weights,
            },
        )
    )

    # Log expected weights for debugging
    logger.info(
        "Expected weights (epoch 0)",
        weights=epoch_0_weights,
        miner_0_weight=epoch_0_weights.get("miner_0", 0),
        burn_weight=epoch_0_weights.get("__owner__", 0),
    )

    # Run scenario using the simplest pattern (classmethod)
    # Worker functions use sensible defaults from testutils.worker_mains
    await ScenarioRunner.execute(
        scenario,
        random_seed=12345,
        run_suffix="maxprice",
    )

    logger.info(
        "Max price filter test completed successfully",
        scenario="max_price_filter",
        epochs_processed=1,
    )
