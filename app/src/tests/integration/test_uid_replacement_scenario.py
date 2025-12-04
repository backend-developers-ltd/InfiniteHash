"""UID replacement scenario integration test.

Tests the behavior when a new miner registers and takes over an existing UID from
a poorly performing miner (90% delivery).

## APS Miner Constraint: Single Price Per Miner

All workers for a given miner MUST have the same price_multiplier. See test_basic_scenario.py
for details. This test already conforms to this constraint.

Scenario:
- Start with 3 miners (miner_0, miner_1, miner_2) and 2 validators
- miner_2 delivers only 90% hashrate consistently
- In epoch 1, block 1: third validator joins
- In epoch 2: new perfect miner (miner_3_replacement) registers and takes miner_2's UID
- Verify weights correctly reflect the UID replacement

Run with:
    pytest app/src/tests/integration/test_uid_replacement_scenario.py -x
    pytest -m integration -v
"""

import pytest
import structlog

from infinite_hashes.testutils.integration.scenario import (
    AssertWeightsEvent,
    DeliveryParams,
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


def partial_delivery_hook(miner_name: str, time: TimeAddress) -> DeliveryParams:
    """Consistent 90% delivery for testing poor performance."""
    return DeliveryParams(hashrate_multiplier_range=(0.49, 0.49), dropout_rate=0.0)


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=False)
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skip
async def test_uid_replacement_scenario(django_db_setup) -> None:
    """Test UID replacement when a new miner takes over a poorly performing miner's UID.

    Timeline:
    - Epoch -1: Register 2 validators and 3 miners (miner_2 has 90% delivery)
    - Epoch 1, Block 1: Third validator joins
    - Epoch 2, Window 0: New perfect miner registers and takes miner_2's UID
    - Verify weights show the replacement effect
    """
    logger.info("Starting UID replacement scenario test")

    # Create scenario with 3 epochs
    scenario = Scenario(
        num_epochs=3,
        default_delivery_hook=perfect_delivery_hook,
        delivery_hooks={
            "miner_2": partial_delivery_hook,  # miner_2 delivers only 90%
        },
    )

    # Base time: initialization phase (before epoch 0)
    t0 = TimeAddress(-1, 5, 0)

    # Third validator registration time: epoch 1, window 0, block 1
    validator_join_time = TimeAddress(0, 5, 60)

    # UID replacement time: epoch 1, window 5, block 60
    uid_replacement_time = TimeAddress(1, 5, 60)

    # Define scenario timeline
    scenario.add_events(
        # ========== INITIAL REGISTRATION (Epoch -1) ==========
        # Register 2 validators
        RegisterValidator(time=t0, name="validator_0", stake=10_000.0),
        RegisterValidator(time=t0, name="validator_1", stake=10_000.0),
        # Register 3 miners - miner_2 will have poor delivery
        RegisterMiner(
            time=t0,
            name="miner_0",
            workers=[
                {"identifier": "worker_0_a", "hashrate_ph": "10.0", "price_multiplier": "0.98"},
                {"identifier": "worker_0_b", "hashrate_ph": "8.0", "price_multiplier": "0.98"},
            ],
        ),
        RegisterMiner(
            time=t0,
            name="miner_1",
            workers=[
                {"identifier": "worker_1_a", "hashrate_ph": "10.0", "price_multiplier": "1.0"},
                {"identifier": "worker_1_b", "hashrate_ph": "8.0", "price_multiplier": "1.0"},
            ],
        ),
        RegisterMiner(
            time=t0,
            name="miner_2",
            workers=[
                {"identifier": "worker_2_a", "hashrate_ph": "10.0", "price_multiplier": "0.96"},
                {"identifier": "worker_2_b", "hashrate_ph": "8.0", "price_multiplier": "0.96"},
            ],
        ),
        # Initial validators publish price commitments
        SetPrices(time=t0.b_dt(1), validator_name="validator_0", ph_budget=36),
        SetPrices(time=t0.b_dt(1), validator_name="validator_1", ph_budget=36),
        # Initial miners submit bidding commitments
        SetCommitment(time=t0.b_dt(2), miner_name="miner_0"),
        SetCommitment(time=t0.b_dt(2), miner_name="miner_1"),
        SetCommitment(time=t0.b_dt(2), miner_name="miner_2"),
        # ========== EPOCH 1: THIRD VALIDATOR JOINS ==========
        RegisterValidator(
            time=validator_join_time,
            name="validator_2_late",
            stake=5_000.0,
        ),
        # New validator publishes prices (2 blocks later)
        SetPrices(time=validator_join_time.b_dt(2), validator_name="validator_2_late", ph_budget=36),
        # ========== EPOCH 2: UID REPLACEMENT ==========
        # New miner registers and takes over miner_2's UID
        RegisterMiner(
            time=uid_replacement_time,
            name="miner_3_replacement",
            workers=[
                {"identifier": "worker_3_a", "hashrate_ph": "10.0", "price_multiplier": "0.96"},
                {"identifier": "worker_3_b", "hashrate_ph": "8.0", "price_multiplier": "0.96"},
            ],
            replace_miner="miner_2",  # Take over miner_2's UID
        ),
        # New miner submits commitment (3 blocks after registration)
        SetCommitment(time=uid_replacement_time, miner_name="miner_3_replacement"),
    )

    # ========== WEIGHT ASSERTIONS ==========

    # Epoch 0 weights: miner_2 (0.96) wins window 0 but underdelivers → banned and receives zero weight.
    # Window 0: miner_0 + miner_2 win (budget=36, each 18 PH); miner_2 fails, so burn absorbs unused budget.
    # Windows 1-5: Only miner_0 + miner_1 participate (each 18 PH).
    epoch_0_weights = compute_expected_weights(
        budget_ph=36,
        miner_0=[(18, 0.98, [1, 1, 1, 1, 1, 1])],
        miner_1=[(18, 1.0, [0, 1, 1, 1, 1, 1])],
        miner_2=[(18, 0.96, [0, 0, 0, 0, 0, 0])],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(1, 0, 2),  # Check at start of epoch 1
            for_epoch=0,
            expected_weights={
                "validator_0": epoch_0_weights,
                "validator_1": epoch_0_weights,
            },
        )
    )

    # Epoch 1 weights: miner_2 is ALREADY banned from Epoch 0
    # All 6 windows: Only miner_0 + miner_1 participate (both fit in 36 PH)
    epoch_1_weights = compute_expected_weights(
        budget_ph=36,
        miner_0=[(18, 0.98, [1, 1, 1, 1, 1, 1])],
        miner_1=[(18, 1.0, [1, 1, 1, 1, 1, 1])],
        miner_2=[(18, 0.96, [0, 0, 0, 0, 0, 0])],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(2, 0, 2),  # Check at start of epoch 2
            for_epoch=1,
            expected_weights={
                "validator_0": epoch_1_weights,
                "validator_1": epoch_1_weights,
                "validator_2_late": epoch_1_weights,
            },
        )
    )

    # Epoch 2 weights: miner_3_replacement took over miner_2's UID (price 0.96)
    # All windows: miner_3 (0.96) + miner_0 (0.98) win, both deliver perfectly
    # miner_1 (1.0) loses all auctions (worst price, doesn't fit in budget)
    epoch_2_weights = compute_expected_weights(
        budget_ph=36,
        miner_0=[(18, 0.98, [1, 1, 1, 1, 1, 1])],
        miner_1=[(18, 1.0, [0, 0, 0, 0, 0, 0])],
        miner_3_replacement=[(18, 0.96, [1, 1, 1, 1, 1, 1])],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(3, 0, 1),  # Check at start of epoch 3
            for_epoch=2,
            expected_weights={
                "validator_0": epoch_2_weights,
                "validator_1": epoch_2_weights,
                "validator_2_late": epoch_2_weights,
            },
        )
    )

    # Run scenario
    await ScenarioRunner.execute(
        scenario,
        random_seed=54321,
        run_suffix="uid_replace",
    )
