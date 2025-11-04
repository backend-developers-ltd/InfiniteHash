"""Burn mechanism integration test.

Exercises dynamic commitment and budget changes and verifies burn allocation.

## APS Miner Constraint: Single Price Per Miner

All workers for a given miner MUST have the same price_multiplier. See test_basic_scenario.py
for details. This test already conforms to this constraint (all workers use 1.0).

Scenario highlights:
- Start with two miners bidding 20 PH and 15 PH against a 33 PH budget.
- Mid-epoch validators raise the budget to 42 PH.
- Later, miners rebalance to 18 PH and 26 PH and republish commitments.
- Verify unused capacity flows to ``__owner__`` in both epochs.

Run with:
    pytest app/src/tests/integration/test_burn_mechanism.py -x
    pytest -m integration -v
"""

import pytest
import structlog

from infinite_hashes.testutils.integration.scenario import (
    AssertWeightsEvent,
    ChangeWorkers,
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
async def test_burn_mechanism(django_db_setup) -> None:
    """Test that commitment and budget changes affect burn allocation."""
    logger.info("Starting burn mechanism test")

    # Create scenario covering commitment & budget changes across epoch 0
    scenario = Scenario(
        num_epochs=3,
        default_delivery_hook=perfect_delivery_hook,
    )

    # Base time: initialization phase
    t0 = TimeAddress(-1, 5, 0)

    # Define scenario timeline
    scenario.add_events(
        RegisterValidator(time=t0, name="validator_0", stake=10_000.0),
        RegisterValidator(time=t0, name="validator_1", stake=10_000.0),
        # Register 2 miners with initial commitments (20 PH and 15 PH)
        RegisterMiner(
            time=t0,
            name="miner_0",
            workers=[
                {"identifier": "worker_0_20", "hashrate_ph": "20.0", "price_multiplier": "1.0"},
                {"identifier": "worker_0_15", "hashrate_ph": "15.0", "price_multiplier": "1.0"},
            ],
        ),
        RegisterMiner(
            time=t0,
            name="miner_1",
            workers=[
                {"identifier": "worker_1_20", "hashrate_ph": "20.0", "price_multiplier": "1.0"},
                {"identifier": "worker_1_15", "hashrate_ph": "15.0", "price_multiplier": "1.0"},
            ],
        ),
        # Initial validator budgets (33 PH)
        SetPrices(time=t0.b_dt(1), validator_name="validator_0", ph_budget=33),
        SetPrices(time=t0.b_dt(1), validator_name="validator_1", ph_budget=33),
        # Initial commitments
        SetCommitment(time=t0.b_dt(2), miner_name="miner_0"),
        SetCommitment(time=t0.b_dt(2), miner_name="miner_1"),
        # Mid-window budget increase to 42 PH (effective window 2 onward)
        SetPrices(time=TimeAddress(0, 1, 30), validator_name="validator_0", ph_budget=42),
        SetPrices(time=TimeAddress(0, 1, 30), validator_name="validator_1", ph_budget=42),
        # Mid-window worker change before window 4: miners rebalance to 18/26 PH
        ChangeWorkers(
            time=TimeAddress(0, 3, 30),
            miner_name="miner_0",
            workers=[
                {"identifier": "worker_0_20", "hashrate_ph": "18.0", "price_multiplier": "1.0"},
                {"identifier": "worker_0_15", "hashrate_ph": "26.0", "price_multiplier": "1.0"},
            ],
        ),
        ChangeWorkers(
            time=TimeAddress(0, 3, 30),
            miner_name="miner_1",
            workers=[
                {"identifier": "worker_1_20", "hashrate_ph": "18.0", "price_multiplier": "1.0"},
                {"identifier": "worker_1_15", "hashrate_ph": "26.0", "price_multiplier": "1.0"},
            ],
        ),
        # Publish updated commitments for new worker capacities
        SetCommitment(time=TimeAddress(0, 3, 31), miner_name="miner_0"),
        SetCommitment(time=TimeAddress(0, 3, 31), miner_name="miner_1"),
    )

    # Budgets per window for epoch 0
    epoch0_weights = compute_expected_weights(
        budget_ph=[33, 33, 42, 42, 42, 42],
        miner_0=[
            (20, 1.0, [0, 0, 1, 1, 0, 0]),
            (15, 1.0, [1, 1, 0, 0, 0, 0]),
            (18, 1.0, [0, 0, 0, 0, 1, 1]),
            (26, 1.0, [0, 0, 0, 0, 0, 0]),
        ],
        miner_1=[
            (20, 1.0, [0, 0, 1, 1, 0, 0]),
            (15, 1.0, [1, 1, 0, 0, 0, 0]),
            (18, 1.0, [0, 0, 0, 0, 1, 1]),
            (26, 1.0, [0, 0, 0, 0, 0, 0]),
        ],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(1, 0, 2),  # Check at start of epoch 1
            for_epoch=0,
            expected_weights={
                "validator_0": epoch0_weights,
                "validator_1": epoch0_weights,
            },
        )
    )

    # Epoch 1 runs entirely with updated budgets and worker capacities
    epoch1_weights = compute_expected_weights(
        budget_ph=42,
        miner_0=[(18, 1.0, [1] * 6), (26, 1.0, [0] * 6)],
        miner_1=[(18, 1.0, [1] * 6), (26, 1.0, [0] * 6)],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(2, 0, 2),  # Check at start of epoch 2
            for_epoch=1,
            expected_weights={
                "validator_0": epoch1_weights,
                "validator_1": epoch1_weights,
            },
        )
    )

    # Run scenario
    await ScenarioRunner.execute(
        scenario,
        random_seed=99999,
        run_suffix="burn_test",
    )
