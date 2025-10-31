"""Basic scenario integration test using event-driven miner and validator workers.

Tests the complete auction lifecycle with runtime registration and weight verification.

## APS Miner Constraint: Single Price Per Miner

All workers for a given miner MUST have the same price_multiplier, as the APS miner
uses a single price for all workers (configured in TOML). Different price_multipliers
across workers of the same miner will raise a ValueError.

This constraint is enforced at miner registration (RegisterMiner) and when changing
workers (ChangeWorkers). The price_multiplier format is kept in test scenarios for
potential future extensions, but currently all workers for each miner must share the
same price.

Run with:
    pytest app/src/tests/integration/test_basic_scenario.py -x
    pytest -m integration -v
    pytest -m "integration and not slow" -v
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


# ============================================================================
# Test Scenario
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=False)
@pytest.mark.integration
@pytest.mark.slow
async def test_basic_scenario(django_db_setup) -> None:
    """Basic end-to-end integration test with validators, miners, and auction processing.

    Tests the complete auction lifecycle:
    - Validator and miner registration (initial + runtime)
    - Price commitment publishing
    - Bidding commitment submission
    - Multi-window auction processing
    - Hashrate delivery simulation
    - Weight calculation and verification
    """
    logger.info("Starting basic scenario integration test")

    # Create scenario with perfect delivery for all miners (3 epochs)
    scenario = Scenario(
        num_epochs=3,
        default_delivery_hook=perfect_delivery_hook,
    )

    # Base time: initialization phase (before epoch 0)
    t0 = TimeAddress(-1, 5, 0)

    # Runtime registration time
    registration_block = TimeAddress(1, 2, 15)

    # Define complete scenario timeline
    scenario.add_events(
        # Initial registration - 2 validators, 2 miners
        RegisterValidator(time=t0, name="validator_0", stake=10_000.0),
        RegisterValidator(time=t0, name="validator_1", stake=10_000.0),
        RegisterMiner(
            time=t0,
            name="miner_0",
            workers=[
                {"identifier": "worker_0_a", "hashrate_ph": "10.0", "price_multiplier": "1.0"},
                {"identifier": "worker_0_b", "hashrate_ph": "8.0", "price_multiplier": "1.0"},  # Same price
            ],
        ),
        RegisterMiner(
            time=t0,
            name="miner_1",
            workers=[
                {"identifier": "worker_1_a", "hashrate_ph": "10.0", "price_multiplier": "1.05"},
                {"identifier": "worker_1_b", "hashrate_ph": "8.0", "price_multiplier": "1.05"},  # Same price
            ],
        ),
        # Initial validators publish price commitments
        SetPrices(time=t0.b_dt(1), validator_name="validator_0", ph_budget=50),
        SetPrices(time=t0.b_dt(1), validator_name="validator_1", ph_budget=50),
        # Initial miners submit bidding commitments
        SetCommitment(time=t0.b_dt(2), miner_name="miner_0"),
        SetCommitment(time=t0.b_dt(2), miner_name="miner_1"),
        # RUNTIME REGISTRATION: Epoch 1, Window 2, Block 15
        # Register new validator and miner mid-simulation
        RegisterValidator(
            time=registration_block,
            name="validator_2_late",
            stake=5_000.0,
        ),
        RegisterMiner(
            time=registration_block,
            name="miner_2_late",
            workers=[
                {"identifier": "worker_2_a", "hashrate_ph": "12.0", "price_multiplier": "0.95"},
            ],
        ),
        # New validator publishes prices (1 block later)
        SetPrices(time=registration_block.b_dt(1), validator_name="validator_2_late", ph_budget=50),
        # New miner submits commitment (3 blocks after registration)
        SetCommitment(time=registration_block.b_dt(3), miner_name="miner_2_late"),
        # AssertFalseEvent(time=TimeAddress(2, 0, 10), message="stop")
    )

    # Add weight assertions to verify correct weight calculation
    # Director automatically tracks when validators commit weights for each epoch

    epoch_0_weights = compute_expected_weights(
        budget_ph=50,
        miner_0=[(10, 1.0, [1] * 6), (8, 1.0, [1] * 6)],  # Same price for both workers
        miner_1=[(10, 1.05, [1] * 6), (8, 1.05, [1] * 6)],  # Same price for both workers
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(1, 0, 1),  # Check at start of epoch 1
            for_epoch=0,
            expected_weights={
                "validator_0": epoch_0_weights,
                "validator_1": epoch_0_weights,
            },
        )
    )

    epoch_1_weights = compute_expected_weights(
        budget_ph=50,
        miner_0=[(10, 1.0, [1] * 6), (8, 1.0, [1] * 6)],  # Same price for both workers
        miner_1=[(10, 1.05, [1] * 6), (8, 1.05, [1] * 6)],  # Same price for both workers
        miner_2_late=[(12, 0.95, [0, 0, 0, 1, 1, 1])],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(2, 0, 1),  # Check at start of epoch 2
            for_epoch=1,
            expected_weights={
                "validator_0": epoch_1_weights,
                "validator_1": epoch_1_weights,
                "validator_2_late": epoch_1_weights,
            },
        )
    )

    epoch_2_weights = compute_expected_weights(
        budget_ph=50,
        miner_0=[(10, 1.0, [1] * 6), (8, 1.0, [1] * 6)],  # Same price for both workers
        miner_1=[(10, 1.05, [1] * 6), (8, 1.05, [1] * 6)],  # Same price for both workers
        miner_2_late=[(12, 0.95, [1] * 6)],
    )
    scenario.add_event(
        AssertWeightsEvent(
            time=TimeAddress(3, 0, 0),  # Check 1 block after last window ends
            for_epoch=2,
            expected_weights={
                "validator_0": epoch_2_weights,
                "validator_1": epoch_2_weights,
                "validator_2_late": epoch_2_weights,
            },
        )
    )

    # Run scenario using the simplest pattern (classmethod)
    # Worker functions use sensible defaults from testutils.worker_mains
    await ScenarioRunner.execute(
        scenario,
        random_seed=12345,
        run_suffix="8a7701",
    )
