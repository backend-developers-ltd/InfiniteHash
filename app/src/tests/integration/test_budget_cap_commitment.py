"""Budget cap commitment integration test.

Runs two scenarios to prove the price commitment cap field scales ILP budget:
- cap=1.0 keeps the full budget
- cap=0.5 halves the budget and reduces winners
"""

import pytest
import structlog
from django.conf import settings as django_settings

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
async def test_budget_cap_commitment_scales_budget(monkeypatch, django_db_setup) -> None:
    """Ensure budget cap from price commitments scales auction budget."""
    base_budget_ph = 10.0
    t0 = TimeAddress(-1, 5, 0)

    async def _run_with_cap(cap_str: str, expected_weights: dict[str, float]) -> None:
        monkeypatch.setenv("PRICE_COMMITMENT_BUDGET_CAP", cap_str)
        monkeypatch.setattr(django_settings, "PRICE_COMMITMENT_BUDGET_CAP", cap_str, raising=False)

        scenario = Scenario(
            num_epochs=1,
            default_delivery_hook=perfect_delivery_hook,
        )

        scenario.add_events(
            RegisterValidator(time=t0, name="validator_0", stake=10_000.0),
            RegisterMiner(
                time=t0,
                name="miner_0",
                workers=[
                    {"identifier": "worker_0", "hashrate_ph": "6.0", "price_multiplier": "1.0"},
                ],
            ),
            RegisterMiner(
                time=t0,
                name="miner_1",
                workers=[
                    {"identifier": "worker_1", "hashrate_ph": "4.0", "price_multiplier": "1.0"},
                ],
            ),
            SetPrices(time=t0.b_dt(1), validator_name="validator_0", ph_budget=base_budget_ph),
            SetCommitment(time=t0.b_dt(2), miner_name="miner_0"),
            SetCommitment(time=t0.b_dt(2), miner_name="miner_1"),
        )

        scenario.add_event(
            AssertWeightsEvent(
                time=TimeAddress(1, 0, 1),
                for_epoch=0,
                expected_weights={"validator_0": expected_weights},
            )
        )

        logger.info("Running budget cap scenario", cap=cap_str)
        await ScenarioRunner.execute(scenario, random_seed=12345)

    expected_full = compute_expected_weights(
        budget_ph=base_budget_ph,
        miner_0=[(6.0, 1.0, [1] * 6)],
        miner_1=[(4.0, 1.0, [1] * 6)],
    )
    await _run_with_cap("1.0", expected_full)

    expected_half = compute_expected_weights(
        budget_ph=base_budget_ph,
        miner_1=[(4.0, 1.0, [1] * 6)],
    )
    await _run_with_cap("0.5", expected_half)
