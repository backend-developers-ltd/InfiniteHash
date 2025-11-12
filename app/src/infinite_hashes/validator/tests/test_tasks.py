import datetime
import unittest.mock

import freezegun
import pytest
from django.test import override_settings

from infinite_hashes.validator.models import PriceObservation, WeightsBatch
from infinite_hashes.validator.tasks import (
    calculate_weights,
    scrape_metrics,
    set_weights,
)


@pytest.mark.django_db
@freezegun.freeze_time("2025-08-25 10:00:00")
def test_calculate_weights(bittensor, luxor):
    bittensor.head.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.Mock(
            number=1010,
        ),
    )
    bittensor.block.return_value.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.Mock(
            number=1007,
            get_timestamp=unittest.mock.AsyncMock(
                return_value=datetime.datetime(2025, 8, 25, 10, 0, 0, 1000, tzinfo=datetime.UTC),
            ),
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    assert calculate_weights() == {
        "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551": 98411991857354,
    }

    assert WeightsBatch.objects.count() == 1

    batch = WeightsBatch.objects.get()

    assert batch.epoch_start == 719
    assert batch.block == 1007
    assert batch.weights == {
        "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551": 98411991857354,
    }


@pytest.mark.django_db
def test_set_weights(bittensor):
    bittensor.head.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.MagicMock(
            number=1010,
        ),
    )
    bittensor.block.return_value.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.MagicMock(
            number=1007,
            get_timestamp=unittest.mock.AsyncMock(
                return_value=datetime.datetime(2025, 7, 2, 10, 0, 0),
            ),
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    batch = WeightsBatch.objects.create(
        block=1007,
        epoch_start=719,
        weights={
            "hotkey_0": 0,
            "hotkey_1": 1,
            "hotkey_3": 3,
        },
    )

    assert set_weights()

    bittensor.subnet.assert_called_once_with(388)
    bittensor.subnet.return_value.weights.commit.assert_awaited_once_with(
        {
            0: 0,
            1: 1,
            3: 3,
        },
    )

    batch.refresh_from_db()

    assert batch.scored is True


@pytest.mark.django_db
def test_set_weights_no_batches_to_score(bittensor):
    bittensor.head.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.MagicMock(
            number=1010,
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    WeightsBatch.objects.create(
        block=1007,
        epoch_start=719,
        should_be_scored=False,
        weights={
            "hotkey_0": 0,
            "hotkey_1": 1,
            "hotkey_3": 3,
        },
    )

    assert not set_weights()

    bittensor.subnet.return_value.weights.commit.assert_not_awaited()


@pytest.mark.django_db
def test_set_weights_expired_batches(bittensor):
    bittensor.head.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.MagicMock(
            number=1010,
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    batch = WeightsBatch.objects.create(
        block=648,
        epoch_start=360,
        should_be_scored=True,
        weights={
            "hotkey_0": 0,
            "hotkey_1": 1,
            "hotkey_3": 3,
        },
    )

    assert not set_weights()

    bittensor.subnet.return_value.weights.commit.assert_not_awaited()

    batch.refresh_from_db()

    assert batch.should_be_scored is False


@pytest.mark.django_db
@pytest.mark.api_integration
@override_settings(BITTENSOR_NETWORK="wss://entrypoint-finney.opentensor.ai:443")
def test_scrape_metrics_real_apis():
    """Test scrape_metrics against real external APIs.

    This test requires:
    - Network access to Binance API
    - Network access to Finney network (for on-chain ALPHA/TAO price)
    - Network access to Hashrate Index API

    Run with: RUN_API_INTEGRATION=1 pytest -k test_scrape_metrics_real_apis
    """
    # Clear any existing observations
    PriceObservation.objects.all().delete()

    # Run the scraper
    scrape_metrics()

    # Verify observations were created
    observations = PriceObservation.objects.all()
    assert observations.count() > 0, "Expected price observations to be created"

    # Check that we have observations for each metric
    expected_metrics = {"TAO_USDC", "ALPHA_TAO", "HASHP_USDC"}  # ALL are required

    # Log what we got for debugging
    missing_metrics = []
    for metric in expected_metrics:
        count = observations.filter(metric=metric).count()
        if count > 0:
            latest = observations.filter(metric=metric).order_by("-observed_at").first()
            print(f"{metric}: {count} observations, latest at {latest.observed_at} with price {latest.price_fp18}")
        else:
            print(f"{metric}: No observations (scraping FAILED)")
            missing_metrics.append(metric)

    # ALL metrics must have been scraped successfully - ALPHA_TAO is CRITICAL
    assert len(missing_metrics) == 0, (
        f"CRITICAL: Failed to scrape required metrics: {missing_metrics}. "
        f"All metrics (TAO_USDC, ALPHA_TAO, HASHP_USDC) are required for auction mechanism. "
        f"Check logs for errors."
    )

    # Verify observations have required fields
    for obs in observations:
        assert obs.metric in expected_metrics, f"Unexpected metric: {obs.metric}"
        assert obs.source in {"binance", "subtensor", "hashrateindex"}, f"Unexpected source: {obs.source}"
        assert obs.price_fp18 > 0, f"Invalid price for {obs.metric}: {obs.price_fp18}"
        assert obs.observed_at is not None, f"Missing observed_at for {obs.metric}"
        assert obs.observed_at <= datetime.datetime.now(datetime.UTC), f"Future timestamp for {obs.metric}"
