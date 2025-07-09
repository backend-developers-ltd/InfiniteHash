import datetime
import unittest.mock

import pytest

from luxor_subnet.validator.models import WeightsBatch
from luxor_subnet.validator.tasks import (
    calculate_weights,
    set_weights,
)


@pytest.mark.django_db
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
                return_value=datetime.datetime(2025, 7, 2, 10, 0, 0),
            ),
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    assert calculate_weights() == {
        "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551": 95298044615003,
    }

    assert WeightsBatch.objects.count() == 1

    batch = WeightsBatch.objects.get()

    assert batch.epoch_start == 719
    assert batch.block == 1007
    assert batch.weights == {
        "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551": 95298044615003,
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
