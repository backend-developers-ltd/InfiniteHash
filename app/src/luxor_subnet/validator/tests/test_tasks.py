import datetime
import unittest.mock

import pytest
from app.src.luxor_subnet.celery_schedules import ScheduleEveryBittensorEpoch
from app.src.luxor_subnet.validator.tasks import (
    BLOCK_TIME,
    Epoch,
    get_epoch,
    get_hashrates,
    set_weights,
)
from django.conf import settings


def test_get_epoch(bittensor):
    bittensor.head.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.Mock(
            number=1000,
        ),
    )
    bittensor.block.return_value.get = unittest.mock.AsyncMock(
        return_value=unittest.mock.Mock(
            number=719,
            get_timestamp=unittest.mock.AsyncMock(
                return_value=datetime.datetime(2025, 7, 1, 10, 0, 0),
            ),
        ),
    )
    bittensor.subnet.return_value.epoch.return_value = range(719, 1080)

    assert get_epoch(netuid=1) == Epoch(
        block=719,
        timestamp=datetime.datetime(2025, 7, 1, 10, 0, 0),
    )


def test_get_hashrates(luxor):
    epoch = Epoch(
        block=719,
        timestamp=datetime.datetime(2025, 7, 2, 10, 0, 0),
    )
    hashrates = get_hashrates(
        epoch,
        "luxor-subaccount-name",
    )

    assert hashrates == (
        epoch,
        {
            "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551": 95206304622594,
        },
    )


def test_set_weights(bittensor):
    assert set_weights(
        epoch=Epoch(
            block=719,
            timestamp=datetime.datetime(2025, 7, 2, 10, 0, 0),
        ),
        hashrates={
            "hotkey_0": 0,
            "hotkey_1": 1,
            "hotkey_3": 3,
        },
        netuid=388,
    )

    bittensor.subnet.assert_called_once_with(388)
    bittensor.subnet.return_value.weights.commit.assert_awaited_once_with(
        {
            0: 0.0,
            1: 0.25,
            3: 0.75,
        },
    )


@pytest.mark.parametrize(
    "block_number,offset,threshold,is_due,wait",
    [
        (
            1078,
            0,
            0,
            False,
            24.0,
        ),
        (
            1079,
            0,
            0,
            False,
            12.0,
        ),
        (
            1080,
            0,
            0,
            True,
            4332.0,
        ),
        (
            1081,
            0,
            0,
            False,
            4320.0,
        ),
        (
            1079,
            0,
            10,
            False,
            12.0,
        ),
        (
            1080,
            0,
            10,
            True,
            4332.0,
        ),
        (
            1081,
            0,
            10,
            True,
            4320.0,
        ),
        (
            1089,
            0,
            10,
            True,
            4224.0,
        ),
        (
            1090,
            0,
            10,
            False,
            4212.0,
        ),
        (
            1091,
            0,
            10,
            False,
            4200.0,
        ),
        (
            1179,
            100,
            10,
            False,
            12.0,
        ),
        (
            1180,
            100,
            10,
            True,
            4332.0,
        ),
        (
            1181,
            100,
            10,
            True,
            4320.0,
        ),
        (
            1189,
            100,
            10,
            True,
            4224.0,
        ),
        (
            1190,
            100,
            10,
            False,
            4212.0,
        ),
    ],
)
def test_bittensor_schedule(bittensor, block_number, offset, threshold, is_due, wait):
    bittensor.blocks.head.return_value.number = block_number

    schedule = ScheduleEveryBittensorEpoch(
        netuid=1,
        threshold=threshold,
        offset=offset,
    )

    last_run_at = datetime.datetime.now() - BLOCK_TIME * settings.BITTENSOR_SUBNET_TEMPO

    assert schedule.is_due(last_run_at) == (
        is_due,
        wait,
    )
