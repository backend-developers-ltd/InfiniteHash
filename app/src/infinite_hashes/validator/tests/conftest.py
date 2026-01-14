import unittest.mock
from collections.abc import Generator

import httpx
import pytest
import pytest_asyncio
from django.conf import settings

from infinite_hashes.testutils.unit.subtensor import SubtensorSimulator


@pytest.fixture
def some() -> Generator[int, None, None]:
    # setup code
    yield 1
    # teardown code


@pytest.fixture
def bittensor():
    mocked = unittest.mock.MagicMock()
    mocked.__aenter__.return_value = mocked
    mocked.blocks.head = unittest.mock.AsyncMock(
        return_value=unittest.mock.MagicMock(
            number=1000,
        ),
    )
    mocked.subnet.return_value.get = unittest.mock.AsyncMock(
        return_value=mocked.subnet.return_value,
    )
    mocked.subnet.return_value.list_neurons = unittest.mock.AsyncMock(
        return_value=[
            unittest.mock.MagicMock(
                hotkey=f"hotkey_{i}",
                uid=i,
            )
            for i in range(10)
        ],
    )
    mocked.subnet.return_value.tempo = 360
    mocked.subnet.return_value.weights.commit = unittest.mock.AsyncMock()

    with unittest.mock.patch(
        "turbobt.Bittensor",
        return_value=mocked,
    ):
        yield mocked


@pytest_asyncio.fixture(scope="session")
async def sim():
    async with SubtensorSimulator() as s:
        yield s


@pytest.fixture
def luxor(httpx_mock):
    httpx_mock.add_response(
        url=httpx.URL(
            f"https://app.luxor.tech/api/v2/pool/workers-hashrate-efficiency/BTC/{settings.LUXOR_SUBACCOUNT_NAME}",
            params={
                "end_date": "2025-08-25",
                "page_number": 1,
                "page_size": 100,
                "start_date": "2025-08-25",
                "tick_size": "1h",
            },
        ),
        match_headers={
            "Authorization": settings.LUXOR_API_KEY_MECHANISM_0,
        },
        json={
            "currency_type": "BTC",
            "start_date": "2025-08-25",
            "end_date": "2025-08-25",
            "tick_size": "1h",
            "hashrate_efficiency_revenue": {
                "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551-1": [
                    {
                        "date_time": "2025-08-25T00:00:00.000Z",
                        "hashrate": "81314993271967",
                        "efficiency": 1,
                        "est_revenue": 1.839999981712026e-06,
                    },
                    {
                        "date_time": "2025-08-25T01:00:00.000Z",
                        "hashrate": "91635742418024",
                        "efficiency": 1,
                        "est_revenue": 2.069999936793465e-06,
                    },
                    {
                        "date_time": "2025-08-25T02:00:00.000Z",
                        "hashrate": "92573992340393",
                        "efficiency": 1,
                        "est_revenue": 2.0899999526591273e-06,
                    },
                    {
                        "date_time": "2025-08-25T03:00:00.000Z",
                        "hashrate": "92886742314516",
                        "efficiency": 1,
                        "est_revenue": 2.100000074278796e-06,
                    },
                    {
                        "date_time": "2025-08-25T04:00:00.000Z",
                        "hashrate": "85693492909688",
                        "efficiency": 1,
                        "est_revenue": 1.939999947353499e-06,
                    },
                    {
                        "date_time": "2025-08-25T05:00:00.000Z",
                        "hashrate": "102581991512327",
                        "efficiency": 1,
                        "est_revenue": 2.320000021427404e-06,
                    },
                    {
                        "date_time": "2025-08-25T06:00:00.000Z",
                        "hashrate": "93824992236885",
                        "efficiency": 1,
                        "est_revenue": 2.1200000901444582e-06,
                    },
                    {
                        "date_time": "2025-08-25T07:00:00.000Z",
                        "hashrate": "90071992547409",
                        "efficiency": 1,
                        "est_revenue": 2.0400000266818097e-06,
                    },
                    {
                        "date_time": "2025-08-25T08:00:00.000Z",
                        "hashrate": "103832991408819",
                        "efficiency": 1,
                        "est_revenue": 2.3499999315390596e-06,
                    },
                    {
                        "date_time": "2025-08-25T09:00:00.000Z",
                        "hashrate": "98828991822852",
                        "efficiency": 1,
                        "est_revenue": 2.2300000637187622e-06,
                    },
                    {
                        "date_time": "2025-08-25T10:00:00.000Z",
                        "hashrate": "101330991615836",
                        "efficiency": 1,
                        "est_revenue": 2.2900001113157487e-06,
                    },
                ],
            },
            "pagination": {
                "page_number": 1,
                "page_size": 1000,
                "item_count": 1,
                "previous_page_url": None,
                "next_page_url": None,
            },
        },
    )
