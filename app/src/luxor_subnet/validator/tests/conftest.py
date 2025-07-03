import unittest.mock
from collections.abc import Generator

import httpx
import pytest
from django.conf import settings


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
    mocked.subnet.return_value.tempo = settings.BITTENSOR_SUBNET_TEMPO
    mocked.subnet.return_value.weights.commit = unittest.mock.AsyncMock()

    with unittest.mock.patch(
        "turbobt.Bittensor",
        return_value=mocked,
    ):
        yield mocked


@pytest.fixture
def luxor(httpx_mock):
    httpx_mock.add_response(
        url=httpx.URL(
            f"https://app.luxor.tech/api/v1/pool/workers-hashrate-efficiency/BTC/{settings.LUXOR_SUBACCOUNT_NAME}",
            params={
                "end_date": "2025-07-02",
                "page_number": 1,
                "page_size": 100,
                "start_date": "2025-07-01",
                "tick_size": "1h",
            },
        ),
        match_headers={
            "Authorization": settings.LUXOR_API_KEY,
        },
        json={
            "currency_type": "BTC",
            "start_date": "2025-07-01",
            "end_date": "2025-07-02",
            "tick_size": "1h",
            "hashrate_efficiency_revenue": {
                "5E4DGEqvwagmcL7VxV5Ndj8r7QhradEsFLGqT14qVWhGW551-1": [
                    {
                        "date_time": "2025-07-01T00:00:00.000Z",
                        "hashrate": "85067992961442",
                        "efficiency": 1,
                        "est_revenue": 1.919999931487837e-06,
                    },
                    {
                        "date_time": "2025-07-01T01:00:00.000Z",
                        "hashrate": "95075992133377",
                        "efficiency": 1,
                        "est_revenue": 2.1500000002561137e-06,
                    },
                    {
                        "date_time": "2025-07-01T02:00:00.000Z",
                        "hashrate": "96952491978114",
                        "efficiency": 1,
                        "est_revenue": 2.190000031987438e-06,
                    },
                    {
                        "date_time": "2025-07-01T03:00:00.000Z",
                        "hashrate": "101330991615836",
                        "efficiency": 1,
                        "est_revenue": 2.2900001113157487e-06,
                    },
                    {
                        "date_time": "2025-07-01T04:00:00.000Z",
                        "hashrate": "89446492599164",
                        "efficiency": 1,
                        "est_revenue": 2.0200000108161476e-06,
                    },
                    {
                        "date_time": "2025-07-01T05:00:00.000Z",
                        "hashrate": "90697492495655",
                        "efficiency": 1,
                        "est_revenue": 2.049999920927803e-06,
                    },
                    {
                        "date_time": "2025-07-01T06:00:00.000Z",
                        "hashrate": "87569992754426",
                        "efficiency": 1,
                        "est_revenue": 1.9799999790848233e-06,
                    },
                    {
                        "date_time": "2025-07-01T07:00:00.000Z",
                        "hashrate": "101956491564082",
                        "efficiency": 1,
                        "est_revenue": 2.3099998998077353e-06,
                    },
                    {
                        "date_time": "2025-07-01T08:00:00.000Z",
                        "hashrate": "84442493013196",
                        "efficiency": 1,
                        "est_revenue": 1.9100000372418435e-06,
                    },
                    {
                        "date_time": "2025-07-01T09:00:00.000Z",
                        "hashrate": "91635742418024",
                        "efficiency": 0.9965986609458923,
                        "est_revenue": 2.069999936793465e-06,
                    },
                    {
                        "date_time": "2025-07-01T10:00:00.000Z",
                        "hashrate": "97499804432830",
                        "efficiency": 1,
                        "est_revenue": 2.1999999262334313e-06,
                    },
                    {
                        "date_time": "2025-07-01T11:00:00.000Z",
                        "hashrate": "92886742314516",
                        "efficiency": 1,
                        "est_revenue": 2.100000074278796e-06,
                    },
                    {
                        "date_time": "2025-07-01T12:00:00.000Z",
                        "hashrate": "91635742418024",
                        "efficiency": 1,
                        "est_revenue": 2.069999936793465e-06,
                    },
                    {
                        "date_time": "2025-07-01T13:00:00.000Z",
                        "hashrate": "91948492392147",
                        "efficiency": 1,
                        "est_revenue": 2.080000058413134e-06,
                    },
                    {
                        "date_time": "2025-07-01T14:00:00.000Z",
                        "hashrate": "92886742314516",
                        "efficiency": 0.9933110475540161,
                        "est_revenue": 2.100000074278796e-06,
                    },
                    {
                        "date_time": "2025-07-01T15:00:00.000Z",
                        "hashrate": "107585991098295",
                        "efficiency": 0.9942196607589722,
                        "est_revenue": 2.429999995001708e-06,
                    },
                    {
                        "date_time": "2025-07-01T16:00:00.000Z",
                        "hashrate": "93199492288639",
                        "efficiency": 1,
                        "est_revenue": 2.1099999685247894e-06,
                    },
                    {
                        "date_time": "2025-07-01T17:00:00.000Z",
                        "hashrate": "97577991926360",
                        "efficiency": 1,
                        "est_revenue": 2.2100000478531e-06,
                    },
                    {
                        "date_time": "2025-07-01T18:00:00.000Z",
                        "hashrate": "96952491978114",
                        "efficiency": 1,
                        "est_revenue": 2.190000031987438e-06,
                    },
                    {
                        "date_time": "2025-07-01T19:00:00.000Z",
                        "hashrate": "96014242055745",
                        "efficiency": 1,
                        "est_revenue": 2.170000016121776e-06,
                    },
                    {
                        "date_time": "2025-07-01T20:00:00.000Z",
                        "hashrate": "103832991408819",
                        "efficiency": 1,
                        "est_revenue": 2.3499999315390596e-06,
                    },
                    {
                        "date_time": "2025-07-01T21:00:00.000Z",
                        "hashrate": "96326992029868",
                        "efficiency": 1,
                        "est_revenue": 2.179999910367769e-06,
                    },
                    {
                        "date_time": "2025-07-01T22:00:00.000Z",
                        "hashrate": "81940493220213",
                        "efficiency": 1,
                        "est_revenue": 1.8499999896448571e-06,
                    },
                    {
                        "date_time": "2025-07-01T23:00:00.000Z",
                        "hashrate": "107585991098295",
                        "efficiency": 1,
                        "est_revenue": 2.429999995001708e-06,
                    },
                    {
                        "date_time": "2025-07-02T00:00:00.000Z",
                        "hashrate": "81314993271967",
                        "efficiency": 1,
                        "est_revenue": 1.839999981712026e-06,
                    },
                    {
                        "date_time": "2025-07-02T01:00:00.000Z",
                        "hashrate": "91635742418024",
                        "efficiency": 1,
                        "est_revenue": 2.069999936793465e-06,
                    },
                    {
                        "date_time": "2025-07-02T02:00:00.000Z",
                        "hashrate": "92573992340393",
                        "efficiency": 1,
                        "est_revenue": 2.0899999526591273e-06,
                    },
                    {
                        "date_time": "2025-07-02T03:00:00.000Z",
                        "hashrate": "92886742314516",
                        "efficiency": 1,
                        "est_revenue": 2.100000074278796e-06,
                    },
                    {
                        "date_time": "2025-07-02T04:00:00.000Z",
                        "hashrate": "85693492909688",
                        "efficiency": 1,
                        "est_revenue": 1.939999947353499e-06,
                    },
                    {
                        "date_time": "2025-07-02T05:00:00.000Z",
                        "hashrate": "102581991512327",
                        "efficiency": 1,
                        "est_revenue": 2.320000021427404e-06,
                    },
                    {
                        "date_time": "2025-07-02T06:00:00.000Z",
                        "hashrate": "93824992236885",
                        "efficiency": 1,
                        "est_revenue": 2.1200000901444582e-06,
                    },
                    {
                        "date_time": "2025-07-02T07:00:00.000Z",
                        "hashrate": "90071992547409",
                        "efficiency": 1,
                        "est_revenue": 2.0400000266818097e-06,
                    },
                    {
                        "date_time": "2025-07-02T08:00:00.000Z",
                        "hashrate": "103832991408819",
                        "efficiency": 1,
                        "est_revenue": 2.3499999315390596e-06,
                    },
                    {
                        "date_time": "2025-07-02T09:00:00.000Z",
                        "hashrate": "98828991822852",
                        "efficiency": 1,
                        "est_revenue": 2.2300000637187622e-06,
                    },
                    {
                        "date_time": "2025-07-02T10:00:00.000Z",
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
