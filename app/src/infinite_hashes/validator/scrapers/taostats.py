import datetime
from typing import Literal

import httpx


class TaoStatsClient:
    def __init__(self, api_key: str):
        self._client = httpx.Client(
            base_url="https://api.taostats.io/api",
            headers={"Authorization": api_key},
            timeout=15,
        )

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._client.__exit__(exc_type, exc_value, tb)

    def price_ohlc(
        self,
        asset: str,
        period: Literal["1d", "1h", "1m"],
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        limit: int = 1,
    ) -> list[dict]:
        r = self._client.get(
            "/price/ohlc/v1",
            params={
                "asset": asset,
                "period": period,
                "timestamp_start": int(start_datetime.timestamp()),
                "timestamp_end": int(end_datetime.timestamp()),
                "page": 1,
                "limit": limit,
            },
        )
        r.raise_for_status()
        return r.json()["data"]

    def dtao_pool_history(
        self,
        netuid: int,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        frequency: Literal["by_block", "by_hour", "by_day"] = "by_block",
        limit: int = 1,
    ) -> list[dict]:
        r = self._client.get(
            "/dtao/pool/history/v1",
            params={
                "netuid": netuid,
                "frequency": frequency,
                "timestamp_start": int(start_datetime.timestamp()),
                "timestamp_end": int(end_datetime.timestamp()),
                "page": 1,
                "limit": limit,
            },
        )
        r.raise_for_status()
        return r.json()["data"]
