import datetime
from typing import Literal

import httpx


class HashrateIndexClient:
    def __init__(self):
        self._client = httpx.Client(
            base_url="https://data.hashrateindex.com/hi-api/hashrateindex",
            timeout=15,
        )

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._client.__exit__(exc_type, exc_value, tb)

    def hashprice(
        self,
        currency: Literal["USD", "BTC"] = "USD",
        hashunit: Literal["PHS", "THS"] = "PHS",
        bucket: Literal["15s", "5m", "15m", "1H", "2H", "6H", "1D", "7D"] = "1H",
        span: Literal["1D", "7D", "1M", "3M", "1Y", "5Y", "ALL"] = "1M",
    ) -> list[dict]:
        r = self._client.get(
            "/hashprice",
            params={
                "currency": currency,
                "hashunit": hashunit,
                "bucket": bucket,
                "span": span,
            },
        )
        r.raise_for_status()
        data = r.json()["data"]
        # unify timestamps to datetime for caller convenience
        return [{"timestamp": datetime.datetime.fromisoformat(i["timestamp"]), "price": i["price"]} for i in data]
