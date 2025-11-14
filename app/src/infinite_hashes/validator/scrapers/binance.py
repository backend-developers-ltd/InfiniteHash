import datetime

import httpx


class BinanceClient:
    """Client for Binance public API to fetch cryptocurrency prices."""

    def __init__(self):
        self._client = httpx.Client(
            base_url="https://data-api.binance.vision/api/v3",
            timeout=15,
        )

    def __enter__(self):
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._client.__exit__(exc_type, exc_value, tb)

    def avg_price(self, symbol: str) -> dict:
        """Get average price for a trading pair.

        Args:
            symbol: Trading pair symbol (e.g., "TAOUSDC")

        Returns:
            {
                "mins": 5,  # Price averaging window
                "price": "123.45000000",
                "closeTime": 1234567890000  # Unix timestamp in milliseconds
            }
        """
        r = self._client.get(
            "/avgPrice",
            params={"symbol": symbol},
        )
        r.raise_for_status()
        data = r.json()

        # Convert to standard format with datetime
        return {
            "timestamp": datetime.datetime.fromtimestamp(data["closeTime"] / 1000, tz=datetime.UTC),
            "price": float(data["price"]),
        }
