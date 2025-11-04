import os

from infinite_hashes.settings import *  # noqa: E402,F403

os.environ["DEBUG_TOOLBAR"] = "False"

PROMETHEUS_EXPORT_MIGRATIONS = False

BITTENSOR_NETUID = 388
BITTENSOR_NETWORK = "local"
BITTENSOR_WALLET_HOTKEY_NAME = "default"
BITTENSOR_WALLET_NAME = "default"

# For API integration tests, use separate _FOR_TESTS env vars
# Otherwise use dummy values for unit tests
LUXOR_API_URL = os.getenv("LUXOR_API_URL_FOR_TESTS", "https://app.luxor.tech/api")

# Per-mechanism API keys with fallback to legacy LUXOR_API_KEY_FOR_TESTS
_default_api_key = os.getenv("LUXOR_API_KEY_FOR_TESTS", "luxor-api-key")
LUXOR_API_KEY_MECHANISM_0 = os.getenv("LUXOR_API_KEY_MECHANISM_0_FOR_TESTS", _default_api_key)
LUXOR_API_KEY_MECHANISM_1 = os.getenv("LUXOR_API_KEY_MECHANISM_1_FOR_TESTS", _default_api_key)

LUXOR_SUBACCOUNT_NAME = os.getenv("LUXOR_SUBACCOUNT_NAME_FOR_TESTS", "luxor-subaccount-name")
LUXOR_SUBACCOUNT_NAME_MECHANISM_0 = os.getenv("LUXOR_SUBACCOUNT_NAME_MECHANISM_0_FOR_TESTS", "infinite")
LUXOR_SUBACCOUNT_NAME_MECHANISM_1 = os.getenv("LUXOR_SUBACCOUNT_NAME_MECHANISM_1_FOR_TESTS", "infinite")

# Map subaccount names to API keys (include legacy LUXOR_SUBACCOUNT_NAME for backward compatibility)
LUXOR_API_KEY_BY_SUBACCOUNT = {
    LUXOR_SUBACCOUNT_NAME: _default_api_key,  # legacy
    LUXOR_SUBACCOUNT_NAME_MECHANISM_0: LUXOR_API_KEY_MECHANISM_0,
    LUXOR_SUBACCOUNT_NAME_MECHANISM_1: LUXOR_API_KEY_MECHANISM_1,
}
