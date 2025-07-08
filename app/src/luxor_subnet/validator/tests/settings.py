import os

from luxor_subnet.settings import *  # noqa: E402,F403

os.environ["DEBUG_TOOLBAR"] = "False"

PROMETHEUS_EXPORT_MIGRATIONS = False

BITTENSOR_NETUID = 388
BITTENSOR_NETWORK = "local"
BITTENSOR_WALLET_HOTKEY_NAME = "default"
BITTENSOR_WALLET_NAME = "default"

LUXOR_API_KEY = "luxor-api-key"
LUXOR_SUBACCOUNT_NAME = "luxor-subaccount-name"
