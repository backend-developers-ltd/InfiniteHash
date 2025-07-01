import os

from luxor_subnet.settings import *  # noqa: E402,F403

os.environ["DEBUG_TOOLBAR"] = "False"

PROMETHEUS_EXPORT_MIGRATIONS = False
