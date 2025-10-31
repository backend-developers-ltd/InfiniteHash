import os

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip api_integration tests unless RUN_API_INTEGRATION is set.

    This replaces the previous '-m "not api_integration"' default deselection.
    """
    if os.getenv("RUN_API_INTEGRATION") not in {"1", "true", "True", "YES", "yes"}:
        skip_live = pytest.mark.skip(reason="RUN_API_INTEGRATION not set; skipping api_integration tests")
        for item in items:
            if "api_integration" in item.keywords:
                item.add_marker(skip_live)

    if os.getenv("RUN_API_INTEGRATION_LONG") not in {"1", "true", "True", "YES", "yes"}:
        skip_long = pytest.mark.skip(reason="RUN_API_INTEGRATION_LONG not set; skipping api_integration_long tests")
        for item in items:
            if "api_integration_long" in item.keywords:
                item.add_marker(skip_long)
