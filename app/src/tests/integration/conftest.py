"""Pytest configuration for integration tests."""

import multiprocessing as mp
import os

import pytest

# Configure multiprocessing method once for all integration tests
# Must be done before any tests run
mp.set_start_method("spawn", force=True)

# Set Django settings for integration tests
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker):
    """Ensure migrations are run and share test DB config with worker processes."""
    with django_db_blocker.unblock():
        from django.core.management import call_command
        from django.db import connections

        # Run migrations to create tables
        call_command("migrate", "--run-syncdb", verbosity=0)

        # Share test database configuration with worker processes
        # Worker processes need to connect to the same test database
        db_config = connections["default"].settings_dict
        if "NAME" in db_config:
            # For PostgreSQL, pytest creates test_<dbname>
            os.environ["TEST_DB_NAME"] = db_config["NAME"]
            os.environ["TEST_DB_PATH"] = db_config["NAME"]  # Keep for SQLite compat

            # Also share other DB connection params for worker processes
            if "HOST" in db_config:
                os.environ["TEST_DB_HOST"] = db_config["HOST"]
            if "PORT" in db_config:
                os.environ["TEST_DB_PORT"] = str(db_config["PORT"])
            if "USER" in db_config:
                os.environ["TEST_DB_USER"] = db_config["USER"]
