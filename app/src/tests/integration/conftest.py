"""Pytest configuration for integration tests."""

import json
import logging
import multiprocessing as mp
import os
import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Configure multiprocessing method once for all integration tests
# Must be done before any tests run
mp.set_start_method("spawn", force=True)

# Set Django settings for integration tests
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")
os.environ.setdefault("PRICE_COMMITMENT_BUDGET_CAP", "1.0")
os.environ.setdefault("APS_MINER_ALLOW_V1", "1")


class _ProxyWorkersHandler(BaseHTTPRequestHandler):
    """Minimal proxy API stub returning an empty workers list."""

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        state_path = os.environ.get("PROXY_WORKERS_STATE")
        payload_dict = {"workers": []}
        if state_path:
            try:
                with open(state_path, encoding="utf-8") as f:
                    payload_dict = json.load(f)
            except FileNotFoundError:
                payload_dict = {"workers": []}
            except Exception:
                payload_dict = {"workers": []}
        payload = json.dumps(payload_dict).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A003 - matching BaseHTTPRequestHandler signature
        return  # silence test noise


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


@pytest.fixture(scope="session")
def proxy_workers_api():
    """Start a dummy proxy API server that always returns no workers."""
    server = _ThreadedHTTPServer(("127.0.0.1", 0), _ProxyWorkersHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{server.server_port}/api/v1/workers"
    state_fd, state_path = tempfile.mkstemp(prefix="proxy_workers_", suffix=".json")
    os.close(state_fd)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"workers": []}, f)
    os.environ["PROXY_WORKERS_STATE"] = state_path
    os.environ["PROXY_WORKERS_API_URL"] = base_url
    try:
        from django.conf import settings

        settings.PROXY_WORKERS_API_URL = base_url
        settings.PROXY_WORKERS_STATE = state_path
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to set proxy workers settings override", exc_info=exc)

    yield base_url

    server.shutdown()
    try:
        os.remove(state_path)
    except OSError:
        pass
    thread.join()


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker, proxy_workers_api):
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
