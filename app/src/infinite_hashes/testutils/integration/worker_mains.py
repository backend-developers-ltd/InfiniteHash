"""Default worker process entry points for scenario testing.

These functions serve as the main entry points for worker processes:
- _simulator_process: Runs the blockchain simulator (HTTP + RPC servers)
- validator_worker_main: Initializes and runs a validator worker
- aps_miner_worker_main: Initializes and runs an APS miner worker
"""

import asyncio
import logging
import multiprocessing as mp
import os
import sys
from contextlib import suppress
from typing import Any

# Silence noisy debug logs from websockets, httpx, httpcore BEFORE any imports
# Set VERBOSE_LOGS=1 to enable full debug output
if not os.environ.get("VERBOSE_LOGS"):
    os.environ.pop("WEBSOCKETS_LOG", None)
    for _logger_name in [
        "websockets",
        "websockets.client",
        "websockets.server",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "hpack",
        "h11",
    ]:
        logging.getLogger(_logger_name).setLevel(logging.WARNING)


_DJANGO_READY = False


def _configure_django() -> None:
    global _DJANGO_READY
    if _DJANGO_READY:
        return

    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        raise RuntimeError("DJANGO_SETTINGS_MODULE must be set before configuring Django")
    os.environ.setdefault("BITTENSOR_NETWORK", "ws://127.0.0.1:9944")
    os.environ.setdefault("BITTENSOR_NETUID", "1")

    import django

    django.setup()
    _DJANGO_READY = True


# ============================================================================
# Simulator Process
# ============================================================================


def simulator_process(ready_event: mp.Event, stop_event: mp.Event) -> None:
    """Default simulator process entry point."""
    import logging

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")
    log_file = open("/tmp/simulator_debug.log", "w", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s", stream=log_file, force=True)

    # Silence noisy websocket/http debug logs (unless VERBOSE_LOGS=1)
    if not os.environ.get("VERBOSE_LOGS"):
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("websockets.client").setLevel(logging.WARNING)
        logging.getLogger("websockets.server").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("hpack").setLevel(logging.WARNING)
        logging.getLogger("h11").setLevel(logging.WARNING)

    _configure_django()

    # Bind process identifier to all logs from simulator
    import structlog as struct_log

    struct_log.contextvars.clear_contextvars()
    struct_log.contextvars.bind_contextvars(worker="simulator")

    from infinite_hashes.testutils.simulator.server import run_http_server, run_rpc_server
    from infinite_hashes.testutils.simulator.state import SimulatorState

    async def run() -> None:
        state = SimulatorState()
        shutdown = asyncio.Event()

        async def wait_for_stop() -> None:
            await asyncio.to_thread(stop_event.wait)
            shutdown.set()

        SIM_HOST = os.environ.get("SIM_HOST", "127.0.0.1")
        SIM_HTTP_PORT = int(os.environ.get("SIM_HTTP_PORT", 8090))
        SIM_RPC_PORT = int(os.environ.get("SIM_RPC_PORT", 9944))

        http_task = asyncio.create_task(run_http_server(SIM_HOST, SIM_HTTP_PORT, state, shutdown))
        rpc_task = asyncio.create_task(run_rpc_server(SIM_HOST, SIM_RPC_PORT, state, shutdown))
        watcher_task = asyncio.create_task(wait_for_stop())

        await asyncio.sleep(1)
        ready_event.set()

        await watcher_task
        with suppress(asyncio.TimeoutError, OSError):
            await asyncio.gather(http_task, rpc_task, return_exceptions=True)

    asyncio.run(run())


# ============================================================================
# Validator Worker
# ============================================================================


def validator_worker_main(worker_id: int, command_queue: Any, response_queue: Any, config: dict[str, Any]) -> None:
    """Default validator worker entry point."""
    try:
        # Install mock Drand BEFORE any imports (including Django setup)
        # This ensures bittensor_drand is mocked before turbobt imports it
        from infinite_hashes.testutils.integration.mock_drand import install_mock as install_mock_drand

        install_mock_drand()

        os.environ["TEST_DB_PATH"] = config["db_path"]
        os.environ["BITTENSOR_WALLET_DIRECTORY"] = config["wallet_dir"]
        os.environ["BITTENSOR_WALLET_NAME"] = config["wallet_name"]
        os.environ["BITTENSOR_WALLET_HOTKEY_NAME"] = config["wallet_hotkey"]
        os.environ["VALIDATOR_DB_SCHEMA"] = config.get("db_schema", "")

        os.environ["DJANGO_SETTINGS_MODULE"] = "infinite_hashes.settings"
        _configure_django()

        # Override database settings to use pytest test database (if running in tests)
        if "TEST_DB_NAME" in os.environ:
            from django.conf import settings as django_settings
            from django.db import connections

            test_db_config = {"NAME": os.environ["TEST_DB_NAME"]}
            if "TEST_DB_HOST" in os.environ:
                test_db_config["HOST"] = os.environ["TEST_DB_HOST"]
            if "TEST_DB_PORT" in os.environ:
                test_db_config["PORT"] = os.environ["TEST_DB_PORT"]
            if "TEST_DB_USER" in os.environ:
                test_db_config["USER"] = os.environ["TEST_DB_USER"]

            # Update settings
            django_settings.DATABASES["default"].update(test_db_config)
            # Close and reconnect with new settings
            connections["default"].close()
            connections.databases["default"] = django_settings.DATABASES["default"]

        from .scenario_runner import _activate_db_schema, _assert_schema_empty, _count_table_rows, _reset_schema

        _activate_db_schema(config.get("db_schema", ""), "VALIDATOR_DB_SCHEMA")

        # Bind worker_id to all logs from this process
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(worker=f"validator_{worker_id}")

        from django.conf import settings as django_settings
        from django.core.management import call_command
        from django.db import connection

        django_settings.AUCTION_WINDOW_BLOCKS = config["auction_window_blocks"]
        django_settings.LUXOR_API_URL = config["luxor_url"]
        django_settings.LUXOR_SUBACCOUNT_NAME_MECHANISM_1 = config.get(
            "luxor_subaccount", django_settings.LUXOR_SUBACCOUNT_NAME_MECHANISM_1
        )
        django_settings.BITTENSOR_WALLET_DIRECTORY = config["wallet_dir"]
        django_settings.BITTENSOR_WALLET_NAME = config["wallet_name"]
        django_settings.BITTENSOR_WALLET_HOTKEY_NAME = config["wallet_hotkey"]
        django_settings.CELERY_TASK_ALWAYS_EAGER = True
        django_settings.CELERY_TASK_EAGER_PROPAGATES = True

        # Fix duplicate logging caused by log propagation in Django LOGGING config
        # Disable propagation for celery.task loggers to prevent logs from
        # bubbling up to celery and root loggers (each with their own handlers)
        import logging

        logging.getLogger("celery.task").propagate = False

        # Log wallet configuration for debugging
        import bittensor_wallet

        test_wallet = bittensor_wallet.Wallet(
            name=config["wallet_name"],
            hotkey=config["wallet_hotkey"],
            path=config["wallet_dir"],
        )
        import structlog

        structlog.get_logger(__name__).info(
            "Validator wallet configured",
            worker_id=worker_id,
            wallet_name=config["wallet_name"],
            wallet_hotkey=config["wallet_hotkey"],
            wallet_dir=config["wallet_dir"],
            actual_hotkey=test_wallet.hotkey.ss58_address,
        )

        _reset_schema(config.get("db_schema", ""))
        _assert_schema_empty(config.get("db_schema", ""))

        connection.close()
        call_command("migrate", "--run-syncdb", verbosity=0)

        if config.get("db_schema"):
            count = _count_table_rows(config["db_schema"], "validator_auctionresult")
            if count:
                raise RuntimeError(
                    f"Validator schema {config['db_schema']} not empty after migrations (found {count} auction results)"
                )

        response_queue.put({"type": "READY", "worker_id": worker_id, "success": True})

        from .validator_worker import run_validator_event_loop

        run_validator_event_loop(
            command_queue,
            response_queue,
            worker_id=worker_id,
            context={"db_schema": config.get("db_schema", "")},
        )
    except Exception as exc:  # noqa: BLE001
        response_queue.put({"type": "READY", "worker_id": worker_id, "success": False, "error": str(exc)})
        raise


# ============================================================================
# APS Miner Worker (APScheduler-based, no Django)
# ============================================================================


def aps_miner_worker_main(worker_id: int, command_queue: Any, response_queue: Any, config: dict[str, Any]) -> None:
    """APS miner worker entry point (APScheduler-based, stateless).

    Unlike the Django miner, this doesn't need database setup.
    Configuration is loaded from TOML file on each task execution.
    """
    try:
        # Install mock Drand BEFORE any imports (needed by turbobt)
        from infinite_hashes.testutils.integration.mock_drand import install_mock as install_mock_drand

        install_mock_drand()

        response_queue.put({"type": "READY", "worker_id": worker_id, "success": True})

        from .aps_miner_worker import run_aps_miner_event_loop

        run_aps_miner_event_loop(
            command_queue,
            response_queue,
            worker_id=worker_id,
            context={
                "config_path": config.get("config_path"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        response_queue.put({"type": "READY", "worker_id": worker_id, "success": False, "error": str(exc)})
