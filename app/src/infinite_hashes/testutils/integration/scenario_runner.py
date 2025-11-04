"""High-level test runner for scenario-based integration testing.

Provides ScenarioRunner class that handles all infrastructure setup/teardown:
- Wallet directory management
- Simulator process lifecycle
- HTTP client setup
- Director orchestration
- Cleanup on success or failure
"""

import datetime as dt
import multiprocessing as mp
import os
import random
import shlex
import tempfile
import uuid
from collections.abc import Callable

import httpx
import structlog
from django.conf import settings

from .director import Director
from .scenario import Scenario
from .worker_mains import aps_miner_worker_main as default_miner_worker_main
from .worker_mains import simulator_process as default_simulator_process
from .worker_mains import validator_worker_main as default_validator_worker_main

logger = structlog.get_logger(__name__)


# ============================================================================
# Database Schema Helpers
# ============================================================================


def _reset_schema(schema_name: str) -> None:
    if not schema_name:
        return

    from django.db import DEFAULT_DB_ALIAS, connections

    conn = connections[DEFAULT_DB_ALIAS]
    if conn.vendor != "postgresql":
        raise RuntimeError("Expected PostgreSQL database for schema management")

    quoted = conn.ops.quote_name(schema_name)
    with conn.cursor() as cursor:
        cursor.execute(f"DROP SCHEMA IF EXISTS {quoted} CASCADE;")
        cursor.execute(f"CREATE SCHEMA {quoted};")


def _assert_schema_empty(schema_name: str) -> None:
    if not schema_name:
        return

    from django.db import DEFAULT_DB_ALIAS, connections

    conn = connections[DEFAULT_DB_ALIAS]
    with conn.cursor() as cursor:
        if conn.vendor == "postgresql":
            cursor.execute(
                """
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = %s
                ORDER BY tablename
                """,
                [schema_name],
            )
            tables = [row[0] for row in cursor.fetchall()]
        else:
            tables = conn.introspection.table_names()

    if tables:
        raise RuntimeError(f"Schema '{schema_name}' is not empty prior to migrations; found tables: {tables}")


def _count_table_rows(schema_name: str, table_name: str) -> int:
    if not schema_name:
        return 0

    from django.db import DEFAULT_DB_ALIAS, connections

    conn = connections[DEFAULT_DB_ALIAS]
    if conn.vendor != "postgresql":
        with conn.cursor() as cursor:
            sql = f"SELECT COUNT(*) FROM {conn.ops.quote_name(table_name)}"  # noqa: S608
            cursor.execute(sql)
            return int(cursor.fetchone()[0])

    from django.db import ProgrammingError

    with conn.cursor() as cursor:
        try:
            sql = f"SELECT COUNT(*) FROM {conn.ops.quote_name(schema_name)}.{conn.ops.quote_name(table_name)}"  # noqa: S608
            cursor.execute(sql)
        except ProgrammingError:
            conn.rollback()
            return 0
        return int(cursor.fetchone()[0])


def _activate_db_schema(schema_name: str, settings_attr: str) -> None:
    if not schema_name:
        return

    from django.conf import settings as django_settings
    from django.db import DEFAULT_DB_ALIAS, connections

    if DEFAULT_DB_ALIAS not in django_settings.DATABASES:
        return

    db_conf = dict(django_settings.DATABASES[DEFAULT_DB_ALIAS])
    options = dict(db_conf.get("OPTIONS", {}))
    existing_tokens = shlex.split(options.get("options", ""))
    filtered_tokens = [tok for tok in existing_tokens if "search_path=" not in tok]
    filtered_tokens.append(f"-c search_path={schema_name}")
    options["options"] = " ".join(filtered_tokens).strip()
    db_conf["OPTIONS"] = options

    django_settings.DATABASES[DEFAULT_DB_ALIAS] = db_conf
    setattr(django_settings, settings_attr, schema_name)

    connections.databases[DEFAULT_DB_ALIAS] = db_conf
    connections[DEFAULT_DB_ALIAS].close()


# ============================================================================
# Scenario Runner
# ============================================================================


class ScenarioRunner:
    """Test runner that handles all technical setup for scenario execution.

    Usage:
        # Pattern 1: Classmethod (simplest) - uses default worker implementations
        await ScenarioRunner.execute(scenario, random_seed=12345)

        # Pattern 2: Context manager
        async with ScenarioRunner(scenario, ...) as runner:
            await runner.run()

        # Pattern 3: Manual control
        runner = ScenarioRunner(scenario, ...)
        await runner.run_and_verify()  # setup, run, cleanup

    Worker functions have sensible defaults and rarely need to be overridden.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        initial_timestamp: dt.datetime | None = None,
        run_suffix: str | None = None,
        random_seed: int | None = None,
        simulator_process_target: Callable | None = None,
        validator_worker_main: Callable | None = None,
        miner_worker_main: Callable | None = None,
    ) -> None:
        self.scenario = scenario
        self.initial_timestamp = initial_timestamp or dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
        self.run_suffix = run_suffix or uuid.uuid4().hex[:6]

        # Worker main functions (use defaults if not provided)
        self.validator_worker_main = validator_worker_main or default_validator_worker_main
        self.miner_worker_main = miner_worker_main or default_miner_worker_main
        self.simulator_process_target = simulator_process_target or default_simulator_process

        if random_seed is not None:
            random.seed(random_seed)

        # Infrastructure state
        self.wallet_dir: str = ""
        self.simulator_process: mp.Process | None = None
        self.simulator_ready: mp.Event | None = None
        self.simulator_stop: mp.Event | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.director: Director | None = None
        self._cleanup_wallet_dir = False

    async def __aenter__(self) -> "ScenarioRunner":
        """Async context manager entry - sets up infrastructure."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - tears down infrastructure."""
        await self.cleanup()

    async def setup(self) -> None:
        """Set up test infrastructure (simulator, wallets, HTTP client)."""
        # Ensure Django is configured in main process (needed for model imports)
        if not os.environ.get("DJANGO_SETTINGS_MODULE"):
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")

        # Configure Django before any model imports (run in thread to avoid async issues)
        import django
        from django.apps import apps

        if not apps.ready:
            os.environ.setdefault("BITTENSOR_NETWORK", "ws://127.0.0.1:9944")
            os.environ.setdefault("BITTENSOR_NETUID", "1")

            import asyncio

            await asyncio.to_thread(django.setup)

        # Setup wallet directory
        if not self.scenario.wallet_dir:
            self.wallet_dir = os.path.join(tempfile.gettempdir(), f"scenario_wallets_{self.run_suffix}")
            self._cleanup_wallet_dir = True
        else:
            self.wallet_dir = self.scenario.wallet_dir
            self._cleanup_wallet_dir = False

        import shutil

        if os.path.exists(self.wallet_dir):
            shutil.rmtree(self.wallet_dir)
        os.makedirs(self.wallet_dir, exist_ok=True)

        # Update scenario with wallet dir
        self.scenario.wallet_dir = self.wallet_dir

        # Configure Django settings
        settings.AUCTION_WINDOW_BLOCKS = 60
        settings.LUXOR_API_URL = "http://localhost:9999"  # Dummy value

        # Start simulator process (if target provided)
        if self.simulator_process_target:
            self.simulator_ready = mp.Event()
            self.simulator_stop = mp.Event()
            self.simulator_process = mp.Process(
                target=self.simulator_process_target,
                args=(self.simulator_ready, self.simulator_stop),
                daemon=True,
            )
            self.simulator_process.start()

            if not self.simulator_ready.wait(timeout=10):
                self.simulator_stop.set()
                self.simulator_process.terminate()
                self.simulator_process.join()
                raise RuntimeError("simulator failed to start")

            logger.info("Simulator started", run_suffix=self.run_suffix)

        # Create HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Create director
        self.director = Director(
            self.scenario,
            self.http_client,
            initial_timestamp=self.initial_timestamp,
            run_suffix=self.run_suffix,
            validator_worker_main=self.validator_worker_main,
            miner_worker_main=self.miner_worker_main,
        )

    async def cleanup(self) -> None:
        """Clean up test infrastructure."""
        # Shutdown director workers
        if self.director:
            self.director.shutdown()

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()

        # Stop simulator
        if self.simulator_stop:
            self.simulator_stop.set()
        if self.simulator_process:
            self.simulator_process.join(timeout=5)
            if self.simulator_process.is_alive():
                self.simulator_process.terminate()
                self.simulator_process.join()

        # Clean up wallet directory if we created it
        if self._cleanup_wallet_dir and self.wallet_dir and os.path.exists(self.wallet_dir):
            import shutil

            shutil.rmtree(self.wallet_dir)

        logger.info("Cleanup complete", run_suffix=self.run_suffix)

    async def run(self) -> None:
        """Execute the scenario."""
        if not self.director:
            raise RuntimeError("runner not initialized - call setup() or use async context manager")
        await self.director.run()

    async def run_and_verify(self) -> None:
        """Convenience method: setup, run, cleanup."""
        try:
            await self.setup()
            await self.run()
        finally:
            await self.cleanup()

    @classmethod
    async def execute(
        cls,
        scenario: Scenario,
        *,
        initial_timestamp: dt.datetime | None = None,
        run_suffix: str | None = None,
        random_seed: int | None = None,
        simulator_process_target: Callable | None = None,
        validator_worker_main: Callable | None = None,
        miner_worker_main: Callable | None = None,
    ) -> None:
        """Execute a scenario with automatic setup and cleanup (classmethod).

        This is the simplest way to run a scenario:
            await ScenarioRunner.execute(scenario, random_seed=12345)

        Worker functions use sensible defaults and rarely need to be specified.
        """
        async with cls(
            scenario,
            initial_timestamp=initial_timestamp,
            run_suffix=run_suffix,
            random_seed=random_seed,
            simulator_process_target=simulator_process_target,
            validator_worker_main=validator_worker_main,
            miner_worker_main=miner_worker_main,
        ) as runner:
            await runner.run()
