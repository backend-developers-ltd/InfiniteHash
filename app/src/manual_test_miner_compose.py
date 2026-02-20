#!/usr/bin/env python3
"""
Manual integration test: APS miner + IHP proxy stack via Docker Compose.

This script spins up the simulated blockchain, provisions a temporary working
directory with Docker Compose assets, and runs the full miner stack (miner
container + IHP proxy + reloader sidecar) to verify that target hashrate
routing is updated and reloaded automatically when auction outcomes change.

Prerequisites:
    - Docker with Compose plugin (`docker compose`) or standalone `docker-compose`
    - Ability to pull the miner image referenced in envs/miner-ihp/docker-compose.yml

Usage:
    python manual_test_miner_compose.py
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import bittensor_wallet
import httpx
import structlog
import tomlkit

from infinite_hashes.testutils.integration.budget_helper import (
    DEFAULT_MECHANISM_1_SHARE,
    DEFAULT_MECHANISM_SPLIT,
    alpha_tao_to_fp18,
    compute_alpha_tao_for_budget,
)
from infinite_hashes.testutils.simulator.state import MECHANISM_SPLIT_VALUE_MAX

COMPOSE_SOURCE = Path(__file__).resolve().parents[2] / "envs" / "miner-ihp" / "docker-compose.yml"
MINER_INSTALLER_SOURCE = Path(__file__).resolve().parents[2] / "installer" / "miner_install.sh"

DEFAULT_BACKUP_POOL_HOST = "btc.global.luxor.tech"
DEFAULT_BACKUP_POOL_PORT = 700

LOGGER = structlog.get_logger(__name__)

SIM_HOST = "127.0.0.1"
SIM_HTTP_PORT = 8090
SIM_RPC_PORT = 9944
NETUID = 1
MECHANISM_ID = int(os.environ.get("SIM_MECHANISM_ID", "1"))

BLOCK_TIME = 12
BLOCKS_PER_WINDOW = 60

RNG = secrets.SystemRandom()


def _parse_mechanism_split_override() -> list[int] | None:
    override = os.environ.get("SIM_MECHANISM_SPLIT")
    if not override:
        return None
    parts = [part.strip() for part in override.split(",") if part.strip()]
    if not parts:
        return None
    split: list[int] = []
    for part in parts:
        try:
            value = int(part, 0)
        except ValueError:
            LOGGER.warning("Invalid SIM_MECHANISM_SPLIT entry, ignoring override", entry=part)
            return None
        split.append(max(0, value))
    return split


def mechanism_share_from_split(split: Sequence[int] | None, mechanism_id: int = MECHANISM_ID) -> float:
    if not split or mechanism_id >= len(split):
        return DEFAULT_MECHANISM_1_SHARE
    denominator = MECHANISM_SPLIT_VALUE_MAX or 1
    return max(0.0, min(1.0, split[mechanism_id] / denominator))


def _mechanism_split_payload() -> list[int]:
    override = _parse_mechanism_split_override()
    split = list(override) if override else list(DEFAULT_MECHANISM_SPLIT)
    if len(split) <= MECHANISM_ID:
        split.extend([0] * (MECHANISM_ID + 1 - len(split)))
    return split


def random_hashrates(min_total: float = 30.0, min_workers: int = 4, max_workers: int = 6) -> list[str]:
    for _ in range(100):
        worker_count = RNG.randint(min_workers, max_workers)
        rates = [round(RNG.uniform(4.0, 10.0), 1) for _ in range(worker_count)]
        if sum(rates) >= min_total:
            return [f"{rate:.1f}" for rate in rates]
    raise RuntimeError("Unable to generate worker hashrates meeting minimum total PH")


def update_miner_config_hashrates(config_path: Path, hashrates: Sequence[str]) -> None:
    doc = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    workers = doc.get("workers")
    if not isinstance(workers, dict):
        workers = {}
        doc["workers"] = workers
    workers["hashrates"] = list(hashrates)
    config_path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def simulator_process_main(ready_event: mp.Event, stop_event: mp.Event):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")
    from infinite_hashes.testutils.integration.worker_mains import simulator_process

    simulator_process(ready_event, stop_event)


def create_test_wallet(wallet_dir: str, name: str = "test_miner") -> tuple[str, str]:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        wallet = bittensor_wallet.Wallet(name=name, hotkey="default", path=wallet_dir)
        wallet.create_new_coldkey(n_words=12, use_password=False, overwrite=True)
        wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True)

    wallet_reloaded = bittensor_wallet.Wallet(name=name, hotkey="default", path=wallet_dir)
    hotkey = wallet_reloaded.hotkey.ss58_address
    coldkey = wallet_reloaded.coldkey.ss58_address
    return hotkey, coldkey


async def setup_blockchain(client: httpx.AsyncClient, miner_hotkey: str, validator_hotkey: str) -> list[int]:
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}",
        json={"tempo": 360},
    )
    response.raise_for_status()

    start_block = 1075
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/head",
        json={"number": start_block, "timestamp": "2025-01-15T10:00:00Z"},
    )
    response.raise_for_status()

    neurons = [
        {"hotkey": validator_hotkey, "uid": 0, "stake": 10000.0, "coldkey": validator_hotkey},
        {"hotkey": miner_hotkey, "uid": 1, "stake": 1000.0, "coldkey": miner_hotkey},
    ]
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}/neurons",
        json={"neurons": neurons},
    )
    response.raise_for_status()

    split = _mechanism_split_payload()
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}/mechanism-split",
        json={"split": split},
    )
    response.raise_for_status()

    LOGGER.info(
        "Blockchain initialized",
        start_block=start_block,
        next_window=start_block + (BLOCKS_PER_WINDOW - (start_block % BLOCKS_PER_WINDOW)),
        miners_registered=1,
        validator_hotkey=validator_hotkey[:16],
        miner_hotkey=miner_hotkey[:16],
        mechanism_split=split,
    )
    return split


async def submit_validator_commitment(
    wallet: bittensor_wallet.Wallet,
    budget_ph: float,
    mechanism_share: float | None = None,
) -> None:
    import turbobt

    from infinite_hashes.consensus.price import PriceCommitment

    TAO_USDC = 45.0
    HASHP_USDC = 50.0
    share = mechanism_share if mechanism_share is not None else DEFAULT_MECHANISM_1_SHARE
    alpha_tao = compute_alpha_tao_for_budget(
        budget_ph,
        tao_usdc=TAO_USDC,
        hashp_usdc=HASHP_USDC,
        mechanism_share=share,
    )
    alpha_tao_fp18 = alpha_tao_to_fp18(alpha_tao)
    tao_usdc_fp18 = int(TAO_USDC * (10**18))
    hashp_usdc_fp18 = int(HASHP_USDC * (10**18))

    commitment = PriceCommitment(
        t="p",
        prices={
            "ALPHA_TAO": alpha_tao_fp18,
            "TAO_USDC": tao_usdc_fp18,
            "HASHP_USDC": hashp_usdc_fp18,
        },
        v=1,
    )
    payload = commitment.to_compact().encode("utf-8")

    async with turbobt.Bittensor(f"ws://{SIM_HOST}:{SIM_RPC_PORT}") as bittensor:
        subnet = bittensor.subnet(NETUID)
        extrinsic = await subnet.commitments.set(data=payload, wallet=wallet)
        await extrinsic.wait_for_finalization()
    LOGGER.info("Validator commitment published", budget_ph=budget_ph)


async def advance_block(client: httpx.AsyncClient) -> dict:
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/blocks/advance",
        json={"steps": 1, "step_seconds": BLOCK_TIME},
    )
    response.raise_for_status()
    return response.json()["head"]


async def get_head(client: httpx.AsyncClient) -> dict:
    response = await client.get(f"http://{SIM_HOST}:{SIM_HTTP_PORT}/head")
    response.raise_for_status()
    return response.json()


async def get_commitments(client: httpx.AsyncClient, block_hash: str) -> dict:
    response = await client.get(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}/commitments",
        params={"block_hash": block_hash},
    )
    response.raise_for_status()
    return response.json()


def docker_compose_base_cmd() -> list[str]:
    """Return ['docker', 'compose'] or ['docker-compose'] depending on availability."""
    if shutil.which("docker"):
        try:
            subprocess.run(
                ["docker", "compose", "version"],
                check=True,
                capture_output=True,
            )
            return ["docker", "compose"]
        except subprocess.CalledProcessError:
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("docker compose command not found (need Docker Compose v2 or docker-compose v1)")


def copy_compose_file(dest_dir: Path) -> Path:
    """Copy the repository compose file into the working directory with test-specific tweaks."""
    compose_target = dest_dir / "docker-compose.yml"
    shutil.copy(COMPOSE_SOURCE, compose_target)

    # Replace home-directory wallet mount with the local test wallet directory.
    text = compose_target.read_text()
    text = text.replace("~/.bittensor:/root/.bittensor", "./wallets:/root/.bittensor")
    miner_image_override = os.environ.get("MINER_IMAGE")
    if miner_image_override:
        pattern = re.compile(r"(?m)^\s*image:\s+(ghcr\.io/backend-developers-ltd/infinitehash-subnet-prod@[\w:+-]+)")

        def _replace(match: re.Match[str]) -> str:
            original = match.group(1)
            if miner_image_override.startswith(original):
                return match.group(0)
            return match.group(0).replace(original, miner_image_override)

        text = pattern.sub(_replace, text)
        text = text.replace("pull_policy: always", "pull_policy: if_not_present")
    compose_target.write_text(text)
    return compose_target


@lru_cache(maxsize=1)
def _installer_script_text() -> str:
    if not MINER_INSTALLER_SOURCE.exists():
        raise RuntimeError(f"Installer script not found: {MINER_INSTALLER_SOURCE}")
    return MINER_INSTALLER_SOURCE.read_text(encoding="utf-8")


def _extract_single_heredoc_from_function(function_name: str) -> str:
    script = _installer_script_text()
    function_match = re.search(
        rf"(?ms)^{re.escape(function_name)}\(\)\s*\{{\n(?P<body>.*?)^\}}",
        script,
    )
    if function_match is None:
        raise RuntimeError(f"Function {function_name} not found in installer script")

    body = function_match.group("body")
    heredoc_matches = list(
        re.finditer(
            r"(?ms)<<'?(?P<tag>[A-Z_]+)'?\n(?P<content>.*?)\n\s*(?P=tag)",
            body,
        )
    )
    if len(heredoc_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one heredoc in installer function {function_name}, got {len(heredoc_matches)}"
        )
    return heredoc_matches[0].group("content").strip("\n") + "\n"


def create_ihp_proxy_env(proxy_dir: Path) -> Path:
    """Write default IHP .env file used by the integration test."""
    proxy_dir.mkdir(parents=True, exist_ok=True)
    env_path = proxy_dir / ".env"
    if env_path.exists():
        return env_path
    env_template = _extract_single_heredoc_from_function("write_default_ihp_env")
    env_path.write_text(env_template, encoding="utf-8")
    return env_path


def create_ihp_pools_config(
    proxy_dir: Path,
    backup_pool_host: str = DEFAULT_BACKUP_POOL_HOST,
    backup_pool_port: int = DEFAULT_BACKUP_POOL_PORT,
) -> Path:
    """Write default IHP pools.toml used by the integration test."""
    proxy_dir.mkdir(parents=True, exist_ok=True)
    pools_path = proxy_dir / "pools.toml"
    if pools_path.exists():
        return pools_path
    pools_template = _extract_single_heredoc_from_function("write_default_ihp_pools")
    pools_content = pools_template.replace("${backup_pool_host}", backup_pool_host).replace(
        "${backup_pool_port}", str(backup_pool_port)
    )
    pools_path.write_text(pools_content, encoding="utf-8")
    return pools_path


def create_miner_config(config_path: Path) -> None:
    """Generate miner TOML config pointed at the Docker host simulator."""
    content = f"""# Docker compose integration test config

[bittensor]
network = "ws://host.docker.internal:{SIM_RPC_PORT}"
netuid = {NETUID}

[wallet]
name = "test_miner"
hotkey_name = "default"
directory = "/root/.bittensor"

[workers]
price_multiplier = "1.0"
hashrates = [
    "5.5",
    "8.2",
    "6.8",
    "9.1",
    "7.3"
]
"""
    config_path.write_text(content, encoding="utf-8")


def load_subnet_target_hashrate(pools_path: Path, pool_name: str = "central-proxy") -> str | None:
    """Return current target_hashrate for subnet pool name from pools.toml."""
    doc = tomlkit.parse(pools_path.read_text(encoding="utf-8"))
    pools = doc.get("pools")
    if not isinstance(pools, dict):
        return None
    main_pools = pools.get("main")
    if not isinstance(main_pools, list):
        return None
    for pool in main_pools:
        if not isinstance(pool, dict):
            continue
        if str(pool.get("name", "")).strip().lower() != pool_name.lower():
            continue
        target_hashrate = pool.get("target_hashrate")
        if isinstance(target_hashrate, str):
            return target_hashrate
        return None
    return None


async def run_simulation(
    client: httpx.AsyncClient,
    validator_wallet,
    compose_cmd: list[str],
    workdir: Path,
    pools_path: Path,
    mechanism_share: float,
    miner_config_path: Path,
) -> None:
    """Advance the simulated chain and monitor IHP target hashrate updates."""
    services_to_tail = ["miner", "ihp-proxy", "ihp-api", "ihp-proxy-reloader"]
    log_processes: dict[str, subprocess.Popen[str]] = {}
    log_tasks: list[asyncio.Task[None]] = []
    for service in services_to_tail:
        cmd = compose_cmd + ["logs", "--no-color", "-f", service]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            LOGGER.warning("Could not tail logs for service %s", service)
            continue
        log_processes[service] = proc
        log_tasks.append(asyncio.create_task(stream_logs(service, proc)))

    head = await get_head(client)
    current_block = head["number"]
    last_window = current_block // BLOCKS_PER_WINDOW
    LOGGER.info("Simulation starting", start_block=current_block)

    baseline_target = load_subnet_target_hashrate(pools_path)
    LOGGER.info("Initial IHP subnet target hashrate", target_hashrate=baseline_target)

    commitment_scheduled = False
    hashrate_update_done = False
    LOGGER.info("Validator will publish next-window budgets on the final block of each window")

    try:
        while True:
            await asyncio.sleep(BLOCK_TIME)
            head = await advance_block(client)
            current_block = head["number"]
            current_window = current_block // BLOCKS_PER_WINDOW
            block_in_window = current_block % BLOCKS_PER_WINDOW

            LOGGER.info(
                "Advanced block",
                block=current_block,
                window=current_window,
                block_in_window=block_in_window,
            )

            if block_in_window == 10 and not hashrate_update_done:
                new_rates = random_hashrates()
                update_miner_config_hashrates(miner_config_path, new_rates)
                total_ph = sum(float(v) for v in new_rates)
                LOGGER.info(
                    "Miner config updated with new hashrates",
                    block=current_block,
                    total_ph=total_ph,
                    hashrates=new_rates,
                )
                hashrate_update_done = True

            if block_in_window == BLOCKS_PER_WINDOW - 1 and not commitment_scheduled:
                budget = RNG.uniform(20, 30)
                LOGGER.info(
                    "Validator publishing next-window budget",
                    block=current_block,
                    budget_ph=budget,
                )
                await submit_validator_commitment(validator_wallet, budget, mechanism_share)
                commitment_scheduled = True

            try:
                commits = await get_commitments(client, head["hash"])
                LOGGER.info("Commitments on chain", count=len(commits.get("entries", {})))
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to read commitments")

            target_hashrate = load_subnet_target_hashrate(pools_path)
            if target_hashrate != baseline_target:
                LOGGER.info("IHP subnet target hashrate updated", target_hashrate=target_hashrate)
                baseline_target = target_hashrate

            sentinel = pools_path.parent / ".reload-ihp"
            if sentinel.exists():
                LOGGER.info("IHP reload sentinel detected", path=str(sentinel))

            if current_window > last_window:
                last_window = current_window
                commitment_scheduled = False
                hashrate_update_done = False

    except KeyboardInterrupt:
        LOGGER.info("Simulation stopped by user")
    finally:
        for task in log_tasks:
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        for proc in log_processes.values():
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=5)


async def stream_logs(name: str, process: subprocess.Popen[str]) -> None:
    """Async helper to forward docker compose logs to stdout."""
    if not process.stdout:
        return
    loop = asyncio.get_running_loop()

    def reader():
        for line in process.stdout:
            print(f"[{name}] {line.rstrip()}")

    await loop.run_in_executor(None, reader)


async def main() -> int:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    workdir = Path(tempfile.mkdtemp(prefix="infhash_miner_compose_"))
    wallets_dir = workdir / "wallets"
    logs_dir = workdir / "logs"
    proxy_dir = workdir / "proxy"
    for path in (wallets_dir, logs_dir, proxy_dir):
        path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Working directory prepared", path=str(workdir))

    compose_cmd = docker_compose_base_cmd()
    compose_file = copy_compose_file(workdir)

    LOGGER.info("Compose file ready", compose_path=str(compose_file))

    miner_wallet_hotkey, _ = create_test_wallet(str(wallets_dir), name="test_miner")
    LOGGER.info("Miner wallet created", hotkey=miner_wallet_hotkey[:16])
    validator_hotkey, _ = create_test_wallet(str(wallets_dir), name="test_validator")
    LOGGER.info("Validator wallet created", hotkey=validator_hotkey[:16])
    validator_wallet = bittensor_wallet.Wallet(
        name="test_validator",
        hotkey="default",
        path=str(wallets_dir),
    )

    create_ihp_proxy_env(proxy_dir)
    pools_path = create_ihp_pools_config(proxy_dir)
    LOGGER.info("IHP proxy config written", pools_path=str(pools_path))

    miner_config_path = workdir / "config.toml"
    create_miner_config(miner_config_path)
    LOGGER.info("Miner config written", path=str(miner_config_path))

    ready_event = mp.Event()
    stop_event = mp.Event()
    # Bind simulator to all interfaces so Docker containers can reach the host.
    os.environ["SIM_HOST"] = "0.0.0.0"  # noqa: S104
    os.environ["SIM_HTTP_PORT"] = str(SIM_HTTP_PORT)
    os.environ["SIM_RPC_PORT"] = str(SIM_RPC_PORT)
    os.environ["DATABASE_URL"] = f"sqlite:///{workdir / 'simulator.sqlite3'}"
    sim_process = mp.Process(
        target=simulator_process_main,
        args=(ready_event, stop_event),
        daemon=True,
    )
    sim_process.start()
    if not ready_event.wait(timeout=10):
        LOGGER.error("Simulator failed to start")
        sim_process.terminate()
        return 1
    LOGGER.info("Simulator started", pid=sim_process.pid)

    client = httpx.AsyncClient(timeout=10.0)
    stack_started = False
    try:
        mechanism_split = await setup_blockchain(client, miner_wallet_hotkey, validator_hotkey)
        mechanism_share = mechanism_share_from_split(mechanism_split)
        subprocess.run(compose_cmd + ["up", "-d", "--remove-orphans"], cwd=workdir, check=True)
        LOGGER.info("Docker compose stack started")
        stack_started = True

        # Allow containers time to start and miner to load config.
        time.sleep(5)

        await run_simulation(
            client,
            validator_wallet,
            compose_cmd,
            workdir,
            pools_path,
            mechanism_share,
            miner_config_path,
        )
    except KeyboardInterrupt:
        LOGGER.info("Received interrupt, shutting down...")
    except Exception:  # noqa: BLE001
        LOGGER.exception("Manual IHP integration test failed")
        if stack_started:
            LOGGER.info("Collecting docker compose logs for diagnostics")
            with contextlib.suppress(Exception):
                subprocess.run(
                    compose_cmd + ["logs", "--no-color", "--tail", "400"],
                    cwd=workdir,
                    check=False,
                )
        return 1
    finally:
        LOGGER.info("Cleaning up")
        if stack_started:
            with contextlib.suppress(Exception):
                subprocess.run(compose_cmd + ["down", "-v"], cwd=workdir, check=False, timeout=30)
        stop_event.set()
        sim_process.terminate()
        sim_process.join(timeout=5)
        await client.aclose()
        with contextlib.suppress(Exception):
            shutil.rmtree(workdir)
        LOGGER.info("Cleanup complete", workdir=str(workdir))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(0)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
