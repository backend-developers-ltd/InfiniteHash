"""Director class for orchestrating scenario execution.

The Director:
1. Processes user-defined events from scenario timeline
2. Auto-injects processing events (sync, process auction, weights)
3. Manages validator/miner worker processes by name
4. Converts TimeAddress to absolute block numbers
5. Uses delivery hook for hashrate simulation
"""

import datetime as dt
import os
import random
import shutil
import tempfile
from collections.abc import Callable
from contextlib import redirect_stdout
from decimal import Decimal
from typing import Any

import bittensor_wallet
import httpx
import structlog
import tomli_w
from django.conf import settings

from infinite_hashes.auctions import utils as auction_utils
from infinite_hashes.testutils.integration.budget_helper import (
    DEFAULT_MECHANISM_1_SHARE,
    DEFAULT_MECHANISM_SPLIT,
    alpha_tao_to_fp18,
    compute_alpha_tao_for_budget,
)

from .scenario import (
    AssertFalseEvent,
    AssertWeightsEvent,
    ChangeWorkers,
    DeliveryHook,
    RegisterMiner,
    RegisterValidator,
    Scenario,
    ScenarioEvent,
    SetCommitment,
    SetDeliveryHook,
    SetPrices,
    TimeAddress,
)
from .worker_process import WorkerProcess

logger = structlog.get_logger(__name__)

BLOCK_INTERVAL_SECONDS = 12  # Real mainnet block time
SIM_HOST = "127.0.0.1"
SIM_HTTP_PORT = 8090

_BRAIINS_PROFILE_TEMPLATE = """[[server]]
name = "InfiniteHash"
port = 3333

[[target]]
name = "InfiniteHashLuxorTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "InfiniteHashLuxor"
identity_pass_through = true

[[target]]
name = "MinerDefaultTarget"
url = "stratum+tcp://btc.global.luxor.tech:700"
user_identity = "MinerDefault"
identity_pass_through = true

[[target]]
name = "MinerBackupTarget"
url = "stratum+tcp://btc.viabtc.io:3333"
user_identity = "MinerBackup"
identity_pass_through = true

[[routing]]
name = "RD"
from = ["InfiniteHash"]

[[routing.goal]]
name = "InfiniteHashLuxorGoal"
hr_weight = 9

[[routing.goal.level]]
targets = ["InfiniteHashLuxorTarget"]

[[routing.goal]]
name = "MinerDefaultGoal"
hr_weight = 10

[[routing.goal.level]]
targets = ["MinerDefaultTarget"]

[[routing.goal.level]]
targets = ["MinerBackupTarget"]
"""


class Director:
    """Orchestrates scenario execution with event-driven timeline."""

    @staticmethod
    def _write_miner_toml_config(
        config_path: str,
        *,
        network: str,
        netuid: int,
        wallet_name: str,
        wallet_hotkey: str,
        wallet_dir: str,
        hashrates: list[str],
        price_multiplier: str,
    ) -> None:
        """Write APS miner TOML configuration file."""
        config_data = {
            "bittensor": {
                "network": network,
                "netuid": netuid,
            },
            "wallet": {
                "name": wallet_name,
                "hotkey_name": wallet_hotkey,
                "directory": wallet_dir,
            },
            "workers": {
                "price_multiplier": price_multiplier,
                "hashrates": hashrates,
            },
        }
        with open(config_path, "wb") as f:
            tomli_w.dump(config_data, f)

    @staticmethod
    def _ensure_braiins_profile(profile_path: str) -> None:
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)
        if os.path.exists(profile_path):
            return
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(_BRAIINS_PROFILE_TEMPLATE)

    def __init__(
        self,
        scenario: Scenario,
        http_client: httpx.AsyncClient,
        *,
        initial_timestamp: dt.datetime,
        run_suffix: str,
        validator_worker_main: Callable,
        miner_worker_main: Callable,
    ) -> None:
        self.scenario = scenario
        self.client = http_client
        self.initial_timestamp = initial_timestamp
        self.run_suffix = run_suffix
        self.validator_worker_main = validator_worker_main
        self.miner_worker_main = miner_worker_main
        self.netuid = 1

        # Blockchain state
        self.tempo = 360  # Mainnet tempo
        self.blocks_per_window = auction_utils.blocks_per_window_default()
        self.initial_block = 1080  # Subnet epoch boundary
        self.current_head = self.initial_block

        # Entity registries (name -> worker process and metadata)
        self.validators: dict[str, WorkerProcess] = {}
        self.miners: dict[str, WorkerProcess] = {}
        self.inactive_miners: dict[str, WorkerProcess] = {}  # Replaced miners, stopped at end
        self.validator_configs: dict[str, dict[str, Any]] = {}
        self.miner_configs: dict[str, dict[str, Any]] = {}
        self.validator_neurons: dict[str, dict[str, Any]] = {}
        self.miner_neurons: dict[str, dict[str, Any]] = {}
        self._temp_dirs: list[str] = []

        # Worker ID counter (validators get 0-N, miners get N+1 onwards)
        self.next_worker_id = 0

        # Luxor subaccount for delivery verification
        self.luxor_subaccount_mechanism_1 = settings.LUXOR_SUBACCOUNT_NAME_MECHANISM_1
        self.mechanism_split_u16 = list(DEFAULT_MECHANISM_SPLIT)
        self.mechanism_1_share = DEFAULT_MECHANISM_1_SHARE

        # Hierarchical delivery hooks (mutable, can be changed via SetDeliveryHook events)
        # Keys can be "miner_name" or "miner_name.worker_identifier"
        self.delivery_hooks: dict[str, DeliveryHook] = dict(scenario.delivery_hooks)
        self.default_delivery_hook: DeliveryHook = scenario.default_delivery_hook

        # Track when validators commit weights for each epoch
        # weight_commit_blocks[epoch][validator_name] = block_number
        self.weight_commit_blocks: dict[int, dict[str, int]] = {}

        # Track last budget_ph set for each validator (for auto-injected price scraping)
        self.validator_budgets: dict[str, float | None] = {}

        # Subnet owner (for burn mechanism)
        self.owner_hotkey: str | None = None
        self.owner_coldkey: str | None = None
        self.owner_uid: int | None = None

    def get_delivery_hook(self, miner_name: str, worker_identifier: str) -> DeliveryHook:
        """Get delivery hook with hierarchical lookup.

        Lookup order:
        1. Worker-specific: "miner_name.worker_identifier"
        2. Miner-level: "miner_name"
        3. Default hook
        """
        # Try worker-specific hook first
        worker_key = f"{miner_name}.{worker_identifier}"
        if worker_key in self.delivery_hooks:
            return self.delivery_hooks[worker_key]

        # Try miner-level hook
        if miner_name in self.delivery_hooks:
            return self.delivery_hooks[miner_name]

        # Fall back to default
        return self.default_delivery_hook

    def time_to_block(self, time: TimeAddress) -> int:
        """Convert TimeAddress to absolute block number.

        Epoch 0, window 0, block 0 corresponds to the validation epoch start
        (60 blocks before subnet epoch boundary).
        """
        # Each subnet epoch is tempo+1 blocks (361 blocks for tempo=360)
        # Validation epoch starts 60 blocks before subnet epoch
        # For epoch 0: validation starts at initial_block - 60
        validation_epoch_start = self.initial_block + (time.epoch * (self.tempo + 1)) - 60

        # Each window is blocks_per_window blocks
        window_offset = time.window * self.blocks_per_window

        # Block within window
        block_offset = time.block

        return validation_epoch_start + window_offset + block_offset

    def block_to_time(self, block: int) -> TimeAddress:
        """Convert absolute block number to TimeAddress."""
        # Calculate blocks from epoch 0 window 0 block 0
        validation_epoch_start = self.initial_block - 60
        delta = block - validation_epoch_start

        # Subnet epochs are tempo+1 blocks (361 for tempo=360)
        subnet_epoch_length = self.tempo + 1
        epoch = delta // subnet_epoch_length
        remaining = delta % subnet_epoch_length
        window = remaining // self.blocks_per_window
        block_in_window = remaining % self.blocks_per_window

        return TimeAddress(epoch, 0, 0).w_dt(window, block_in_window)

    def block_timestamp(self, block_number: int) -> dt.datetime:
        """Get timestamp for a specific block."""
        delta = block_number - self.initial_block
        return self.initial_timestamp + dt.timedelta(seconds=BLOCK_INTERVAL_SECONDS * delta)

    async def _register_subnet_owner(self) -> None:
        """Create and register subnet owner neuron (for burn mechanism)."""
        from contextlib import redirect_stdout

        # Create owner wallet
        wallet_dir = self.scenario.wallet_dir
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            wallet = bittensor_wallet.Wallet(name="subnet_owner", hotkey="default", path=wallet_dir)
            wallet.create_new_coldkey(n_words=12, use_password=False, overwrite=True)
            wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True)
        wallet_reloaded = bittensor_wallet.Wallet(name="subnet_owner", hotkey="default", path=wallet_dir)

        self.owner_hotkey = wallet_reloaded.hotkey.ss58_address
        self.owner_coldkey = wallet_reloaded.coldkey.ss58_address
        self.owner_uid = 0  # Owner gets UID 0

        # Create owner neuron
        owner_neuron = {
            "hotkey": self.owner_hotkey,
            "uid": self.owner_uid,
            "stake": 100_000.0,  # Large stake for owner
            "coldkey": self.owner_coldkey,
        }

        # Register owner on blockchain
        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/neurons",
            json={"neurons": [owner_neuron]},
        )
        response.raise_for_status()

        # Set owner in simulator state
        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}",
            json={
                "owner_hotkey": self.owner_hotkey,
                "owner_coldkey": self.owner_coldkey,
            },
        )
        response.raise_for_status()

        logger.info(
            "Registered subnet owner",
            owner_hotkey=self.owner_hotkey,
            owner_coldkey=self.owner_coldkey,
            owner_uid=self.owner_uid,
        )

    async def _initialize_mechanisms(self) -> None:
        """Ensure simulator has two mechanisms with configured emission split."""
        payload = {
            "count": len(self.mechanism_split_u16),
            "split": self.mechanism_split_u16,
        }
        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/mechanism-split",
            json=payload,
        )
        response.raise_for_status()
        logger.info(
            "Configured mechanism emission split",
            split=payload["split"],
            count=payload["count"],
        )

    async def advance_to_time(self, time: TimeAddress) -> None:
        """Advance simulator to a specific TimeAddress."""
        target_block = self.time_to_block(time)
        await self.advance_to_block(target_block)

    async def advance_to_block(self, target_block: int) -> None:
        """Advance simulator to a specific block number."""
        if target_block < self.current_head:
            raise RuntimeError(f"cannot rewind head from {self.current_head} to {target_block}")
        steps = target_block - self.current_head
        if steps == 0:
            return

        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/blocks/advance",
            json={"steps": steps, "step_seconds": BLOCK_INTERVAL_SECONDS},
        )
        response.raise_for_status()
        head_info = response.json().get("head", {})
        self.current_head = int(head_info.get("number", target_block))

        logger.info(
            "Advanced simulator",
            target_block=target_block,
            steps=steps,
            current_head=self.current_head,
            timestamp=head_info.get("timestamp"),
        )

    async def setup_blockchain_state(self, starting_time: TimeAddress | None = None) -> None:
        """Initialize blockchain state (tempo and initial block).

        Args:
            starting_time: Optional time address to start blockchain at.
                          If None, starts at validation epoch 0 start.
        """
        # Set tempo
        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}",
            json={"tempo": self.tempo},
        )
        response.raise_for_status()
        payload = response.json()
        self.tempo = int(payload.get("tempo", self.tempo))

        # Determine starting block
        if starting_time is not None:
            start_block = self.time_to_block(starting_time)
        else:
            # Default: validation epoch 0 starts 60 blocks before subnet epoch
            start_block = self.initial_block - 60

        # Set head to starting block
        start_ts = self.block_timestamp(start_block)
        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/head",
            json={
                "number": start_block,
                "timestamp": start_ts.isoformat().replace("+00:00", "Z"),
            },
        )
        response.raise_for_status()
        self.current_head = start_block

        # Create and register subnet owner (for burn mechanism)
        await self._register_subnet_owner()
        await self._initialize_mechanisms()

        logger.info(
            "Blockchain state initialized",
            tempo=self.tempo,
            starting_time=str(starting_time) if starting_time else None,
            current_head=self.current_head,
            owner_uid=self.owner_uid,
        )

    async def process_register_validator(self, event: RegisterValidator) -> None:
        """Register a new validator and immediately start it."""
        validator_idx = len(self.validators)
        schema_name = f"test_validator_{self.run_suffix}_{validator_idx}"

        # Create wallet
        wallet_dir = self.scenario.wallet_dir
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            wallet = bittensor_wallet.Wallet(name=event.name, hotkey="default", path=wallet_dir)
            wallet.create_new_coldkey(n_words=12, use_password=False, overwrite=True)
            wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True)
        wallet_reloaded = bittensor_wallet.Wallet(name=event.name, hotkey="default", path=wallet_dir)
        hotkey = wallet_reloaded.hotkey.ss58_address

        # Create config
        config = {
            "db_path": os.path.join(tempfile.gettempdir(), f"validator_{validator_idx}.db"),
            "wallet_dir": wallet_dir,
            "wallet_name": event.name,
            "wallet_hotkey": "default",
            "auction_window_blocks": self.blocks_per_window,
            "luxor_url": settings.LUXOR_API_URL,
            "luxor_subaccount": self.luxor_subaccount_mechanism_1,
            "db_schema": schema_name,
            **event.wallet_config,
        }

        # Create neuron metadata
        neuron = {
            "hotkey": hotkey,
            "uid": len(self.validator_neurons) + len(self.miner_neurons) + 1,  # +1 for owner at UID 0
            "stake": event.stake,
            "coldkey": wallet_reloaded.coldkey.ss58_address,
        }

        # Create worker process
        worker = WorkerProcess(
            worker_id=self.next_worker_id,
            target=self.validator_worker_main,
            config=config,
        )
        self.next_worker_id += 1

        # Store in registries
        self.validators[event.name] = worker
        self.validator_configs[event.name] = config
        self.validator_neurons[event.name] = neuron

        # Start worker and register neuron on blockchain
        worker.start()

        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/neurons",
            json={"neurons": [neuron]},
        )
        response.raise_for_status()

        logger.info(
            "Registered validator",
            name=event.name,
            hotkey=hotkey,
            uid=neuron["uid"],
            stake=event.stake,
            time=str(event.time),
        )

    async def process_register_miner(self, event: RegisterMiner) -> None:
        """Register a new miner and immediately start it.

        If replace_miner is specified, the new miner takes over that miner's UID
        and the old miner is stopped and removed.

        APS miner uses TOML config with single price for all workers.
        """
        miner_idx = len(self.miners)

        # Handle miner replacement - stop and remove old miner if replacing
        old_miner_name = event.replace_miner
        replaced_uid = None

        if old_miner_name is not None:
            # Check if the old miner exists
            if old_miner_name in self.miner_neurons:
                replaced_uid = self.miner_neurons[old_miner_name]["uid"]

                logger.info(
                    "Replacing miner",
                    old_miner=old_miner_name,
                    new_miner=event.name,
                    uid=replaced_uid,
                )

                # Move old miner to inactive (will be stopped at end of scenario)
                old_worker = self.miners.pop(old_miner_name, None)
                if old_worker:
                    self.inactive_miners[old_miner_name] = old_worker

                # Remove old miner from configs and neurons
                self.miner_configs.pop(old_miner_name, None)
                self.miner_neurons.pop(old_miner_name, None)
            else:
                logger.warning(
                    "Miner replacement requested but miner not found",
                    miner_name=old_miner_name,
                )

        # APS miner requirement: assert all workers have same price
        prices = {w.get("price_multiplier") for w in event.workers}
        if len(prices) > 1:
            raise ValueError(
                f"APS miner requires all workers to have the same price_multiplier. "
                f"Miner {event.name} has workers with different prices: {prices}. "
                f"See integration test notes for details."
            )
        price_multiplier = str(prices.pop() if prices else "1.0")

        # Create wallet
        wallet_dir = self.scenario.wallet_dir
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            wallet = bittensor_wallet.Wallet(name=event.name, hotkey="default", path=wallet_dir)
            wallet.create_new_coldkey(n_words=12, use_password=False, overwrite=True)
            wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True)
        wallet_reloaded = bittensor_wallet.Wallet(name=event.name, hotkey="default", path=wallet_dir)
        hotkey = wallet_reloaded.hotkey.ss58_address

        # Extract hashrates for APS miner TOML config
        hashrates = [w.get("hashrate_ph") for w in event.workers]

        # Prepare per-miner working directory
        miner_workdir = tempfile.mkdtemp(prefix=f"aps_miner_{self.run_suffix}_{miner_idx}_")
        self._temp_dirs.append(miner_workdir)
        config_path = os.path.join(miner_workdir, "config.toml")
        self._write_miner_toml_config(
            config_path,
            network="ws://127.0.0.1:9944",  # Use simulator network
            netuid=self.netuid,  # Use director's netuid (1)
            wallet_name=event.name,
            wallet_hotkey="default",
            wallet_dir=wallet_dir,
            hashrates=hashrates,
            price_multiplier=price_multiplier,
        )

        brainsproxy_dir = os.path.join(miner_workdir, "brainsproxy")
        active_profile_path = os.path.join(brainsproxy_dir, "active_profile.toml")
        self._ensure_braiins_profile(active_profile_path)
        reload_sentinel_path = os.path.join(brainsproxy_dir, ".reconfigure")
        logs_dir = os.path.join(miner_workdir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create config dict (for worker process and delivery simulation)
        config = {
            "config_path": config_path,  # APS miner uses config path
            "wallet_dir": wallet_dir,
            "wallet_name": event.name,
            "wallet_hotkey": "default",
            "auction_window_blocks": self.blocks_per_window,
            "luxor_url": settings.LUXOR_API_URL,
            "hotkey": hotkey,
            "workers": event.workers,  # Keep original format for delivery simulation
            "brainsproxy_active_profile": active_profile_path,
            "brainsproxy_reload_sentinel": reload_sentinel_path,
            "logs_dir": logs_dir,
            "work_dir": miner_workdir,
            **event.wallet_config,
        }

        # Determine UID
        if replaced_uid is not None:
            uid = replaced_uid
        else:
            uid = len(self.validator_neurons) + len(self.miner_neurons) + 1  # +1 for owner at UID 0

        # Create neuron metadata
        neuron = {
            "hotkey": hotkey,
            "uid": uid,
            "stake": 1000.0,  # Default stake for miners
            "coldkey": wallet_reloaded.coldkey.ss58_address,
        }

        # Create worker process
        worker = WorkerProcess(
            worker_id=self.next_worker_id,
            target=self.miner_worker_main,
            config=config,
        )
        self.next_worker_id += 1

        # Store in registries
        self.miners[event.name] = worker
        self.miner_configs[event.name] = config
        self.miner_neurons[event.name] = neuron

        # Start worker and register neuron on blockchain
        worker.start()

        response = await self.client.post(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/neurons",
            json={"neurons": [neuron]},
        )
        response.raise_for_status()

        logger.info(
            "Registered miner (APS)",
            name=event.name,
            hotkey=hotkey,
            uid=neuron["uid"],
            workers=len(event.workers),
            price_multiplier=price_multiplier,
            config_path=config_path,
            replaced=old_miner_name,
            time=str(event.time),
        )

    async def process_set_prices(self, event: SetPrices) -> None:
        """Validator scrapes prices and publishes commitment.

        If ph_budget is specified, adjusts ALPHA_TAO price to achieve target budget.
        """
        validator = self.validators.get(event.validator_name)
        if not validator:
            raise RuntimeError(f"validator {event.validator_name} not found")

        # Store budget for future auto-injected price updates
        if event.ph_budget is not None:
            self.validator_budgets[event.validator_name] = event.ph_budget

        # Compute prices (optionally with custom budget)
        if event.ph_budget is not None:
            alpha_tao = compute_alpha_tao_for_budget(
                event.ph_budget,
                mechanism_share=self.mechanism_1_share,
            )
            alpha_tao_fp18 = alpha_tao_to_fp18(alpha_tao)

            # Send custom prices
            validator.send_command(
                "SCRAPE_PRICES",
                {
                    "ALPHA_TAO": alpha_tao_fp18,
                },
            )

            logger.info(
                "Set prices with custom budget",
                validator=event.validator_name,
                ph_budget=event.ph_budget,
                alpha_tao=alpha_tao,
                time=str(event.time),
            )
        else:
            # Use default simulated prices
            validator.send_command("SCRAPE_PRICES")

            logger.info(
                "Set prices",
                validator=event.validator_name,
                time=str(event.time),
            )

        validator.send_command("PUBLISH_LOCAL_COMMITMENT")
        # send_command raises exception on failure

    async def process_set_commitment(self, event: SetCommitment) -> None:
        """Miner publishes bidding commitment."""
        miner = self.miners.get(event.miner_name)
        if not miner:
            raise RuntimeError(f"miner {event.miner_name} not found")

        miner.send_command("SUBMIT_COMMITMENT", {"force": True, **event.commitment_data})

        logger.info(
            "Set commitment",
            miner=event.miner_name,
            time=str(event.time),
        )

    async def process_change_workers(self, event: ChangeWorkers) -> None:
        """Change a miner's worker configuration in-place.

        For APS miner: updates TOML config file (hot reload handles the rest).
        """
        miner_process = self.miners.get(event.miner_name)
        if not miner_process:
            raise RuntimeError(f"miner {event.miner_name} not found")

        miner_config = self.miner_configs.get(event.miner_name)
        if miner_config is None:
            raise RuntimeError(f"miner config for {event.miner_name} not found")

        # APS miner requirement: assert all workers have same price
        prices = {w.get("price_multiplier") for w in event.workers}
        if len(prices) > 1:
            raise ValueError(
                f"APS miner requires all workers to have the same price_multiplier. "
                f"ChangeWorkers for {event.miner_name} has workers with different prices: {prices}. "
                f"See integration test notes for details."
            )
        price_multiplier = str(prices.pop() if prices else "1.0")

        normalized_workers: list[dict[str, Any]] = []
        hashrates: list[str] = []
        for worker in event.workers:
            identifier = worker.get("identifier")
            if not identifier:
                raise ValueError(f"worker entry missing identifier: {worker}")
            hashrate = worker.get("hashrate_ph")
            price = worker.get("price_multiplier")
            if hashrate is None or price is None:
                raise ValueError(f"worker {identifier} missing hashrate or price multiplier")
            normalized_workers.append(
                {
                    **worker,
                    "identifier": identifier,
                    "hashrate_ph": str(hashrate),
                    "price_multiplier": str(price),
                }
            )
            hashrates.append(str(hashrate))

        # Update simulator-side config used for delivery simulation
        miner_config["workers"] = normalized_workers

        # Update TOML config file (APS miner hot reload will pick up changes)
        config_path = miner_config.get("config_path")
        if config_path:
            self._write_miner_toml_config(
                config_path,
                network="ws://127.0.0.1:9944",  # Use simulator network
                netuid=self.netuid,  # Use director's netuid (1)
                wallet_name=miner_config["wallet_name"],
                wallet_hotkey=miner_config["wallet_hotkey"],
                wallet_dir=miner_config["wallet_dir"],
                hashrates=hashrates,
                price_multiplier=price_multiplier,
            )

        logger.info(
            "Updated miner workers (APS hot reload)",
            miner=event.miner_name,
            worker_count=len(normalized_workers),
            price_multiplier=price_multiplier,
            time=str(event.time),
        )

    async def process_set_delivery_hook(self, event: SetDeliveryHook) -> None:
        """Change the delivery hook for a miner or specific worker at runtime.

        Target can be:
        - "miner_0" - applies to all workers of miner_0
        - "miner_0.worker1" - applies only to worker1 of miner_0
        """
        self.delivery_hooks[event.target] = event.hook
        logger.info(
            "Changed delivery hook",
            target=event.target,
            time=str(event.time),
        )

    async def process_assert_weights(self, event: AssertWeightsEvent) -> None:
        """Assert that validators have committed expected weights for a specific epoch.

        Converts normalized weights (sum=1.0) to max-based format (0-65535)
        and compares with simulator values at the calculated reveal block.

        Uses cached commit blocks tracked by the Director to determine when
        each validator committed their weights for the specified epoch.
        """
        epoch = event.for_epoch

        # Get cached commit blocks for this epoch
        if epoch not in self.weight_commit_blocks:
            raise ValueError(
                f"No weight commits found for epoch {epoch}. "
                f"Available epochs: {sorted(self.weight_commit_blocks.keys())}"
            )

        epoch_commits = self.weight_commit_blocks[epoch]

        # Verify all expected validators have committed
        missing_validators = set(event.expected_weights.keys()) - set(epoch_commits.keys())
        if missing_validators:
            raise ValueError(
                f"Missing weight commits for validators in epoch {epoch}: {missing_validators}. "
                f"Validators that committed: {sorted(epoch_commits.keys())}"
            )

        # Get commit block from first validator
        first_validator = next(iter(epoch_commits.keys()))
        commit_block = epoch_commits[first_validator]

        tempo_plus_one = self.tempo + 1
        netuid_plus_one = self.netuid + 1

        # Calculate which subnet epoch contains the commit block (absolute blockchain epoch)
        # Inverse of: first_block = (epoch * tempo_plus_one) - netuid_plus_one
        commit_subnet_epoch = (commit_block + netuid_plus_one) // tempo_plus_one

        # Calculate reveal epoch (3 epochs after commit)
        reveal_period = 3
        reveal_epoch = commit_subnet_epoch + reveal_period

        # First block inside the reveal epoch
        first_reveal_block = (reveal_epoch * tempo_plus_one) - netuid_plus_one

        # Target block (first reveal block + security offset of 3)
        # SECURITY_BLOCK_OFFSET ensures weights are definitely stored before querying
        SECURITY_BLOCK_OFFSET = 3
        target_block = first_reveal_block + SECURITY_BLOCK_OFFSET

        logger.info(
            "Asserting weights",
            time=str(event.time),
            for_epoch=epoch,
            commit_block=commit_block,
            commit_subnet_epoch=commit_subnet_epoch,
            reveal_epoch=reveal_epoch,
            first_reveal_block=first_reveal_block,
            target_block=target_block,
            validators=list(event.expected_weights.keys()),
        )

        # Query simulator for all validators' weights at target block
        initial_weights_response = await self.client.get(
            f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/mechanisms/1/weights",
            params={"block": target_block},
        )
        initial_weights_response.raise_for_status()
        aggregated_weights = initial_weights_response.json().get("weights", {})

        logger.debug(
            "Queried simulator for weights",
            for_epoch=epoch,
            target_block=target_block,
            validator_count=len(aggregated_weights),
            returned_hotkeys=list(aggregated_weights.keys()),
        )

        # Build uid -> miner/owner name mapping
        uid_to_miner = {neuron["uid"]: name for name, neuron in self.miner_neurons.items()}

        # Add owner with special key "__owner__" for burn weight assertions
        if self.owner_uid is not None:
            uid_to_miner[self.owner_uid] = "__owner__"

        # Check each validator's weights
        all_passed = True
        failed_validators = []  # Track failures for detailed error message
        for validator_name, expected_normalized in event.expected_weights.items():
            # Convert normalized weights to max-based (0-65535)
            max_weight = max(expected_normalized.values()) if expected_normalized else 1.0
            expected_maxbased = {}
            for miner_name, norm_weight in expected_normalized.items():
                maxbased = int((norm_weight / max_weight) * 65535 + 0.5)
                expected_maxbased[miner_name] = maxbased

            # Find this validator's hotkey and actual weights
            validator_neuron = self.validator_neurons.get(validator_name)
            if not validator_neuron:
                logger.error(
                    "Validator not found in neurons",
                    validator=validator_name,
                    time=str(event.time),
                )
                all_passed = False
                continue

            validator_hotkey = validator_neuron["hotkey"]

            weights_response = await self.client.get(
                f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{self.netuid}/mechanisms/1/weights",
                params={"block": target_block, "hotkey": validator_hotkey},
            )
            weights_response.raise_for_status()
            actual_weights_by_uid = weights_response.json().get("weights", {})

            if not actual_weights_by_uid:
                logger.error(
                    "No weights found for validator",
                    validator=validator_name,
                    expected_hotkey=validator_hotkey,
                    returned_hotkeys=list(aggregated_weights.keys()),
                    time=str(event.time),
                )
                all_passed = False
                continue

            # Convert actual weights to miner names
            actual_weights = {}
            for uid, weight in actual_weights_by_uid.items():
                uid_int = int(uid)
                miner_name = uid_to_miner.get(uid_int)
                if miner_name:
                    actual_weights[miner_name] = weight

            # Compare weights
            mismatches = []
            all_miners = set(expected_maxbased.keys()) | set(actual_weights.keys())
            for miner_name in all_miners:
                expected = expected_maxbased.get(miner_name, 0)
                actual = actual_weights.get(miner_name, 0)
                # Allow small rounding differences (±50) to cover normalization drift
                if abs(actual - expected) > 50:
                    mismatches.append(
                        {
                            "miner": miner_name,
                            "expected": expected,
                            "actual": actual,
                            "diff": actual - expected,
                        }
                    )

            if mismatches:
                all_passed = False
                failed_validators.append(
                    {
                        "validator": validator_name,
                        "expected_normalized": expected_normalized,
                        "expected_maxbased": expected_maxbased,
                        "actual": actual_weights,
                        "mismatches": mismatches,
                    }
                )
                logger.error(
                    "Weight assertion FAILED",
                    validator=validator_name,
                    time=str(event.time),
                    target_block=target_block,
                    mismatches=mismatches,
                )
            else:
                logger.info(
                    "Weight assertion PASSED",
                    validator=validator_name,
                    time=str(event.time),
                    target_block=target_block,
                    num_weights=len(expected_maxbased),
                )

        if not all_passed:
            # Build detailed error message
            error_lines = [
                f"Weight assertion FAILED for epoch {epoch} at {event.time} (block {target_block})",
                f"Failed validators: {len(failed_validators)}/{len(event.expected_weights)}",
                "",
            ]

            for failure in failed_validators:
                error_lines.append(f"Validator: {failure['validator']}")
                error_lines.append("  Expected (normalized):")
                for miner, norm_weight in sorted(failure["expected_normalized"].items()):
                    error_lines.append(f"    {miner}: {norm_weight:.4f}")
                error_lines.append("  Expected (max-based):")
                for miner, maxbased in sorted(failure["expected_maxbased"].items()):
                    error_lines.append(f"    {miner}: {maxbased}")
                error_lines.append("  Actual:")
                for miner, actual in sorted(failure["actual"].items()):
                    error_lines.append(f"    {miner}: {actual}")
                error_lines.append("  Mismatches:")
                for mismatch in failure["mismatches"]:
                    error_lines.append(
                        f"    {mismatch['miner']}: expected {mismatch['expected']}, "
                        f"got {mismatch['actual']} (diff: {mismatch['diff']:+d})"
                    )
                error_lines.append("")

            raise AssertionError("\n".join(error_lines))

    async def process_assert_false(self, event: AssertFalseEvent) -> None:
        """Stop simulation by raising an assertion failure.

        Useful for debugging - allows stopping at a specific time to inspect state.
        """
        logger.warning(
            "Simulation stopped by AssertFalseEvent",
            time=str(event.time),
            message=event.message,
        )
        raise AssertionError(f"AssertFalseEvent at {event.time}: {event.message}")

    async def process_event(self, event: ScenarioEvent) -> None:
        """Dispatch event to appropriate handler."""
        if isinstance(event, RegisterValidator):
            await self.process_register_validator(event)
        elif isinstance(event, RegisterMiner):
            await self.process_register_miner(event)
        elif isinstance(event, SetPrices):
            await self.process_set_prices(event)
        elif isinstance(event, SetCommitment):
            await self.process_set_commitment(event)
        elif isinstance(event, ChangeWorkers):
            await self.process_change_workers(event)
        elif isinstance(event, SetDeliveryHook):
            await self.process_set_delivery_hook(event)
        elif isinstance(event, AssertWeightsEvent):
            await self.process_assert_weights(event)
        elif isinstance(event, AssertFalseEvent):
            await self.process_assert_false(event)
        else:
            raise RuntimeError(f"unknown event type: {type(event)}")

    async def simulate_hashrate_for_window(self, epoch: int, window: int) -> None:
        """Generate and insert hashrate snapshot data and scraping events for a single window.

        Creates:
        - Hashrate snapshots at 1-minute intervals (matching Luxor update frequency)
        - Scraping events (heartbeats) at 30-second intervals (matching validator scraping frequency)

        This simulates production behavior where data arrives progressively and validators
        continuously scrape data, creating a heartbeat trail for data completeness verification.

        Each validator worker inserts the data into its own database schema.
        """
        # Calculate window boundaries
        window_start_block = self.time_to_block(TimeAddress(epoch, window, 0))
        # Window 0-4 are 60 blocks, window 5 is 61 blocks
        window_duration = 61 if window == 5 else 60
        window_end_block = window_start_block + window_duration - 1

        window_start_ts = self.block_timestamp(window_start_block)
        window_end_ts = self.block_timestamp(window_end_block)

        # Scraper captures data every ~1 minute (matching Luxor update frequency)
        tick_seconds = 60

        # Collect snapshot data for all miners in this window
        snapshot_data = []
        current_ts = window_start_ts
        minute_count = 0

        while current_ts <= window_end_ts:
            minute_count += 1
            current_time_address = self.block_to_time(
                window_start_block + int((current_ts - window_start_ts).total_seconds() // BLOCK_INTERVAL_SECONDS)
            )

            for miner_name, miner_config in self.miner_configs.items():
                # Get miner hotkey for worker naming
                miner_hotkey = self.miner_neurons[miner_name]["hotkey"]

                # Create snapshots for each worker
                for worker_cfg in miner_config["workers"]:
                    worker_identifier = worker_cfg["identifier"]

                    # Get delivery params using hierarchical lookup (worker -> miner -> default)
                    worker_hook = self.get_delivery_hook(miner_name, worker_identifier)
                    delivery_params = worker_hook(miner_name, current_time_address)

                    hashrate_ph = Decimal(worker_cfg["hashrate_ph"])
                    multiplier = Decimal(str(delivery_params.sample_multiplier()))

                    # Apply dropout
                    if random.random() < delivery_params.dropout_rate:
                        hashrate_hs = 0
                        efficiency = 0.0
                    else:
                        hashrate_hs = int((hashrate_ph * multiplier * Decimal(10) ** 15).to_integral_value())
                        efficiency = 95.0

                    # Worker name format: {hotkey}.{identifier}
                    worker_name = f"{miner_hotkey}.{worker_identifier}"

                    snapshot_data.append(
                        {
                            "snapshot_time": current_ts.isoformat(),  # Serialize datetime
                            "subaccount_name": self.luxor_subaccount_mechanism_1,
                            "worker_name": worker_name,
                            "hashrate": hashrate_hs,
                            "efficiency": float(efficiency),
                            "revenue": 0.0,
                            "last_updated": current_ts.isoformat(),  # Serialize datetime
                        }
                    )

            current_ts += dt.timedelta(seconds=tick_seconds)

        # Generate scraping events (validator heartbeats)
        # Scraping happens every ~30 seconds (every ~2.5 blocks at 12s/block)
        scraping_interval_seconds = 30
        scraping_events = []
        scraping_ts = window_start_ts

        while scraping_ts <= window_end_ts:
            # Calculate block number for this scraping event
            blocks_from_start = int((scraping_ts - window_start_ts).total_seconds() // BLOCK_INTERVAL_SECONDS)
            scraping_block = window_start_block + blocks_from_start

            # Count total workers at this time (for the worker_count field)
            total_workers = sum(len(miner_config["workers"]) for miner_config in self.miner_configs.values())

            scraping_events.append(
                {
                    "scraped_at": scraping_ts.isoformat(),  # Serialize datetime
                    "block_number": scraping_block,
                    "worker_count": total_workers,
                }
            )

            scraping_ts += dt.timedelta(seconds=scraping_interval_seconds)

        # Send snapshot data to each validator to insert into their own database
        for validator_name in self.validators:
            # Insert hashrate samples
            result = self.validators[validator_name].send_command(
                "INSERT_HASHRATE_SAMPLES", {"snapshots": snapshot_data}
            )
            logger.debug(
                "Inserted hashrate samples",
                validator=validator_name,
                epoch=epoch,
                window=window,
                created=result.get("created", 0),
            )

            # Insert scraping events (heartbeats)
            result = self.validators[validator_name].send_command("INSERT_SCRAPING_EVENTS", {"events": scraping_events})
            logger.debug(
                "Inserted scraping events",
                validator=validator_name,
                epoch=epoch,
                window=window,
                created=result.get("created", 0),
            )

        logger.debug(
            "Simulated hashrate for window",
            epoch=epoch,
            window=window,
            window_range=(window_start_block, window_end_block),
            minutes_covered=minute_count,
            snapshots_count=len(snapshot_data),
            scraping_events_count=len(scraping_events),
        )

    async def process_window(self, epoch: int, window: int) -> None:
        """Process a single validation window with auto-injected events."""
        window_start = TimeAddress(epoch, window, 0)
        # Window 0-4 are 60 blocks, window 5 is 61 blocks
        window_duration = 61 if window == 5 else 60

        logger.info(
            "Processing window",
            epoch=epoch,
            window=window,
            time_start=str(window_start),
            block_start=self.time_to_block(window_start),
        )

        # Advance to window start
        await self.advance_to_time(window_start)

        # Simulate hashrate data for this window (progressive data arrival)
        await self.simulate_hashrate_for_window(epoch, window)

        # Process any user events at window start (block 0)
        events_at_start = self.scenario.get_events_at(window_start)
        for event in events_at_start:
            await self.process_event(event)

        # Auto-inject: miners compute auction results for current window
        for miner_name in self.miners:
            self.miners[miner_name].send_command("COMPUTE_CURRENT_AUCTION")

        # Process user events during window (blocks 1 to end)
        for block in range(1, window_duration):
            block_time = TimeAddress(epoch, window, block)
            events_at_block = self.scenario.get_events_at(block_time)
            if events_at_block:
                await self.advance_to_time(block_time)
                for event in events_at_block:
                    await self.process_event(event)

        # Advance to next window/epoch boundary
        if window == 5:
            # Last window: advance to epoch boundary (next epoch window 0)
            next_time = TimeAddress(epoch + 1, 0, 0)
        else:
            # Advance to next window
            next_time = TimeAddress(epoch, window + 1, 0)

        await self.advance_to_time(next_time)

        # Auto-inject: validators process auction for completed window
        for validator_name in self.validators:
            self.validators[validator_name].send_command("PROCESS_AUCTION")

        # Auto-inject: validators scrape prices and publish commitments (including ban bitmaps)
        for validator_name in self.validators:
            # Use stored budget if available
            budget_ph = self.validator_budgets.get(validator_name)
            if budget_ph is not None:
                alpha_tao = compute_alpha_tao_for_budget(
                    budget_ph,
                    mechanism_share=self.mechanism_1_share,
                )
                alpha_tao_fp18 = alpha_tao_to_fp18(alpha_tao)
                self.validators[validator_name].send_command(
                    "SCRAPE_PRICES",
                    {
                        "ALPHA_TAO": alpha_tao_fp18,
                    },
                )
            else:
                # Use default prices if no budget was set
                self.validators[validator_name].send_command("SCRAPE_PRICES")

            self.validators[validator_name].send_command("PUBLISH_LOCAL_COMMITMENT")

        logger.info(
            "Completed window",
            epoch=epoch,
            window=window,
            current_head=self.current_head,
        )

    async def _process_initialization(self) -> None:
        """Process all initialization events (epoch < 0)."""
        init_events = [e for e in self.scenario.sorted_events() if e.time.epoch < 0]

        # Find earliest initialization event and start blockchain there
        starting_time = None
        if init_events:
            first_init_event = min(init_events, key=lambda e: (e.time.epoch, e.time.window, e.time.block))
            starting_time = first_init_event.time

        # Initialize blockchain state
        await self.setup_blockchain_state(starting_time=starting_time)

        # Process all initialization events, advancing blockchain as needed
        for event in init_events:
            event_block = self.time_to_block(event.time)
            if event_block > self.current_head:
                await self.advance_to_block(event_block)
            await self.process_event(event)

        logger.info("Initialization complete", validators=len(self.validators), miners=len(self.miners))

    async def _process_finalization(self) -> None:
        """Process all finalization events (after last epoch completes).

        Events scheduled after the main epoch loop (e.g., final assertions)
        are processed here.
        """
        # Last epoch is num_epochs - 1
        last_epoch = self.scenario.num_epochs - 1

        # Get events that occur after the last window of the last epoch
        # Last window ends at TimeAddress(last_epoch, 5, 60)
        # We want events at (last_epoch, 5, 61+) or (last_epoch+1, ...)
        finalization_events = []
        for event in self.scenario.sorted_events():
            # Skip initialization events
            if event.time.epoch < 0:
                continue
            # Skip events in the main loop
            if event.time.epoch < last_epoch:
                continue
            if event.time.epoch == last_epoch and event.time.window < 5:
                continue
            if event.time.epoch == last_epoch and event.time.window == 5 and event.time.block <= 60:
                continue
            # This is a finalization event
            finalization_events.append(event)

        if not finalization_events:
            return

        logger.info(
            "Processing finalization events",
            count=len(finalization_events),
            events=[str(e.time) for e in finalization_events],
        )

        # Process all finalization events, advancing blockchain as needed
        for event in finalization_events:
            event_block = self.time_to_block(event.time)
            if event_block > self.current_head:
                await self.advance_to_block(event_block)
            await self.process_event(event)

        logger.info("Finalization complete", events_processed=len(finalization_events))

    async def run(self) -> None:
        """Execute full scenario across all epochs."""
        await self._process_initialization()

        # Process each validation epoch
        for epoch in range(self.scenario.num_epochs):
            logger.info("Starting validation epoch", epoch=epoch)

            # Process each window (0-5)
            for window in range(6):
                await self.process_window(epoch, window)

            # Calculate and commit weights after all windows
            logger.info("Calculating weights for epoch", epoch=epoch)

            # Track commit block for each validator
            if epoch not in self.weight_commit_blocks:
                self.weight_commit_blocks[epoch] = {}

            for validator_name in self.validators:
                self.validators[validator_name].send_command("CALCULATE_AUCTION_WEIGHTS")
                self.validators[validator_name].send_command("SET_AUCTION_WEIGHTS")
                # Record the block at which this validator committed weights
                self.weight_commit_blocks[epoch][validator_name] = self.current_head

        # Process any finalization events (e.g., final assertions)
        await self._process_finalization()

        logger.info("Scenario execution complete", epochs=self.scenario.num_epochs)

    def shutdown(self) -> None:
        """Stop all worker processes."""
        for validator in self.validators.values():
            validator.stop()
        for miner in self.miners.values():
            miner.stop()

        # Stop replaced/inactive miners
        if self.inactive_miners:
            logger.info("Stopping inactive miners", count=len(self.inactive_miners))
            for name, inactive_miner in self.inactive_miners.items():
                logger.debug("Stopping inactive miner", name=name)
                inactive_miner.stop()

        for path in self._temp_dirs:
            shutil.rmtree(path, ignore_errors=True)
