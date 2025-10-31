#!/usr/bin/env python3
"""Manual test: Real APScheduler miner connected to simulated chain.

The miner runs the actual production code (aps_miner) with:
- Real bittensor wallet
- Real TOML config
- Real APScheduler with our HardTimeoutExecutor
- Real commitment publishing
- Real auction computation

But it's connected to the simulator, not real subtensor.
The miner is "in the Matrix" - it doesn't know it's simulated!

The simulator:
- Advances blocks every ~12s
- Puts fake validator price commitments on chain
- Records miner's bidding commitments
- Miner reads commitments and thinks it's real subtensor

Usage:
    python manual_test_real_miner.py
"""

import asyncio
import logging
import multiprocessing as mp
import os
import secrets
import signal
import sys
import tempfile
from pathlib import Path

import bittensor_wallet
import httpx
import structlog

# Simulator config
SIM_HOST = "127.0.0.1"
SIM_HTTP_PORT = 8090
SIM_RPC_PORT = 9944
NETUID = 1

# Timing
BLOCK_TIME = 12  # seconds
BLOCKS_PER_WINDOW = 60

# Budget config (in PH - Petahash)
TARGET_BUDGET_PH = 40.0

RNG = secrets.SystemRandom()

logger = structlog.get_logger(__name__)


def simulator_process_main(ready_event: mp.Event, stop_event: mp.Event):
    """Run simulator in separate process."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infinite_hashes.settings")

    from infinite_hashes.testutils.integration.worker_mains import simulator_process

    simulator_process(ready_event, stop_event)


def create_test_wallet(wallet_dir: str, name: str = "test_miner") -> tuple[str, str]:
    """Create a test wallet and return (hotkey, coldkey) SS58 addresses."""
    import contextlib
    import io

    # Suppress wallet creation output
    with contextlib.redirect_stdout(io.StringIO()):
        wallet = bittensor_wallet.Wallet(name=name, hotkey="default", path=wallet_dir)
        wallet.create_new_coldkey(n_words=12, use_password=False, overwrite=True)
        wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True)

    # Reload to get addresses
    wallet_reloaded = bittensor_wallet.Wallet(name=name, hotkey="default", path=wallet_dir)
    hotkey = wallet_reloaded.hotkey.ss58_address
    coldkey = wallet_reloaded.coldkey.ss58_address

    return hotkey, coldkey


def create_miner_config(config_path: Path, wallet_dir: str) -> None:
    """Create miner TOML config file."""
    config_content = f"""# Manual test miner configuration

[bittensor]
network = "ws://{SIM_HOST}:{SIM_RPC_PORT}"
netuid = {NETUID}

[wallet]
name = "test_miner"
hotkey_name = "default"
directory = "{wallet_dir}"

[workers]
# Single price multiplier for all workers
price_multiplier = "1.0"

# List of worker hashrates (in PH - Petahash)
# For testing: 3-5 workers with 2-10 PH each, totaling ~30-50 PH
hashrates = [
    "5.5",
    "8.2",
    "6.8",
    "9.1",
    "7.3"
]
"""
    config_path.write_text(config_content)


async def setup_blockchain(client: httpx.AsyncClient, miner_hotkey: str, validator_hotkey: str):
    """Initialize blockchain state and register miner and validator."""
    # Set tempo
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}",
        json={"tempo": 360},
    )
    response.raise_for_status()

    # Set initial head 5 blocks before new window starts
    # Window boundaries are at multiples of 60: 1020, 1080, 1140, etc.
    # Starting at 1075 means next window starts at 1080 (in 5 blocks)
    start_block = 1075
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/head",
        json={
            "number": start_block,
            "timestamp": "2025-01-15T10:00:00Z",
        },
    )
    response.raise_for_status()

    # Register miner and validator neurons
    neurons = [
        {
            "hotkey": validator_hotkey,
            "uid": 0,
            "stake": 10000.0,  # High stake for validator
            "coldkey": validator_hotkey,
        },
        {
            "hotkey": miner_hotkey,
            "uid": 1,
            "stake": 1000.0,
            "coldkey": miner_hotkey,
        },
    ]

    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}/neurons",
        json={"neurons": neurons},
    )
    response.raise_for_status()

    print(f"✓ Blockchain initialized at block {start_block}")
    print("  Next window starts at block 1080 (in 5 blocks)")
    print(f"✓ Validator registered: UID 0, hotkey {validator_hotkey[:16]}...")
    print(f"✓ Miner registered: UID 1, hotkey {miner_hotkey[:16]}...")


async def submit_validator_commitment(wallet: bittensor_wallet.Wallet, budget_ph: float):
    """Submit validator price commitment using real turbobt mechanism.

    This uses the actual commitment publishing like validators do in production.
    Must include all three required prices: ALPHA_TAO, TAO_USDC, HASHP_USDC.
    """
    import turbobt

    from infinite_hashes.consensus.price import PriceCommitment
    from infinite_hashes.testutils.integration.budget_helper import (
        DEFAULT_MECHANISM_1_SHARE,
        alpha_tao_to_fp18,
        compute_alpha_tao_for_budget,
    )

    # Use the same default prices as compute_alpha_tao_for_budget
    # These MUST match for the budget computation to work correctly
    TAO_USDC = 45.0
    HASHP_USDC = 50.0

    # Compute ALPHA_TAO price for target budget using those prices
    alpha_tao = compute_alpha_tao_for_budget(
        budget_ph,
        tao_usdc=TAO_USDC,
        hashp_usdc=HASHP_USDC,
        mechanism_share=DEFAULT_MECHANISM_1_SHARE,
    )
    alpha_tao_fp18 = alpha_tao_to_fp18(alpha_tao)

    # Convert to FP18
    tao_usdc_fp18 = int(TAO_USDC * (10**18))
    hashp_usdc_fp18 = int(HASHP_USDC * (10**18))

    # Create proper PriceCommitment with all three required prices
    commitment = PriceCommitment(
        t="p",
        prices={
            "ALPHA_TAO": alpha_tao_fp18,
            "TAO_USDC": tao_usdc_fp18,
            "HASHP_USDC": hashp_usdc_fp18,
        },
        v=1,
    )

    # Serialize to compact format
    commitment_str = commitment.to_compact()
    payload = commitment_str.encode("utf-8")

    # Publish using turbobt like real validators do
    async with turbobt.Bittensor(f"ws://{SIM_HOST}:{SIM_RPC_PORT}") as bittensor:
        subnet = bittensor.subnet(NETUID)
        extrinsic = await subnet.commitments.set(
            data=payload,
            wallet=wallet,
        )
        await extrinsic.wait_for_finalization()
        print(f"  ✓ Validator commitment published: Budget target = {budget_ph:.1f} PH")


async def advance_block(client: httpx.AsyncClient) -> dict:
    """Advance blockchain by 1 block."""
    response = await client.post(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/blocks/advance",
        json={"steps": 1, "step_seconds": BLOCK_TIME},
    )
    response.raise_for_status()
    return response.json()["head"]


async def get_head(client: httpx.AsyncClient) -> dict:
    """Get current head block."""
    response = await client.get(f"http://{SIM_HOST}:{SIM_HTTP_PORT}/head")
    response.raise_for_status()
    return response.json()


async def get_commitments(client: httpx.AsyncClient, block_hash: str) -> dict:
    """Get all commitments at a block."""
    response = await client.get(
        f"http://{SIM_HOST}:{SIM_HTTP_PORT}/subnets/{NETUID}/commitments",
        params={"block_hash": block_hash},
    )
    response.raise_for_status()
    return response.json()


def start_real_miner(config_path: Path) -> mp.Process:
    """Start the real APScheduler miner in a subprocess."""

    def miner_main():
        # Configure structlog for console output
        import structlog

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

        # Run the actual miner entry point
        os.environ["BITTENSOR_NETWORK"] = f"ws://{SIM_HOST}:{SIM_RPC_PORT}"
        os.environ["BITTENSOR_NETUID"] = str(NETUID)

        # Import and run
        sys.path.insert(0, str(Path(__file__).parent / "app" / "src"))
        from infinite_hashes.aps_miner.config import MinerConfig
        from infinite_hashes.aps_miner.scheduler import run_scheduler

        # Load config and run
        MinerConfig.load(config_path)
        run_scheduler(str(config_path))

    process = mp.Process(target=miner_main, daemon=False)
    process.start()
    return process


async def main():
    """Run manual test."""
    print("=" * 70)
    print("InfiniteHash Real Miner Test (Miner in the Matrix!)")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  • Simulator: ws://{SIM_HOST}:{SIM_RPC_PORT}")
    print(f"  • Block time: {BLOCK_TIME}s")
    print(f"  • Blocks per window: {BLOCKS_PER_WINDOW}")
    print(f"  • Fake validator budget: ~{TARGET_BUDGET_PH} PH")
    print()

    # Setup test directory
    test_dir = Path(tempfile.gettempdir()) / "infinite_hash_test"
    test_dir.mkdir(exist_ok=True)

    wallet_dir = str(test_dir / "wallets")
    config_path = test_dir / "miner_config.toml"

    print("1. Creating test wallets...")
    os.makedirs(wallet_dir, exist_ok=True)

    # Create miner wallet
    miner_hotkey, miner_coldkey = create_test_wallet(wallet_dir, name="test_miner")
    print("✓ Miner wallet created")
    print(f"  Hotkey:  {miner_hotkey}")
    print(f"  Coldkey: {miner_coldkey}")

    # Create validator wallet
    validator_hotkey, validator_coldkey = create_test_wallet(wallet_dir, name="test_validator")
    print("✓ Validator wallet created")
    print(f"  Hotkey:  {validator_hotkey}")
    print(f"  Coldkey: {validator_coldkey}")

    # Load validator wallet for commitment submission
    validator_wallet = bittensor_wallet.Wallet(name="test_validator", hotkey="default", path=wallet_dir)

    print("\n2. Creating miner config...")
    create_miner_config(config_path, wallet_dir)
    print(f"✓ Config created: {config_path}")

    print("\n3. Starting simulator...")
    ready_event = mp.Event()
    stop_event = mp.Event()

    sim_process = mp.Process(
        target=simulator_process_main,
        args=(ready_event, stop_event),
        daemon=True,
    )
    sim_process.start()

    if not ready_event.wait(timeout=10):
        print("✗ Simulator failed to start")
        sim_process.terminate()
        return 1

    print("✓ Simulator started")

    client = httpx.AsyncClient(timeout=10.0)
    miner_process = None

    try:
        print("\n4. Setting up blockchain...")
        await setup_blockchain(client, miner_hotkey, validator_hotkey)

        print("\n5. Starting REAL miner (APScheduler)...")
        miner_process = start_real_miner(config_path)
        await asyncio.sleep(2)  # Give miner time to start

        if miner_process.is_alive():
            print("✓ Miner started (running actual aps_miner code)")
            print(f"  PID: {miner_process.pid}")
            print(f"  Config: {config_path}")
        else:
            print("✗ Miner failed to start")
            return 1

        print("\n6. Running simulation...")
        print("   Press Ctrl+C to stop\n")
        print("=" * 70)

        # Get initial state
        head = await get_head(client)
        current_block = head["number"]
        last_window = current_block // BLOCKS_PER_WINDOW

        # Submit initial validator commitment
        budget = TARGET_BUDGET_PH + RNG.uniform(-5, 5)
        print(f"\n💰 Block {current_block}: Validator submitting budget = {budget:.1f} PH")
        await submit_validator_commitment(validator_wallet, budget)

        # Verify validator commitment was stored
        commits = await get_commitments(client, head["hash"])
        print(f"  Verified: {len(commits.get('entries', {}))} commitment(s) on chain")

        # Main simulation loop
        while True:
            await asyncio.sleep(BLOCK_TIME)

            # Advance block
            head = await advance_block(client)
            current_block = head["number"]
            current_window = current_block // BLOCKS_PER_WINDOW
            block_in_window = current_block % BLOCKS_PER_WINDOW

            print(f"\n⏰ Block {current_block} (Window {current_window}, Block {block_in_window}/{BLOCKS_PER_WINDOW})")

            # Check commitments
            try:
                commits = await get_commitments(client, head["hash"])
                entries = commits.get("entries", {})

                if entries:
                    print(f"  📝 Commitments on chain: {len(entries)}")
                    for hotkey, payload_hex in entries.items():
                        short_hotkey = hotkey[:16]
                        payload_bytes = bytes.fromhex(payload_hex.removeprefix("0x"))

                        # Parse commitment to show type and content
                        from infinite_hashes.consensus.bidding import BiddingCommitment
                        from infinite_hashes.consensus.parser import parse_commitment
                        from infinite_hashes.consensus.price import PriceCommitment

                        commitment = parse_commitment(payload_bytes)
                        if isinstance(commitment, PriceCommitment):
                            # Display price commitment
                            prices_display = []
                            for key in ["ALPHA_TAO", "TAO_USDC", "HASHP_USDC"]:
                                if key in commitment.prices:
                                    val = commitment.prices[key] / (10**18)
                                    prices_display.append(f"{key}={val:.6f}")

                            # Compute budget using the same formula as the auction algorithm
                            budget_str = ""
                            if all(k in commitment.prices for k in ["ALPHA_TAO", "TAO_USDC", "HASHP_USDC"]):
                                alpha_tao = commitment.prices["ALPHA_TAO"] / (10**18)
                                tao_usdc = commitment.prices["TAO_USDC"] / (10**18)
                                hashp_usdc = commitment.prices["HASHP_USDC"] / (10**18)

                                # Same calculation as _compute_budget_ph in bidding.py
                                alpha_usdc = alpha_tao * tao_usdc
                                blocks_per_day = 7200
                                miners_share = 0.41
                                daily_alpha = miners_share * blocks_per_day
                                daily_usdc = alpha_usdc * daily_alpha
                                budget_ph = daily_usdc / hashp_usdc if hashp_usdc > 0 else 0

                                budget_str = f" → Budget: {budget_ph:.1f} PH"

                            print(
                                f"     {short_hotkey}... [PRICE]: {', '.join(prices_display) if prices_display else 'empty'}{budget_str}"
                            )
                        elif isinstance(commitment, BiddingCommitment):
                            # Display bidding commitment
                            bids = commitment.bids or []
                            hashrates = [bid[0] for bid in bids]
                            total_ph = sum(float(hr) for hr in hashrates)
                            print(f"     {short_hotkey}... [BID]: {len(bids)} workers, {total_ph:.1f} PH total")
                        else:
                            # Unknown type
                            payload_str = payload_bytes.decode("utf-8", errors="ignore")[:50]
                            print(f"     {short_hotkey}...: {payload_str}...")
                else:
                    print("  📝 No commitments on chain")
            except Exception as e:
                print(f"  ⚠️  Could not fetch commitments: {e}")

            # Check if new window started
            if current_window > last_window:
                print("  🎯 New window started!")
                last_window = current_window

                # Change validator budget
                budget = TARGET_BUDGET_PH + RNG.uniform(-10, 10)
                print(f"  💰 Validator submitting new budget = {budget:.1f} PH")
                await submit_validator_commitment(validator_wallet, budget)

            # Check if miner is still alive
            if not miner_process.is_alive():
                print("\n✗ Miner process died!")
                break

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("✓ Test stopped by user")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        print("\nCleaning up...")

        # Stop miner
        if miner_process and miner_process.is_alive():
            print("  Stopping miner...")
            miner_process.terminate()
            miner_process.join(timeout=5)
            if miner_process.is_alive():
                miner_process.kill()
            print("  ✓ Miner stopped")

        # Stop simulator
        stop_event.set()
        sim_process.terminate()
        sim_process.join(timeout=2)
        print("  ✓ Simulator stopped")

        await client.aclose()
        print("✓ Cleanup complete")

    return 0


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    sys.exit(asyncio.run(main()))
