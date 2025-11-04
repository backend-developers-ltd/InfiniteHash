import asyncio
import collections
import datetime
import enum
import logging
from typing import Any, NamedTuple, TypeAlias

import bittensor_wallet
import httpx
import structlog
import tenacity
import turbobt
import turbobt.substrate.exceptions
import websockets
from asgiref.sync import async_to_sync
from celery import Task
from celery.utils.log import get_task_logger
from django.conf import settings
from django.db import transaction

from infinite_hashes.auctions.mechanism_split import fetch_mechanism_share_fraction
from infinite_hashes.celery import app
from infinite_hashes.utils import run_async
from infinite_hashes.validator import auction_processing as auct
from infinite_hashes.validator.hashrates import get_hashrates_async
from infinite_hashes.validator.locks import Locked, LockType, get_advisory_lock
from infinite_hashes.validator.models import AuctionResult, BannedMiner, LuxorSnapshot, WeightsBatch

WEIGHT_SETTING_ATTEMPTS = 100
WEIGHT_SETTING_FAILURE_BACKOFF = 5

Hashrates: TypeAlias = dict[str, list[int]]


class Epoch(NamedTuple):
    block: int
    timestamp: datetime.datetime


class TickSize(enum.Enum):
    HOUR = "1h", datetime.timedelta(hours=1)
    DAY = "1d", datetime.timedelta(days=1)

    def __init__(self, label, timedelta):
        self.label = label
        self.timedelta = timedelta


logger = structlog.wrap_logger(get_task_logger(__name__))


async def commit_mechanism_weights(
    bittensor: turbobt.Bittensor,
    netuid: int,
    mechanism_id: int,
    weights: dict[int, float],
    version_key: int = 0,
    block_time: int = 12,
) -> int:
    """Commit weights for a specific mechanism using commit/reveal scheme.

    This replicates turbobt's subnet.weights.commit() but for any mechanism ID.

    Args:
        bittensor: Bittensor client instance
        netuid: Subnet UID
        mechanism_id: Mechanism ID (0 for default, 1 for auctions, etc.)
        weights: Dictionary mapping UID to weight (float)
        version_key: Weights version key (default: 0)
        block_time: Block time in seconds (default: 12)

    Returns:
        Reveal round block number when weights can be revealed

    Raises:
        ValueError: If weights cannot be normalized
    """
    import bittensor_drand

    # Normalize weights to u16 range (same as turbobt does)
    weights_u16 = normalize_weights_to_u16(weights)

    try:
        uids, weight_values = zip(*weights_u16.items())
    except ValueError:
        # Empty weights
        uids, weight_values = [], []

    # Get current block and subnet hyperparameters
    async with bittensor.blocks[-1] as block:
        subnet = await bittensor.subnet(netuid).get()
        hyperparameters = await subnet.get_hyperparameters()

        # Generate encrypted commit and calculate reveal round
        # Note: This calls external Drand network for time-lock encryption
        commit, reveal_round = bittensor_drand.get_encrypted_commit(
            uids=list(uids),
            weights=list(weight_values),
            version_key=version_key,
            tempo=hyperparameters["tempo"],
            current_block=block.number,
            netuid=netuid,
            subnet_reveal_period_epochs=hyperparameters["commit_reveal_period"],
            block_time=block_time,
            hotkey=bittensor.wallet.hotkey.public_key,
        )

        logger.info(
            "Generated weight commit for mechanism",
            mechanism_id=mechanism_id,
            block_number=block.number,
            reveal_round=reveal_round,
            num_weights=len(weights_u16),
        )

        # Submit commit extrinsic
        result = await bittensor.subtensor.subtensor_module.commit_timelocked_mechanism_weights(
            netuid=netuid,
            commit=commit,
            mechanism_id=mechanism_id,
            reveal_round=reveal_round,
            commit_reveal_version=4,  # CRV3
            wallet=bittensor.wallet,
        )

        logger.info(
            "Committed weights for mechanism",
            mechanism_id=mechanism_id,
            reveal_round=reveal_round,
            extrinsic_hash=result.extrinsic_hash if hasattr(result, "extrinsic_hash") else None,
        )

    return reveal_round


def normalize_weights_to_u16(weights_by_uid: dict[int, float]) -> dict[int, int]:
    """Normalize weights to u16 range (0-65535) for on-chain submission.

    Uses max-based normalization (same as turbobt):
    - Divides each weight by the maximum weight
    - Scales to u16 range by multiplying by 65535
    - Max weight becomes 65535, others proportionally smaller

    Args:
        weights_by_uid: Dictionary mapping UID to weight (float)

    Returns:
        Dictionary mapping UID to u16 weight (int), max weight = 65535

    Raises:
        ValueError: If max weight is 0 or weights are empty
    """
    if not weights_by_uid:
        raise ValueError("Cannot normalize empty weights")

    max_weight = max(weights_by_uid.values())
    if max_weight <= 0:
        raise ValueError(f"Cannot normalize weights with max {max_weight}")

    U16_MAX = 65535
    weights_u16 = {uid: round((weight / max_weight) * U16_MAX) for uid, weight in weights_by_uid.items()}

    return weights_u16


def send_to_dead_letter_queue(task: Task, exc, task_id, args, kwargs, einfo):
    """Hook to put a task into dead letter queue when it fails."""
    if task.app.conf.task_always_eager:
        return  # do not run failed task again in eager mode

    logger.warning(
        "Sending failed task to dead letter queue",
        task=task,
        exc=exc,
        task_id=task_id,
        args=args,
        kwargs=kwargs,
        einfo=einfo,
    )
    task.apply_async(args=args, kwargs=kwargs, queue="dead_letter")


@app.task(on_failure=send_to_dead_letter_queue)
def demo_task(x, y):
    logger.info("adding two numbers", x=x, y=y)
    return x + y


async def calculate_weights_async(
    window: datetime.timedelta = None,
):
    """Calculate weights for mechanism 0 based on hashrates."""
    # Determine scoring window based on cutoff date
    if window is None:
        current_date = datetime.date.today()

        if current_date >= datetime.date(2025, 8, 25):
            window = datetime.timedelta(hours=6)
        else:
            window = datetime.timedelta(days=1)

    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )
        epoch = subnet.epoch(block.number)

        batches = WeightsBatch.objects.filter(
            epoch_start=epoch.start,
            mechanism_id=0,
        )

        if await batches.acount():
            ids = [str(b.id) async for b in batches]
            logger.debug("Already have weights for this epoch: %s", ",".join(ids))
            return

        subnet_tempo = subnet.tempo + 1
        validation_start_block_number = epoch.start + int(settings.VALIDATION_OFFSET * subnet_tempo)
        blocks_left = validation_start_block_number - block.number

        if blocks_left > 0:
            logger.debug(
                "Too early to calculate weights. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
                epoch.start,
                block.number,
                round((block.number - epoch.start) / subnet_tempo, 2),
                subnet_tempo,
            )
            return

        if abs(blocks_left) >= settings.VALIDATION_THRESHOLD * subnet_tempo:
            logger.error(
                "Too late to calculate weights. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
                epoch.start,
                block.number,
                round((block.number - epoch.start) / subnet_tempo, 2),
                subnet_tempo,
            )
            return

        validation_start_block = await bittensor.block(validation_start_block_number).get()
        validation_start_datetime = await validation_start_block.get_timestamp()

        hashrates = await get_hashrates_async(
            settings.LUXOR_SUBACCOUNT_NAME,
            start=validation_start_datetime - window,
            end=validation_start_datetime,
        )
        weights = {hotkey: sum(hotkey_hashrates) for hotkey, hotkey_hashrates in hashrates.items()}

        batch = await WeightsBatch.objects.acreate(
            block=validation_start_block.number,
            epoch_start=epoch.start,
            mechanism_id=0,
            weights=weights,
            scored=False,
            should_be_scored=True,
        )

        logger.debug(
            "Weight batch saved: %s. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
            batch.id,
            epoch.start,
            block.number,
            round((block.number - epoch.start) / subnet_tempo, 2),
            subnet_tempo,
        )

        return weights


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def calculate_weights(event_loop: Any = None):
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.VALIDATION_SCHEDULING)
        except Locked:
            logger.debug("Another thread already scheduling validation")
            return

        return run_async(calculate_weights_async, event_loop=event_loop)


async def set_weights_async() -> bool:
    batches = [
        batch
        async for batch in WeightsBatch.objects.filter(
            mechanism_id=0,
            scored=False,
            should_be_scored=True,
        )
    ]

    if not batches:
        logger.debug("No batches to score")
        return False

    async with turbobt.Bittensor(
        settings.BITTENSOR_NETWORK,
        wallet=bittensor_wallet.Wallet(
            name=settings.BITTENSOR_WALLET_NAME,
            hotkey=settings.BITTENSOR_WALLET_HOTKEY_NAME,
            path=str(settings.BITTENSOR_WALLET_DIRECTORY),
        ),
    ) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )
        epoch = subnet.epoch(block.number)

        expired_batches = [batch for batch in batches if batch.epoch_start < epoch.start]

        if expired_batches:
            for batch in expired_batches:
                batch.should_be_scored = False

            await WeightsBatch.objects.abulk_update(
                expired_batches,
                fields=("should_be_scored",),
            )

            logger.error(
                "Expired batches: [%s]",
                ", ".join(str(batch.id) for batch in batches),
            )
            return False

        if len(batches) > 1:
            logger.error("Unexpected number batches eligible for scoring: %s", len(batches))

            for batch in batches[:-1]:
                batch.scored = True

            batches = [batches[-1]]

        logger.info(
            "Selected batches for scoring: [%s]",
            ", ".join(str(batch.id) for batch in batches),
        )

        # Query neurons at current block, not historical validation block
        # (We can't go back in time for neuron state)
        neurons = {neuron.hotkey: neuron for neuron in await subnet.list_neurons()}

        weights_by_uid = collections.defaultdict[int, float](float)

        for batch in batches:
            for hotkey, hotkey_weight in batch.weights.items():
                try:
                    neuron = neurons[hotkey]
                except KeyError:
                    continue

                weights_by_uid[neuron.uid] += hotkey_weight

            batch.scored = True

        if not weights_by_uid:
            return False

        # Note: subnet.weights.commit() handles u16 normalization and commit/reveal internally
        # This is turbobt's high-level API for mechanism 0 (backward compatibility)
        # For mechanism 1, we use commit_mechanism_weights() which replicates this behavior
        async for attempt in tenacity.AsyncRetrying(
            before_sleep=tenacity.before_sleep_log(logger, logging.ERROR),
            stop=tenacity.stop_after_attempt(WEIGHT_SETTING_ATTEMPTS),
            wait=tenacity.wait_fixed(WEIGHT_SETTING_FAILURE_BACKOFF),
        ):
            with attempt:
                await subnet.weights.commit(weights_by_uid)

        await WeightsBatch.objects.abulk_update(
            batches,
            fields=("scored",),
        )

        return True


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def set_weights(*, event_loop: Any = None) -> bool:
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.WEIGHT_SETTING)
        except Locked:
            logger.debug("Another thread already scheduling validation")
            return False

        return run_async(set_weights_async, event_loop=event_loop)


# --- Auction scheduling ---


# Deprecated placeholder selector removed; using ILP-based selection in consensus.bidding


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def process_auctions(*, event_loop: Any = None) -> int:
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.AUCTION_SCHEDULING)
        except Locked:
            logger.debug("Another thread already processing auctions")
            return 0

        return run_async(auct.process_auctions_async, event_loop=event_loop)


# --- Auction weights (mechanism 1) ---


async def get_burn_uid(bittensor: turbobt.Bittensor, netuid: int, neurons: list | None = None) -> int | None:
    """Get the UID used to receive the burn allocation (subnet owner's earliest neuron)."""

    def _coerce_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return "0x" + value.hex()
        return str(value)

    def _neuron_attr(neuron: Any, *names: str) -> Any:
        for name in names:
            if hasattr(neuron, name):
                value = getattr(neuron, name)
                if value is not None:
                    return value
            if isinstance(neuron, dict) and neuron.get(name) is not None:
                return neuron[name]
        extras = getattr(neuron, "extras", None)
        if isinstance(extras, dict):
            for name in names:
                if extras.get(name) is not None:
                    return extras[name]
        return None

    async def _query_owner_uid(hotkey: str | None) -> int | None:
        if not hotkey:
            return None
        try:
            uids = getattr(bittensor.subtensor.subtensor_module, "Uids", None)
            if uids and hasattr(uids, "get"):
                owner_uid = await uids.get(netuid, hotkey)
                if owner_uid is not None:
                    try:
                        owner_uid = int(owner_uid)
                    except (TypeError, ValueError):
                        pass
                logger.info("SN Owner UID", owner_uid=owner_uid)
                return owner_uid
        except Exception as exc:
            logger.warning("Could not get UID for owner hotkey", error=str(exc), owner_hotkey=hotkey)
        return None

    try:
        subnet = await bittensor.subnet(netuid).get()
    except Exception as exc:
        logger.error("Error retrieving subnet info", error=str(exc), netuid=netuid)
        subnet = None

    owner_coldkey = None
    owner_hotkey = None

    if subnet is not None:
        owner_coldkey = _coerce_str(getattr(subnet, "owner_coldkey", None) or getattr(subnet, "owner_ss58", None))
        owner_hotkey = _coerce_str(getattr(subnet, "owner_hotkey", None) or getattr(subnet, "owner_hotkey_ss58", None))

    if neurons is None:
        try:
            subnet_obj = bittensor.subnet(netuid)
            neurons = await subnet_obj.list_neurons()
        except Exception as exc:
            logger.warning("Failed fetching neuron list", error=str(exc), netuid=netuid)
            neurons = []

    sn_owner_uid: int | None = None

    if owner_coldkey is None:
        logger.warning("Owner coldkey missing, attempting fallback via owner hotkey lookup", netuid=netuid)
        if owner_hotkey is None:
            logger.warning("Owner hotkey unavailable, cannot determine burn UID via fallback", netuid=netuid)
            return None

        owner_hotkey_str = _coerce_str(owner_hotkey)
        sn_owner_uid = await _query_owner_uid(owner_hotkey_str)

        owner_neuron = None
        for neuron in neurons:
            neuron_hotkey = _coerce_str(_neuron_attr(neuron, "hotkey", "hotkey_ss58"))
            if neuron_hotkey == owner_hotkey_str:
                owner_neuron = neuron
                break

        if owner_neuron is None:
            logger.warning(
                "Owner neuron not found in neuron list, falling back to owner UID", owner_hotkey=owner_hotkey_str
            )
            return sn_owner_uid

        owner_coldkey = _coerce_str(_neuron_attr(owner_neuron, "coldkey", "coldkey_ss58"))
        if owner_coldkey is None:
            logger.warning("Owner coldkey missing on neuron, falling back to owner UID", owner_hotkey=owner_hotkey_str)
            return sn_owner_uid

    owner_neurons = [
        neuron for neuron in neurons if _coerce_str(_neuron_attr(neuron, "coldkey", "coldkey_ss58")) == owner_coldkey
    ]

    if not owner_neurons:
        logger.warning("No neurons found with owner coldkey, falling back to owner UID", owner_coldkey=owner_coldkey)
        if sn_owner_uid is None:
            sn_owner_uid = await _query_owner_uid(_coerce_str(owner_hotkey))
        return sn_owner_uid

    logger.info("Found owner neurons", count=len(owner_neurons), owner_coldkey=owner_coldkey)

    def _registration_block(neuron: Any) -> float:
        value = _neuron_attr(neuron, "registration_block")
        if value is None:
            return float("inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    burn_candidate = min(owner_neurons, key=_registration_block)

    burn_uid = _neuron_attr(burn_candidate, "uid")
    if burn_uid is None:
        logger.warning("Burn candidate missing UID, falling back to owner UID")
        if sn_owner_uid is None:
            sn_owner_uid = await _query_owner_uid(_coerce_str(owner_hotkey))
        return sn_owner_uid

    logger.info("Selected burn UID %s from owner coldkey %s", burn_uid, owner_coldkey)
    return int(burn_uid)


async def calculate_auction_weights_async():
    """Calculate weights for mechanism 1 based on auction winners with delivery.

    Only runs during window 0 of validation epoch[n+1].
    At this point, window[n][5] has finished, so all 6 windows of validation_epoch[n] are processed.
    Calculates and commits weights for the PREVIOUS validation epoch.
    """
    from infinite_hashes.auctions import utils as auction_utils

    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )
        current_epoch = subnet.epoch(block.number)
        blocks_per_window = auction_utils.blocks_per_window_default()
        subnet_tempo = subnet.tempo + 1

        # Check if we're in window 0
        window_number, (window_start, window_end) = auction_utils.current_validation_window(
            block.number, current_epoch.start, blocks_per_window
        )

        # Determine which subnet epoch's validation we should calculate weights for
        if window_number == 0:
            # Window 0 starts at a new validation epoch boundary
            # Calculate which subnet epoch this window belongs to
            validation_epoch_start_for_window = window_start
            subnet_epoch_for_window = validation_epoch_start_for_window + blocks_per_window

            # Calculate weights for the PREVIOUS subnet epoch (relative to the window we're in)
            previous_subnet_epoch_start = subnet_epoch_for_window - subnet_tempo
        else:
            # Not in window 0, skip weight calculation
            logger.debug(
                "Not in window 0, skipping auction weight calculation",
                current_window=window_number,
                block_number=block.number,
            )
            return None

        # Check if we already created a weights batch for this validation epoch
        existing_batch = await asyncio.to_thread(
            lambda: WeightsBatch.objects.filter(
                epoch_start=previous_subnet_epoch_start,
                mechanism_id=1,
            ).first()
        )

        if existing_batch:
            logger.debug(
                "Auction weights batch already exists for validation epoch %s (block %s)",
                previous_subnet_epoch_start,
                existing_batch.block,
            )
            return None

        # Get all 6 windows for the previous validation epoch
        previous_windows = auction_utils.validation_windows_for_subnet_epoch(
            previous_subnet_epoch_start, blocks_per_window
        )

        # Check that ALL 6 windows are processed
        def _check_all_windows_processed():
            results = []
            for window_num, (start, end) in enumerate(previous_windows, start=0):
                result = AuctionResult.objects.filter(
                    epoch_start=previous_subnet_epoch_start,
                    start_block=start,
                    end_block=end,
                ).first()
                if result:
                    results.append((window_num, result))
            return results

        processed_windows = await asyncio.to_thread(_check_all_windows_processed)

        if len(processed_windows) < 6:
            logger.debug(
                "Not all windows processed for previous validation epoch, skipping weight calculation",
                previous_epoch=previous_subnet_epoch_start,
                processed_windows_count=len(processed_windows),
                required=6,
                processed=[w for w, _ in processed_windows],
            )
            return None

        # Calculate weights: ALPHA-based payment calculation
        # Note: Auction results only contain delivered winners (non-delivered are banned and filtered during auction processing)
        #
        # Algorithm:
        # 1. For each window: compute payment in ALPHA = delivered_hashrate * hashp_usdc * price_multiplier / alpha_usdc * window_fraction
        # 2. Track window budget and burn (both in ALPHA)
        # 3. Sum ALPHA payments across windows
        # 4. Normalize weights based on ALPHA payments (burn is excluded)
        #
        # Note: Budget is the amount of PH we can afford from daily ALPHA production.
        # Payments and burn are expressed in ALPHA to normalize across different price conditions.

        payments_alpha: dict[str, float] = {}  # hotkey -> total ALPHA payment across windows
        total_budget_alpha = 0.0
        total_spent_alpha = 0.0
        window_details = []

        logger.info(
            "Calculating ALPHA-based auction weights from windows",
            previous_epoch=previous_subnet_epoch_start,
            windows_to_process=len(processed_windows),
        )

        blocks_per_day = 7200
        base_miner_share = 0.41  # Base miner allocation per block (pre-mechanism split)
        mechanism_share_fraction = await fetch_mechanism_share_fraction(
            bittensor,
            settings.BITTENSOR_NETUID,
            mechanism_id=1,
        )
        if mechanism_share_fraction is None or mechanism_share_fraction <= 0:
            raise RuntimeError(
                f"Mechanism emission share must be positive; received {mechanism_share_fraction!r} for mechanism 1"
            )

        miner_share_per_block = base_miner_share * mechanism_share_fraction
        daily_alpha = blocks_per_day * miner_share_per_block

        logger.debug(
            "Mechanism emission share",
            mechanism_id=1,
            mechanism_share_fraction=mechanism_share_fraction,
            base_miner_share=base_miner_share,
            miner_share_per_block=miner_share_per_block,
            daily_alpha=daily_alpha,
        )

        for window_num, result in processed_windows:
            # Price consensus values are stored as Decimals
            hashp_usdc_decimal = result.hashp_usdc
            alpha_tao_decimal = result.alpha_tao
            tao_usdc_decimal = result.tao_usdc

            if not (hashp_usdc_decimal and alpha_tao_decimal and tao_usdc_decimal):
                logger.warning(
                    "Missing price consensus for window, skipping",
                    window=window_num,
                    epoch=previous_subnet_epoch_start,
                    has_hashp=bool(hashp_usdc_decimal),
                    has_alpha=bool(alpha_tao_decimal),
                    has_tao=bool(tao_usdc_decimal),
                )
                continue

            # Convert to float for calculations
            hashp_usdc = float(hashp_usdc_decimal)  # USDC per PH per day
            alpha_tao = float(alpha_tao_decimal)
            tao_usdc = float(tao_usdc_decimal)

            # Compute daily ALPHA budget and PH budget
            # Window duration in blocks (windows 0-4 are 60 blocks, window 5 is 61 blocks)
            window_blocks = 61 if window_num == 5 else 60
            window_alpha_budget = daily_alpha * (window_blocks / blocks_per_day)

            # ALPHA to USDC conversion
            alpha_usdc = alpha_tao * tao_usdc

            # Daily PH budget: How much continuous PH capacity we can afford for a full day
            # This is constant and represents the auction budget capacity
            daily_usdc_budget = daily_alpha * alpha_usdc
            daily_budget_ph = daily_usdc_budget / hashp_usdc if hashp_usdc > 0 else 0.0

            window_spent_alpha = 0.0

            for winner in result.winners:
                hotkey = winner.get("hotkey")
                hashrate = winner.get("hashrate", 0)
                price_multiplier_fp18 = winner.get("price", 10**18)  # Default to 1.0 if missing

                if not hotkey:
                    continue

                try:
                    # Convert to float
                    hashrate_ph = float(hashrate)
                    price_multiplier = float(price_multiplier_fp18) / (10**18)

                    # Payment calculation in ALPHA:
                    # 1. Cost in USDC: hashrate_ph * hashp_usdc * price_multiplier * (window_blocks / blocks_per_day)
                    # 2. Convert to ALPHA: cost_usdc / alpha_usdc
                    cost_usdc = hashrate_ph * hashp_usdc * price_multiplier * (window_blocks / blocks_per_day)
                    payment_alpha = cost_usdc / alpha_usdc if alpha_usdc > 0 else 0.0

                    payments_alpha[hotkey] = payments_alpha.get(hotkey, 0.0) + payment_alpha
                    window_spent_alpha += payment_alpha

                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.warning(
                        "Failed to calculate payment",
                        window=window_num,
                        hotkey=hotkey,
                        hashrate=hashrate,
                        error=str(e),
                    )
                    continue

            total_spent_alpha += window_spent_alpha
            total_budget_alpha += window_alpha_budget
            window_burn_alpha = max(0.0, window_alpha_budget - window_spent_alpha)

            window_details.append(
                {
                    "window": window_num,
                    "window_blocks": window_blocks,
                    "budget_alpha_window": window_alpha_budget,  # ALPHA budget for this window
                    "spent_alpha_window": window_spent_alpha,  # ALPHA spent in this window
                    "burn_alpha_window": window_burn_alpha,  # ALPHA burned in this window
                    "budget_ph_daily": daily_budget_ph,  # Daily PH capacity (constant)
                    "hashp_usdc": hashp_usdc,
                    "alpha_tao": alpha_tao,
                    "tao_usdc": tao_usdc,
                    "alpha_usdc": alpha_usdc,
                    "daily_usdc_budget": daily_usdc_budget,
                }
            )

        if not payments_alpha:
            logger.warning(
                "No delivered winners across all windows for validation epoch %s",
                previous_subnet_epoch_start,
            )
            return None

        # Calculate burn statistics (in ALPHA)
        total_burn_alpha = total_budget_alpha - total_spent_alpha
        burn_proportion = total_burn_alpha / total_budget_alpha if total_budget_alpha > 0 else 0.0

        logger.info(
            "Budget and burn analysis (ALPHA-based)",
            epoch=previous_subnet_epoch_start,
            total_budget_alpha=total_budget_alpha,
            total_spent_alpha=total_spent_alpha,
            total_burn_alpha=total_burn_alpha,
            burn_proportion=burn_proportion,
            window_details=window_details,
        )

        # Implement burn mechanism: allocate unused emissions to subnet owner
        # Get neurons to map burn UID to hotkey
        neurons = await subnet.list_neurons()

        # Get burn UID (subnet owner's earliest registered neuron)
        burn_uid = await get_burn_uid(bittensor, settings.BITTENSOR_NETUID, neurons)

        if burn_uid is not None and total_burn_alpha > 0:
            # Find hotkey for burn UID
            burn_hotkey = None
            for neuron in neurons:
                if neuron.uid == burn_uid:
                    burn_hotkey = neuron.hotkey
                    break

            if burn_hotkey:
                # Add burn amount to payments (will be included in weight normalization)
                payments_alpha[burn_hotkey] = payments_alpha.get(burn_hotkey, 0.0) + total_burn_alpha
                logger.info(
                    "Burn allocated to subnet owner",
                    burn_uid=burn_uid,
                    burn_hotkey=burn_hotkey,
                    burn_alpha=total_burn_alpha,
                )
            else:
                logger.warning(
                    "Burn UID found but hotkey not found in neuron list",
                    burn_uid=burn_uid,
                )
        elif burn_uid is None:
            logger.warning("Could not determine burn UID, burn will be excluded from weights")
        # If total_burn_alpha <= 0, no burn to allocate

        # Normalize weights based on ALPHA payments (including burn)
        weights: dict[str, float] = {}
        total_payments_alpha = sum(payments_alpha.values())
        if total_payments_alpha > 0:
            weights = {hk: payment / total_payments_alpha for hk, payment in payments_alpha.items()}

        # Create batch for mechanism 1
        batch = await WeightsBatch.objects.acreate(
            block=block.number,
            epoch_start=previous_subnet_epoch_start,
            mechanism_id=1,
            weights=weights,
            scored=False,
            should_be_scored=True,
        )

        logger.info(
            "Auction weights batch saved: %s. Previous validation epoch: #%s, current block: #%s, window: %s, winners: %s",
            batch.id,
            previous_subnet_epoch_start,
            block.number,
            window_number,
            len(weights),
        )

        return weights


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def calculate_auction_weights(*, event_loop: Any = None):
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.AUCTION_VALIDATION_SCHEDULING)
        except Locked:
            logger.debug("Another thread already calculating auction weights")
            return

        return run_async(calculate_auction_weights_async, event_loop=event_loop)


async def set_auction_weights_async() -> bool:
    """Set weights for mechanism 1 using set_mechanism_weights."""
    batches = [
        batch
        async for batch in WeightsBatch.objects.filter(
            mechanism_id=1,
            scored=False,
            should_be_scored=True,
        )
    ]

    if not batches:
        logger.debug("No auction weight batches to score")
        return False

    async with turbobt.Bittensor(
        settings.BITTENSOR_NETWORK,
        wallet=bittensor_wallet.Wallet(
            name=settings.BITTENSOR_WALLET_NAME,
            hotkey=settings.BITTENSOR_WALLET_HOTKEY_NAME,
            path=str(settings.BITTENSOR_WALLET_DIRECTORY),
        ),
    ) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )
        epoch = subnet.epoch(block.number)

        # Batches for the previous epoch are valid, only expire batches older than that
        subnet_tempo = subnet.tempo + 1
        previous_epoch_start = epoch.start - subnet_tempo
        expired_batches = [batch for batch in batches if batch.epoch_start < previous_epoch_start]

        if expired_batches:
            for batch in expired_batches:
                batch.should_be_scored = False

            await WeightsBatch.objects.abulk_update(
                expired_batches,
                fields=("should_be_scored",),
            )

            logger.error(
                "Expired auction weight batches: [%s]",
                ", ".join(str(batch.id) for batch in expired_batches),
            )
            return False

        if len(batches) > 1:
            logger.error("Unexpected number of auction weight batches: %s", len(batches))

            for batch in batches[:-1]:
                batch.scored = True

            batches = [batches[-1]]

        logger.info(
            "Selected auction weight batches for scoring: [%s]",
            ", ".join(str(batch.id) for batch in batches),
        )

        # Query neurons at current block, not historical validation block
        # (We can't go back in time for neuron state)
        neurons = {neuron.hotkey: neuron for neuron in await subnet.list_neurons()}

        weights_by_uid = collections.defaultdict[int, float](float)

        for batch in batches:
            for hotkey, hotkey_weight in batch.weights.items():
                try:
                    neuron = neurons[hotkey]
                except KeyError:
                    logger.warning(
                        "Hotkey not found in neurons",
                        hotkey=hotkey,
                        weight=hotkey_weight,
                    )
                    continue

                weights_by_uid[neuron.uid] += hotkey_weight

            batch.scored = True

        if not weights_by_uid:
            logger.warning("No weights to set for mechanism 1")
            return False

        # Commit weights using commit/reveal scheme (same pattern as mechanism 0)
        async for attempt in tenacity.AsyncRetrying(
            before_sleep=tenacity.before_sleep_log(logger, logging.ERROR),
            stop=tenacity.stop_after_attempt(1),
            wait=tenacity.wait_fixed(WEIGHT_SETTING_FAILURE_BACKOFF),
        ):
            with attempt:
                reveal_round = await commit_mechanism_weights(
                    bittensor=bittensor,
                    netuid=settings.BITTENSOR_NETUID,
                    mechanism_id=1,
                    weights=weights_by_uid,
                    version_key=0,
                )

        await WeightsBatch.objects.abulk_update(
            batches,
            fields=("scored",),
        )

        logger.info(
            "Auction weights committed for mechanism 1 (commit/reveal): %s UIDs, reveal_round: %s, batches: [%s]",
            len(weights_by_uid),
            reveal_round,
            ", ".join(str(batch.id) for batch in batches),
        )

        return True


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def set_auction_weights(*, event_loop: Any = None) -> bool:
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.AUCTION_WEIGHT_SETTING)
        except Locked:
            logger.debug("Another thread already setting auction weights")
            return False

        return run_async(set_auction_weights_async, event_loop=event_loop)


# --- Luxor scraper ---


async def fetch_current_luxor_hashrates(
    subaccount_name: str,
    api_key: str,
) -> dict[str, dict[str, any]]:
    """Fetch current worker data from Luxor API.

    Args:
        subaccount_name: Luxor subaccount name
        api_key: Luxor API key for this subaccount

    Returns:
        {
            "worker_name": {
                "hashrate": 12345,
                "efficiency": 100.0,
                "revenue": 0.0,
                "last_updated": "2025-01-01T00:00:00Z"
            }
        }
    """
    url = "/v1/pool/workers/BTC"
    params = {
        "subaccount_names": subaccount_name,  # Pass as string, not array
        "status": "ACTIVE",  # Only fetch active workers
        "page_number": 1,
        "page_size": 1000,  # Get all workers in one page
    }

    async with httpx.AsyncClient(
        base_url=settings.LUXOR_API_URL,
        headers={"Authorization": api_key},
        timeout=30.0,
    ) as client:
        logger.debug(
            "Fetching Luxor workers",
            url=f"{settings.LUXOR_API_URL}{url}",
            params=params,
            subaccount=subaccount_name,
        )
        response = await client.get(url, params=params)
        logger.debug(
            "Luxor API response",
            status=response.status_code,
            headers=dict(response.headers),
        )
        response.raise_for_status()
        data = response.json()

    logger.debug(
        "Luxor API response parsed",
        response_keys=list(data.keys()),
        total_active=data.get("total_active"),
        total_inactive=data.get("total_inactive"),
        workers_count=len(data.get("workers", [])),
    )

    # Extract worker data
    workers = data.get("workers", [])
    current = {}

    for worker in workers:
        worker_name = worker.get("name", "")
        if not worker_name:
            logger.debug("Skipping worker with no name", worker_keys=list(worker.keys()))
            continue

        # Hashrate is already in H/s, no conversion needed
        hashrate_hs = int(worker.get("hashrate", 0))

        # Efficiency is a decimal (e.g., 0.996 = 99.6%)
        efficiency_decimal = worker.get("efficiency", 0)
        efficiency_percent = float(efficiency_decimal * 100) if efficiency_decimal else 0.0

        current[worker_name] = {
            "hashrate": hashrate_hs,
            "efficiency": efficiency_percent,
            "revenue": 0.0,  # Not available in workers endpoint
            "last_updated": worker.get("last_share_time", ""),
        }

    logger.debug(
        "Extracted workers",
        count=len(current),
        worker_names=list(current.keys())[:5] if current else [],
    )

    return current


async def scrape_luxor_async(subaccount_name: str) -> int:
    """Scrape current Luxor hashrate data and store if changed.

    Args:
        subaccount_name: Luxor subaccount name to scrape

    Returns:
        Number of workers recorded (0 if no change detected)
    """
    from infinite_hashes.validator.models import ValidatorScrapingEvent

    # Look up API key for this subaccount
    api_key = settings.LUXOR_API_KEY_BY_SUBACCOUNT.get(subaccount_name)
    if not api_key:
        logger.error(
            "No API key configured for subaccount",
            subaccount=subaccount_name,
            configured_subaccounts=list(settings.LUXOR_API_KEY_BY_SUBACCOUNT.keys()),
        )
        raise ValueError(f"No API key configured for subaccount: {subaccount_name}")

    snapshot_time = datetime.datetime.now(tz=datetime.UTC)

    # Get current block number for heartbeat tracking
    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        head = await bittensor.head.get()
        current_block = head.number

    def parse_last_updated(value: Any) -> datetime.datetime | None:
        """Normalize Luxor last_updated values to timezone-aware datetimes."""
        if not value:
            return None
        if isinstance(value, datetime.datetime):
            return value if value.tzinfo else value.replace(tzinfo=datetime.UTC)
        if isinstance(value, str):
            normalized = value
            if normalized.endswith("Z"):
                normalized = f"{normalized[:-1]}+00:00"
            try:
                parsed = datetime.datetime.fromisoformat(normalized)
            except ValueError:
                logger.warning("Unable to parse Luxor last_updated", raw=value)
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.UTC)
        logger.warning("Unexpected last_updated type", type=type(value))
        return None

    try:
        current = await fetch_current_luxor_hashrates(subaccount_name, api_key)

        if not current:
            logger.debug("No workers found in Luxor API", subaccount=subaccount_name)
            # Still record scraping event even if no workers (validator is alive)
            await ValidatorScrapingEvent.objects.acreate(
                block_number=current_block,
                worker_count=0,
            )
            return 0

        # Get most recent snapshot for comparison
        last_snapshots = {}
        async for snapshot in (
            LuxorSnapshot.objects.filter(subaccount_name=subaccount_name)
            .order_by("worker_name", "-snapshot_time")
            .distinct("worker_name")
        ):
            last_snapshots[snapshot.worker_name] = {
                "hashrate": snapshot.hashrate,
                "efficiency": float(snapshot.efficiency),
                "last_updated": parse_last_updated(snapshot.last_updated),
            }

        parsed_last_updated: dict[str, datetime.datetime | None] = {}

        # Check which workers have new data (last_updated changed)
        # Only check last_updated (last share time) as it's the canonical signal
        # that the worker submitted a new share with fresh data
        workers_to_store = []

        for worker_name, data in current.items():
            prev = last_snapshots.get(worker_name)
            current_last_updated = parse_last_updated(data.get("last_updated"))
            parsed_last_updated[worker_name] = current_last_updated

            if current_last_updated is None:
                logger.debug(
                    "Skipping worker with missing last_updated",
                    worker=worker_name,
                    last_updated_raw=data.get("last_updated"),
                )
                continue

            # Store if: new worker OR last_updated changed
            if prev is None:
                workers_to_store.append(worker_name)
                logger.debug(
                    "New worker detected",
                    worker=worker_name,
                    last_updated=data["last_updated"],
                )
            elif current_last_updated != prev.get("last_updated"):
                workers_to_store.append(worker_name)
                logger.debug(
                    "Worker last_updated changed",
                    worker=worker_name,
                    old_last_updated=prev.get("last_updated"),
                    new_last_updated=current_last_updated,
                )

        if not workers_to_store:
            logger.debug(
                "No changes detected (no new shares)",
                subaccount=subaccount_name,
                workers=len(current),
            )
            return 0

        # Store new snapshots only for workers with changes
        snapshots = []
        for worker_name in workers_to_store:
            data = current[worker_name]
            last_updated = parsed_last_updated.get(worker_name)
            if last_updated is None:
                logger.debug(
                    "Skipping worker with unparsable last_updated",
                    worker=worker_name,
                    last_updated_raw=data.get("last_updated"),
                )
                continue

            snapshots.append(
                LuxorSnapshot(
                    snapshot_time=snapshot_time,
                    subaccount_name=subaccount_name,
                    worker_name=worker_name,
                    hashrate=data["hashrate"],
                    efficiency=data["efficiency"],
                    revenue=data["revenue"],
                    last_updated=last_updated,
                )
            )

        await LuxorSnapshot.objects.abulk_create(snapshots)

        # Record scraping event (heartbeat for data completeness tracking)
        await ValidatorScrapingEvent.objects.acreate(
            block_number=current_block,
            worker_count=len(snapshots),
        )

        total_hashrate = sum(s.hashrate for s in snapshots)
        logger.info(
            "Luxor snapshot recorded",
            subaccount=subaccount_name,
            workers=len(snapshots),
            total_hashrate_ph=total_hashrate / 1e15,
            snapshot_time=snapshot_time.isoformat(),
            block_number=current_block,
        )

        return len(snapshots)

    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error fetching Luxor data",
            status=e.response.status_code,
            response=e.response.text[:500],
            request_url=str(e.request.url),
            request_headers={k: v for k, v in e.request.headers.items() if k.lower() != "authorization"},
        )
        raise
    except Exception as e:
        logger.exception("Error scraping Luxor data", error=str(e))
        raise


@app.task(autoretry_for=(httpx.HTTPStatusError,))
def scrape_luxor(subaccount_name: str | None = None) -> int:
    """Celery task to scrape Luxor hashrate data."""
    if subaccount_name is None:
        subaccount_name = settings.LUXOR_SUBACCOUNT_NAME_MECHANISM_1

    return async_to_sync(scrape_luxor_async)(subaccount_name)


async def cleanup_old_luxor_snapshots_async(days: int = 1) -> int:
    """Delete Luxor snapshots older than specified days.

    Args:
        days: Delete snapshots older than this many days (default: 1)

    Returns:
        Number of snapshots deleted
    """
    cutoff = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(days=days)

    # Count before deletion
    count_before = await LuxorSnapshot.objects.filter(snapshot_time__lt=cutoff).acount()

    if count_before == 0:
        logger.debug("No old Luxor snapshots to delete", cutoff=cutoff.isoformat())
        return 0

    # Delete old snapshots
    result = await LuxorSnapshot.objects.filter(snapshot_time__lt=cutoff).adelete()

    # result is a tuple: (count, {model: count})
    deleted_count = result[0] if isinstance(result, tuple) else result

    logger.info(
        "Cleaned up old Luxor snapshots",
        deleted_count=deleted_count,
        cutoff=cutoff.isoformat(),
        days=days,
    )

    return deleted_count


@app.task()
def cleanup_old_luxor_snapshots(days: int = 1) -> int:
    """Celery task to cleanup old Luxor snapshots.

    Args:
        days: Delete snapshots older than this many days (default: 1)

    Returns:
        Number of snapshots deleted
    """
    return async_to_sync(cleanup_old_luxor_snapshots_async)(days)


# --- Price consensus tasks ---


def to_fp18(value) -> int:
    """Convert a value to fixed-point 18 decimal representation."""
    from decimal import Decimal

    d = Decimal(str(value))
    return int((d * (10**18)).to_integral_value())


@app.task
def scrape_metrics():
    """Scrape configured metrics from configured sources and store observations.

    Simplicity first: one primary source per metric for now, using settings.
    """
    from infinite_hashes.validator.models import PriceObservation
    from infinite_hashes.validator.scrapers.hashrateindex import HashrateIndexClient
    from infinite_hashes.validator.scrapers.taostats import TaoStatsClient

    now = datetime.datetime.now(datetime.UTC)
    conf = getattr(settings, "PRICE_SOURCES", {})

    # Helper utilities to keep each scraper concise
    def _row_ts(row, *, ts_key: str, ts_is_iso: bool) -> datetime.datetime:
        ts = row[ts_key]
        if ts_is_iso:
            return datetime.datetime.fromisoformat(ts)
        # assume tz-aware datetime
        return ts

    def _log_series(label: str, rows: list[dict], *, ts_key: str, ts_is_iso: bool) -> None:
        first = _row_ts(rows[0], ts_key=ts_key, ts_is_iso=ts_is_iso).isoformat() if rows else None
        last = _row_ts(rows[-1], ts_key=ts_key, ts_is_iso=ts_is_iso).isoformat() if rows else None
        logger.info(f"{label} scrape response", count=len(rows), first=first, last=last)

    def _store_rows(
        *, metric: str, source: str, rows: list[dict], ts_key: str, value_key: str, ts_is_iso: bool
    ) -> None:
        if not rows:
            return
        # Sort ascending by timestamp and store up to 5 points
        rows_sorted = sorted(rows, key=lambda r: _row_ts(r, ts_key=ts_key, ts_is_iso=ts_is_iso))[:5]
        for r in rows_sorted:
            observed_at = _row_ts(r, ts_key=ts_key, ts_is_iso=ts_is_iso)
            PriceObservation.objects.create(
                metric=metric,
                source=source,
                observed_at=observed_at,
                price_fp18=to_fp18(r[value_key]),
            )
        newest = max(rows_sorted, key=lambda r: _row_ts(r, ts_key=ts_key, ts_is_iso=ts_is_iso))
        logger.info(
            f"{metric} stored observations",
            stored=len(rows_sorted),
            newest_observed_at=_row_ts(newest, ts_key=ts_key, ts_is_iso=ts_is_iso).isoformat(),
            newest_price=newest[value_key],
        )

    # TAO_USDC via TaoStats OHLC close (last day, up to 5 entries)
    if "TAO_USDC" in conf and settings.TAOSTATS_API_KEY:
        try:
            with TaoStatsClient(api_key=settings.TAOSTATS_API_KEY) as ts:
                # Query last day with hourly buckets; request 5 entries for resiliency
                rows = ts.price_ohlc(
                    asset="TAO",
                    period="1h",
                    start_datetime=now - datetime.timedelta(days=1),
                    end_datetime=now,
                    limit=5,
                )
            _log_series("TAO_USDC", rows, ts_key="timestamp", ts_is_iso=True)
            _store_rows(
                metric="TAO_USDC", source="taostats", rows=rows, ts_key="timestamp", value_key="close", ts_is_iso=True
            )
        except Exception:  # noqa: BLE001
            logger.exception("TAO_USDC scrape failed")

    # ALPHA_TAO via TaoStats dtao pool history (last day, up to 5 entries)
    if "ALPHA_TAO" in conf and settings.TAOSTATS_API_KEY:
        try:
            with TaoStatsClient(api_key=settings.TAOSTATS_API_KEY) as ts:
                # Use by_hour over last day to get multiple points; request 5 entries
                rows = ts.dtao_pool_history(
                    netuid=conf["ALPHA_TAO"].get("dtao_netuid", 89),
                    start_datetime=now - datetime.timedelta(days=1),
                    end_datetime=now,
                    frequency="by_hour",
                    limit=5,
                )
            _log_series("ALPHA_TAO", rows, ts_key="timestamp", ts_is_iso=True)
            _store_rows(
                metric="ALPHA_TAO", source="taostats", rows=rows, ts_key="timestamp", value_key="price", ts_is_iso=True
            )
        except Exception:  # noqa: BLE001
            logger.exception("ALPHA_TAO scrape failed")

    # HASHP_USDC via Hashrate Index hashprice USD (last day, take up to 5 most recent entries)
    if "HASHP_USDC" in conf:
        try:
            with HashrateIndexClient() as hi:
                # Request last day with a bucket supported by the API for span=1D
                # The API expects bucket to be one of ["15s", "5m", "15m"] for span=1D
                rows = hi.hashprice(
                    currency="USD",
                    hashunit=conf["HASHP_USDC"].get("hashunit", "PHS"),
                    bucket="15m",
                    span="1D",
                )
            _log_series("HASHP_USDC", rows, ts_key="timestamp", ts_is_iso=False)
            # Keep only newest 5 entries before storing
            latest5 = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:5]
            _store_rows(
                metric="HASHP_USDC",
                source="hashrateindex",
                rows=latest5,
                ts_key="timestamp",
                value_key="price",
                ts_is_iso=False,
            )
        except Exception:  # noqa: BLE001
            logger.exception("HASHP_USDC scrape failed")


def _select_freshest(metric: str, max_age_sec: int, priority: list[str]) -> int | None:
    """Select the freshest price observation for a metric."""
    from infinite_hashes.validator.models import PriceObservation

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=max_age_sec)
    logger.debug(
        "Selecting freshest observation",
        metric=metric,
        max_age_sec=max_age_sec,
        cutoff=cutoff.isoformat(),
        priority=priority,
    )
    for src in priority:
        qs = PriceObservation.objects.filter(metric=metric, source=src)
        latest = qs.order_by("-observed_at").first()
        if latest is not None:
            logger.debug(
                "Observation candidate",
                metric=metric,
                source=src,
                observed_at=latest.observed_at.isoformat(),
                age_sec=(datetime.datetime.now(datetime.UTC) - latest.observed_at).total_seconds(),
            )
        obs = qs.filter(observed_at__gte=cutoff).order_by("-observed_at").first()
        if obs:
            return int(obs.price_fp18)  # type: ignore[return-value]
    logger.info("No fresh observation found", metric=metric)
    return None


@app.task
def publish_local_commitment(*, event_loop: Any = None):
    """Publish local price commitment with ban bitmap to the chain.

    Only publishes if the commitment has changed from the current on-chain value.
    """

    conf = getattr(settings, "PRICE_SOURCES", {})

    prices: dict[str, int] = {}
    required = ["TAO_USDC", "ALPHA_TAO", "HASHP_USDC"]
    for metric in required:
        mconf = conf.get(metric, {})
        fp = _select_freshest(metric, mconf.get("max_age_sec", 300), mconf.get("priority", []))
        if fp is None:
            logger.warning("Skipping publish: metric missing or stale", metric=metric)
            return
        prices[metric] = fp

    # Query banned miners from last 90 days
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=90)
    banned_hotkeys = set(BannedMiner.objects.filter(banned_at__gte=cutoff).values_list("hotkey", flat=True))

    return run_async(_publish_price_commitment_async, prices, banned_hotkeys, event_loop=event_loop)


async def _publish_price_commitment_async(prices: dict[str, int], banned_hotkeys: set[str]):
    """Publish price commitment with ban bitmap to the chain."""
    from infinite_hashes.consensus.price import PriceCommitment

    async with turbobt.Bittensor(
        settings.BITTENSOR_NETWORK,
        wallet=bittensor_wallet.Wallet(
            name=settings.BITTENSOR_WALLET_NAME,
            hotkey=settings.BITTENSOR_WALLET_HOTKEY_NAME,
            path=str(settings.BITTENSOR_WALLET_DIRECTORY),
        ),
    ) as bittensor:
        subnet = bittensor.subnet(settings.BITTENSOR_NETUID)

        # Get current block and neurons
        block = await bittensor.head.get()
        neurons = await subnet.list_neurons()

        # Map banned hotkeys to UIDs
        hotkey_to_uid = {n.hotkey: n.uid for n in neurons}
        banned_uids = {hotkey_to_uid[hk] for hk in banned_hotkeys if hk in hotkey_to_uid}

        # Create ban bitmap
        bans_bitmap = PriceCommitment.create_ban_bitmap(banned_uids)

        # Create commitment with prices and bans
        commit = PriceCommitment(t="p", prices=prices, bans=bans_bitmap, v=1)
        payload = commit.to_compact_bytes()

        # Fetch current commitment to check if it changed
        try:
            current_commits = await subnet.commitments.fetch(block_hash=block.hash)
            current_payload = current_commits.get(bittensor.wallet.hotkey.ss58_address)

            if current_payload:
                # Compare with new payload
                if isinstance(current_payload, str):
                    current_payload_bytes = current_payload.encode("utf-8")
                else:
                    current_payload_bytes = bytes(current_payload)

                if current_payload_bytes == payload:
                    logger.debug(
                        "Commitment unchanged, skipping publish",
                        hotkey=bittensor.wallet.hotkey.ss58_address,
                        banned_uids_count=len(banned_uids),
                    )
                    return False
        except Exception as e:
            logger.warning("Failed to fetch current commitment, publishing anyway", error=str(e))

        # Publish new commitment
        logger.info(
            "Publishing commitment",
            wallet_name=settings.BITTENSOR_WALLET_NAME,
            wallet_hotkey=settings.BITTENSOR_WALLET_HOTKEY_NAME,
            hotkey=bittensor.wallet.hotkey.ss58_address,
            payload_len=len(payload),
            banned_uids_count=len(banned_uids),
        )

        extrinsic = await subnet.commitments.set(
            data=payload,
            wallet=bittensor.wallet,
        )
        await extrinsic.wait_for_finalization()

        logger.info(
            "Commitment published",
            bytes_len=len(payload),
            hotkey=bittensor.wallet.hotkey.ss58_address,
            banned_uids=sorted(banned_uids),
        )

        return True
