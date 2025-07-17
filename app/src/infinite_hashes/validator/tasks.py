import asyncio
import collections
import datetime
import enum
import logging
from typing import NamedTuple, TypeAlias

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

from infinite_hashes.celery import app
from infinite_hashes.validator.locks import Locked, LockType, get_advisory_lock
from infinite_hashes.validator.models import WeightsBatch

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
    window: datetime.timedelta = datetime.timedelta(days=1),
):
    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(settings.BITTENSOR_NETUID).get(),
        )
        epoch = subnet.epoch(block.number)

        batches = WeightsBatch.objects.filter(
            epoch_start=epoch.start,
        )

        if await batches.acount():
            ids = [str(b.id) async for b in batches]
            logger.debug("Already have weights for this epoch: %s", ",".join(ids))
            return

        validation_start_block_number = epoch.start + int(settings.VALIDATION_OFFSET * subnet.tempo)
        blocks_left = validation_start_block_number - block.number

        if blocks_left > 0:
            logger.debug(
                "Too early to calculate weights. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
                epoch.start,
                block.number,
                round((block.number - epoch.start) / subnet.tempo, 2),
                subnet.tempo,
            )
            return

        if abs(blocks_left) >= settings.VALIDATION_THRESHOLD * subnet.tempo:
            logger.error(
                "Too late to calculate weights. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
                epoch.start,
                block.number,
                round((block.number - epoch.start) / subnet.tempo, 2),
                subnet.tempo,
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
            weights=weights,
        )

        logger.debug(
            "Weight batch saved: %s. Epoch start block: #%s, current block: #%s (=start + %s*%s)",
            batch.id,
            epoch.start,
            block.number,
            round((block.number - epoch.start) / subnet.tempo, 2),
            subnet.tempo,
        )

        return weights


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def calculate_weights():
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.VALIDATION_SCHEDULING)
        except Locked:
            logger.debug("Another thread already scheduling validation")
            return

        return async_to_sync(calculate_weights_async)()


async def get_hashrates_async(
    subaccount_name: str,
    start: datetime.datetime,
    end: datetime.datetime,
    page_size: int = 100,
    tick_size: TickSize = TickSize.HOUR,
) -> Hashrates:
    hashrates = collections.defaultdict(list)

    async with httpx.AsyncClient(
        base_url="https://app.luxor.tech/api",
        headers={
            "Authorization": settings.LUXOR_API_KEY,
        },
    ) as luxor:
        window = end - start
        samples = int(window / tick_size.timedelta)

        page_url = httpx.URL(
            f"/v1/pool/workers-hashrate-efficiency/BTC/{subaccount_name}",
            params={
                "end_date": end.date().isoformat(),
                "page_number": 1,
                "page_size": page_size,
                "start_date": start.date().isoformat(),
                "tick_size": tick_size.label,
            },
        )

        while True:
            response = await luxor.get(page_url)
            response.raise_for_status()
            response_json = response.json()

            for worker_name, worker_hashrates in response_json["hashrate_efficiency_revenue"].items():
                worker_hotkey = worker_name[:48]
                worker_hashrates = [
                    int(hashrate["hashrate"])
                    for hashrate in worker_hashrates
                    if hashrate["date_time"] >= start.isoformat()
                ]

                if len(worker_hashrates) < samples:
                    worker_hashrates += [0] * (samples - len(worker_hashrates))

                hashrates[worker_hotkey].append(int(sum(worker_hashrates) / len(worker_hashrates)))

            if not response_json["pagination"]["next_page_url"]:
                break

            page_url = response_json["pagination"]["next_page_url"]

    return hashrates


async def set_weights_async() -> bool:
    batches = [
        batch
        async for batch in WeightsBatch.objects.filter(
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

        validation_start_block_number = epoch.start + int(settings.VALIDATION_OFFSET * subnet.tempo)
        validation_start_block = await bittensor.block(validation_start_block_number).get()

        async with validation_start_block:
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
def set_weights() -> bool:
    with transaction.atomic():
        try:
            get_advisory_lock(LockType.WEIGHT_SETTING)
        except Locked:
            logger.debug("Another thread already scheduling validation")
            return False

        return async_to_sync(set_weights_async)()
