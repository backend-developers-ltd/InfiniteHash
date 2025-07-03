import asyncio
import collections
import datetime
import enum
from typing import NamedTuple, TypeAlias

import bittensor_wallet
import celery
import httpx
import structlog
import turbobt
import turbobt.substrate.exceptions
import websockets
from asgiref.sync import async_to_sync
from celery import Task
from celery.utils.log import get_task_logger
from django.conf import settings

from luxor_subnet.celery import app

BLOCK_TIME = datetime.timedelta(seconds=12)

Hashrates: TypeAlias = dict[str, int]


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


async def get_epoch_async(netuid: int) -> Epoch:
    async with turbobt.Bittensor(settings.BITTENSOR_NETWORK) as bittensor:
        block, subnet = await asyncio.gather(
            bittensor.head.get(),
            bittensor.subnet(netuid).get(),
        )

        assert subnet
        assert subnet.tempo == settings.BITTENSOR_SUBNET_TEMPO

        epoch = subnet.epoch(block.number)
        epoch_block = await bittensor.block(epoch.start).get()
        epoch_timestamp = await epoch_block.get_timestamp()

        return Epoch(
            epoch_block.number,
            epoch_timestamp,
        )


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def get_epoch(netuid: int) -> Epoch:
    return async_to_sync(get_epoch_async)(netuid)


async def get_hashrates_async(
    epoch: Epoch,
    subaccount_name: str,
    window: datetime.timedelta = datetime.timedelta(days=1),
    page_size: int = 100,
    tick_size: TickSize = TickSize.HOUR,
) -> tuple[
    Epoch,
    Hashrates,
]:
    validation_start = epoch.timestamp + BLOCK_TIME * settings.VALIDATION_OFFSET
    hashrates = collections.defaultdict(list)

    async with httpx.AsyncClient(
        base_url="https://app.luxor.tech/api",
        headers={
            "Authorization": settings.LUXOR_API_KEY,
        },
    ) as luxor:
        start, end = (
            validation_start - window,
            validation_start,
        )
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

    return (
        epoch,
        {hotkey: sum(hotkey_hashrates) for hotkey, hotkey_hashrates in hashrates.items()},
    )


@app.task(
    autoretry_for=(httpx.HTTPError,),
)
def get_hashrates(
    epoch: Epoch,
    subaccount_name: str,
) -> tuple[
    Epoch,
    Hashrates,
]:
    return async_to_sync(get_hashrates_async)(
        epoch,
        subaccount_name,
    )


async def set_weights_async(
    epoch: Epoch,
    hashrates: Hashrates,
    netuid: int,
) -> bool:
    if not hashrates:
        return False

    total_hashrate = sum(hashrates.values())

    async with turbobt.Bittensor(
        settings.BITTENSOR_NETWORK,
        wallet=bittensor_wallet.Wallet(
            settings.BITTENSOR_WALLET_NAME,
            settings.BITTENSOR_WALLET_HOTKEY_NAME,
        ),
    ) as bittensor:
        subnet = bittensor.subnet(netuid)

        async with bittensor.block(epoch.block + settings.VALIDATION_OFFSET):
            neurons = {neuron.hotkey: neuron for neuron in await subnet.list_neurons()}

        weights = {}

        for hotkey, hashrate in hashrates.items():
            try:
                neuron = neurons[hotkey]
            except KeyError:
                continue

            weights[neuron.uid] = hashrate / total_hashrate

        if not weights:
            return False

        await subnet.weights.commit(weights)

        return True


@app.task(
    autoretry_for=(
        turbobt.substrate.exceptions.SubstrateException,
        websockets.WebSocketException,
    ),
)
def set_weights(
    epoch: Epoch,
    hashrates: Hashrates,
    netuid: int,
) -> bool:
    return async_to_sync(set_weights_async)(
        epoch,
        hashrates,
        netuid,
    )


@app.task(
    ignore_result=True,
)
def update_weights(netuid, subaccount_name):
    chain = celery.chain(
        get_epoch.s(netuid),
        get_hashrates.s(subaccount_name),
        set_weights.s(netuid),
    )
    chain.apply_async()
