import datetime
from collections.abc import Callable

import celery.schedules
import turbobt
from asgiref.sync import async_to_sync


class ScheduleEveryBittensorEpoch(celery.schedules.BaseSchedule):
    BLOCK_TIME = datetime.timedelta(seconds=12)
    SUBNET_TEMPO = 360

    def __init__(
        self,
        netuid: int = 0,
        offset: int = 0,
        threshold: int = 0,
        tempo: int = SUBNET_TEMPO,
        network: str = "finney",
        block_time: datetime.timedelta = BLOCK_TIME,
        nowfun: Callable | None = None,
        app: celery.Celery | None = None,
    ):
        super().__init__(nowfun, app)

        self.netuid = netuid
        self.offset = offset
        self.threshold = threshold
        self.tempo = tempo
        self.network = network
        self.block_time = block_time

    @async_to_sync
    async def get_block_number(self) -> int:
        async with turbobt.Bittensor(self.network) as bittensor:
            block = await bittensor.blocks.head()

            return block.number

    def epoch(self, block_number: int) -> range:
        """
        The logic from Subtensor's Rust function:
            pub fn blocks_until_next_epoch(netuid: NetUid, tempo: u16, block_number: u64) -> u64
        See https://github.com/opentensor/subtensor/blob/f8db5d06c0439d4fb5db66be3632e4d89a8829c0/pallets/subtensor/src/coinbase/run_coinbase.rs#L846
        """

        netuid_plus_one = self.netuid + 1
        tempo_plus_one = self.tempo + 1
        adjusted_block = block_number + netuid_plus_one
        remainder = adjusted_block % tempo_plus_one

        if remainder == self.tempo:
            remainder = -1

        return range(
            block_number - remainder - 1,
            block_number - remainder + self.tempo,
        )

    def is_due(self, last_run_at: datetime.datetime) -> tuple[bool, datetime.datetime]:
        remaining = self.remaining_estimate(last_run_at)
        is_due = remaining.total_seconds() <= 0

        if is_due:
            remaining += (self.tempo + 1) * self.block_time

        return celery.schedules.schedstate(
            is_due=is_due,
            next=remaining.total_seconds(),
        )

    def remaining_estimate(self, last_run_at: datetime.datetime) -> datetime.timedelta:
        block_number = self.get_block_number()

        epoch = self.epoch(block_number)
        blocks_left = (epoch.start + self.offset) - block_number

        if blocks_left < 0 and abs(blocks_left) >= self.threshold:
            blocks_left = (epoch.stop + self.offset) - block_number

        remaining = blocks_left * self.block_time

        return remaining
