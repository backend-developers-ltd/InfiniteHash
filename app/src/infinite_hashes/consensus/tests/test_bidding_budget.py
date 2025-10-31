import pytest

from infinite_hashes.auctions.mechanism_split import (
    MECHANISM_SPLIT_DENOMINATOR,
    fetch_mechanism_share_fraction,
)


class _FakeState:
    def __init__(self, storage):
        # storage mapping: (key, params_tuple) -> value
        self._storage = storage

    async def getStorage(self, key, *params, block_hash=None):  # noqa: ARG002,N803
        return self._storage.get((key, tuple(params)))


class _FakeSubtensor:
    def __init__(self, storage):
        self.state = _FakeState(storage)


class _FakeBT:
    def __init__(self, storage):
        self.subtensor = _FakeSubtensor(storage)


@pytest.mark.asyncio
async def test_infer_mechanism_share_from_split():
    split_value = 40000
    storage = {
        ("SubtensorModule.MechanismEmissionSplit", (1,)): [1000, split_value],
        ("SubtensorModule.MechanismCountCurrent", (1,)): 2,
    }
    bt = _FakeBT(storage)
    share_fraction = await fetch_mechanism_share_fraction(bt, netuid=1, mechanism_id=1)
    expected = split_value / MECHANISM_SPLIT_DENOMINATOR
    assert pytest.approx(expected, rel=1e-12) == share_fraction


@pytest.mark.asyncio
async def test_infer_mechanism_share_even_split_when_missing():
    storage = {
        ("SubtensorModule.MechanismEmissionSplit", (1,)): None,
        ("SubtensorModule.MechanismCountCurrent", (1,)): 3,
    }
    bt = _FakeBT(storage)
    with pytest.raises(RuntimeError):
        await fetch_mechanism_share_fraction(bt, netuid=1, mechanism_id=1)


@pytest.mark.asyncio
async def test_infer_mechanism_share_returns_none_without_state():
    class _NoStateBT:
        subtensor = None

    with pytest.raises(RuntimeError):
        await fetch_mechanism_share_fraction(_NoStateBT(), netuid=1, mechanism_id=1)
