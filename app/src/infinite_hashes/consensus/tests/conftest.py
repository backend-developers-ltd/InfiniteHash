import pytest_asyncio

from infinite_hashes.testutils.unit.subtensor import SubtensorSimulator


@pytest_asyncio.fixture(scope="session")
async def sim():
    async with SubtensorSimulator() as s:
        yield s


@pytest_asyncio.fixture(autouse=True)
async def _clean_sim_between_tests(sim):
    # Reset simulator responses to a minimal baseline between tests
    sim.reset()
    yield
