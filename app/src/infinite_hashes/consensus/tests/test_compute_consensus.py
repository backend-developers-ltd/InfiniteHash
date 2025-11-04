import pytest
import scalecodec.utils.ss58 as ss58

from infinite_hashes.consensus.price import (
    PriceCommitment,
    _gamma_msi_fp18,
    compute_ban_consensus,
    compute_price_consensus,
)


def fp18(x: float) -> int:
    return int(x * 1e18)


@pytest.mark.asyncio
async def test_compute_price_consensus_with_simulator(sim):
    # Deterministic hotkeys
    hk1 = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    hk2 = ss58.ss58_encode(bytes([1]) * 32)
    hk3 = ss58.ss58_encode(bytes([2]) * 32)
    hk4 = ss58.ss58_encode(bytes([3]) * 32)
    hotkeys = [hk1, hk2, hk3, hk4]

    stakes = {
        hk1: 2.0,
        hk2: 1.0,
        hk3: 3.0,
        hk4: 0.5,
    }

    prices = {
        hk1: {"TAO_USDC": 10.0, "ALPHA_TAO": 0.5},
        hk2: {"TAO_USDC": 9.8},
        hk3: {"TAO_USDC": 10.2, "ALPHA_TAO": 0.45},
        hk4: {"TAO_USDC": 12.0},
    }

    # Inject commitments
    entries = {}
    for hk in hotkeys:
        pm = {m: int(fp18(v)) for m, v in prices.get(hk, {}).items()}
        payload = PriceCommitment(t="p", prices=pm, v=1).to_compact()
        entries[hk] = payload
    sim.set_commitments(netuid=1, entries=entries)

    # Provide neurons via simulator (SCALE-encoded via registry)
    sim.set_neurons_lite(
        netuid=1,
        neurons=[{"hotkey": hk, "stake": stakes[hk]} for hk in hotkeys],
    )

    block_number = 123
    # Set block context (header number + matching block hash)
    sim.set_block_context(number=block_number)
    out = await compute_price_consensus(
        netuid=1,
        block_number=block_number,
        metrics=["TAO_USDC", "ALPHA_TAO", "HASHP_USDC"],
        bt=sim.bt,
    )

    # Compute expected using the core γ‑MSI implementation
    stakes_fp = {hk: fp18(stakes[hk]) for hk in hotkeys}
    tao_prices = {hk: fp18(v) for hk, m in prices.items() if (v := m.get("TAO_USDC")) is not None}
    alpha_prices = {hk: fp18(v) for hk, m in prices.items() if (v := m.get("ALPHA_TAO")) is not None}

    expected = {
        "TAO_USDC": _gamma_msi_fp18(tao_prices, stakes_fp, gamma=0.67),
        "ALPHA_TAO": _gamma_msi_fp18(alpha_prices, stakes_fp, gamma=0.67),
        "HASHP_USDC": None,
    }

    for m in ["TAO_USDC", "ALPHA_TAO", "HASHP_USDC"]:
        assert out[m] == expected[m]


@pytest.mark.asyncio
async def test_compute_price_consensus_no_metrics_discovered_returns_empty(sim):
    # Commitments that fail validation or produce no prices should lead to empty metrics discovery
    # All commitments invalid: wrong discriminant and invalid compact
    # Use valid ss58 hotkeys
    hk1 = ss58.ss58_encode(bytes([4]) * 32)
    hk2 = ss58.ss58_encode(bytes([5]) * 32)
    sim.set_commitments(
        netuid=1,
        entries={
            hk1: "1;x;A=1;",  # wrong type token
            hk2: "1;p",  # incomplete compact
        },
    )
    # Provide a neuron so stake map exists, but no metrics discovered
    sim.set_neurons_lite(netuid=1, neurons=[{"hotkey": hk1, "stake": 1.0}])

    block_number = 999
    # Set block context (header number + matching block hash)
    sim.set_block_context(number=block_number)
    out = await compute_price_consensus(
        netuid=1,
        block_number=block_number,
        metrics=None,
        bt=sim.bt,
    )
    assert out == {}


@pytest.mark.asyncio
async def test_compute_ban_consensus_majority_rule(sim):
    """Test that UIDs are banned only when >50% of stake supports the ban."""
    # Setup hotkeys with different stakes
    hk1 = ss58.ss58_encode(bytes([10]) * 32)
    hk2 = ss58.ss58_encode(bytes([11]) * 32)
    hk3 = ss58.ss58_encode(bytes([12]) * 32)
    hk4 = ss58.ss58_encode(bytes([13]) * 32)

    stakes = {
        hk1: 40.0,  # 40% stake
        hk2: 30.0,  # 30% stake
        hk3: 20.0,  # 20% stake
        hk4: 10.0,  # 10% stake
    }

    # Setup neurons with UIDs
    # hk1 (uid=0), hk2 (uid=1), hk3 (uid=2), hk4 (uid=3)
    sim.set_neurons_lite(
        netuid=1,
        neurons=[
            {"hotkey": hk1, "stake": stakes[hk1], "uid": 0},
            {"hotkey": hk2, "stake": stakes[hk2], "uid": 1},
            {"hotkey": hk3, "stake": stakes[hk3], "uid": 2},
            {"hotkey": hk4, "stake": stakes[hk4], "uid": 3},
        ],
    )

    # Create ban bitmaps:
    # - hk1 (40% stake) bans uid=2 and uid=3
    # - hk2 (30% stake) bans uid=2
    # - hk3 (20% stake) bans nothing (target, won't be banned)
    # - hk4 (10% stake) doesn't submit commitment
    #
    # Expected results:
    # - uid=2 has 70% stake (hk1+hk2) -> BANNED (>50%)
    # - uid=3 has 40% stake (hk1 only) -> NOT BANNED (<=50%)

    ban_bitmap_hk1 = PriceCommitment.create_ban_bitmap({2, 3})
    ban_bitmap_hk2 = PriceCommitment.create_ban_bitmap({2})
    ban_bitmap_hk3 = b"\x00" * 32  # no bans

    entries = {
        hk1: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, bans=ban_bitmap_hk1, v=1).to_compact_bytes(),
        hk2: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, bans=ban_bitmap_hk2, v=1).to_compact_bytes(),
        hk3: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, bans=ban_bitmap_hk3, v=1).to_compact_bytes(),
    }
    sim.set_commitments(netuid=1, entries=entries)

    block_number = 456
    sim.set_block_context(number=block_number)

    banned_hotkeys = await compute_ban_consensus(
        netuid=1,
        block_number=block_number,
        bt=sim.bt,
    )

    # Only uid=2 should be banned (hk3)
    assert banned_hotkeys == {hk3}


@pytest.mark.asyncio
async def test_compute_ban_consensus_no_bans(sim):
    """Test that no bans are returned when no validators submit ban bitmaps."""
    hk1 = ss58.ss58_encode(bytes([20]) * 32)
    hk2 = ss58.ss58_encode(bytes([21]) * 32)

    sim.set_neurons_lite(
        netuid=1,
        neurons=[
            {"hotkey": hk1, "stake": 50.0, "uid": 0},
            {"hotkey": hk2, "stake": 50.0, "uid": 1},
        ],
    )

    # No ban bitmaps (all zeros)
    entries = {
        hk1: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, v=1).to_compact(),
        hk2: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, v=1).to_compact(),
    }
    sim.set_commitments(netuid=1, entries=entries)

    block_number = 789
    sim.set_block_context(number=block_number)

    banned_hotkeys = await compute_ban_consensus(
        netuid=1,
        block_number=block_number,
        bt=sim.bt,
    )

    assert banned_hotkeys == set()


@pytest.mark.asyncio
async def test_compute_ban_consensus_exact_50_percent_not_banned(sim):
    """Test that exactly 50% stake does NOT result in a ban (needs >50%)."""
    hk1 = ss58.ss58_encode(bytes([30]) * 32)
    hk2 = ss58.ss58_encode(bytes([31]) * 32)
    hk3 = ss58.ss58_encode(bytes([32]) * 32)

    sim.set_neurons_lite(
        netuid=1,
        neurons=[
            {"hotkey": hk1, "stake": 50.0, "uid": 0},
            {"hotkey": hk2, "stake": 50.0, "uid": 1},
            {"hotkey": hk3, "stake": 0.0, "uid": 2},
        ],
    )

    # hk1 (50% stake) bans uid=2
    # hk2 (50% stake) doesn't ban uid=2
    # Result: exactly 50% stake -> NOT BANNED
    ban_bitmap_hk1 = PriceCommitment.create_ban_bitmap({2})

    entries = {
        hk1: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, bans=ban_bitmap_hk1, v=1).to_compact_bytes(),
        hk2: PriceCommitment(t="p", prices={"TAO_USDC": fp18(10.0)}, v=1).to_compact(),
    }
    sim.set_commitments(netuid=1, entries=entries)

    block_number = 999
    sim.set_block_context(number=block_number)

    banned_hotkeys = await compute_ban_consensus(
        netuid=1,
        block_number=block_number,
        bt=sim.bt,
    )

    # Exactly 50% should NOT ban
    assert banned_hotkeys == set()
