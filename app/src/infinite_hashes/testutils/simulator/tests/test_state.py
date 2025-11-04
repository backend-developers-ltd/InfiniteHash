import datetime as dt

import pytest

from infinite_hashes.testutils.simulator import server as simulator_server
from infinite_hashes.testutils.simulator.state import MECHANISM_SPLIT_VALUE_MAX, SimulatorState


@pytest.fixture()
def simulator_state() -> SimulatorState:
    return SimulatorState()


def test_default_head_initialises_block_zero(simulator_state: SimulatorState):
    head = simulator_state.head()
    assert head.number == 0
    assert head.hash.startswith("0x")


def test_setting_head_creates_block(simulator_state: SimulatorState):
    simulator_state.set_head(5, timestamp="2024-01-01T00:00:00Z")
    head = simulator_state.head()
    assert head.number == 5
    assert head.timestamp == dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    block = simulator_state.get_block(5)
    assert block.number == 5


def test_epoch_calculation_uses_tempo(simulator_state: SimulatorState):
    """Test epoch calculation matches turbobt's algorithm with netuid offset."""
    # For netuid=1, tempo=9, block=25:
    # netuid_plus_one = 2, tempo_plus_one = 10
    # adjusted_block = 25 + 2 = 27
    # remainder = 27 % 10 = 7
    # start = 25 - 7 - 1 = 17
    # end = 25 - 7 + 9 = 27
    simulator_state.update_subnet(1, tempo=9)
    simulator_state.set_head(25)
    epoch = simulator_state.subnet_epoch(1)
    assert epoch["start"] == 17
    assert epoch["end"] == 27


def test_commitments_roundtrip(simulator_state: SimulatorState):
    simulator_state.set_head(3)
    block_hash = simulator_state.head().hash
    simulator_state.publish_commitment(1, block_hash=block_hash, hotkey="hk", payload="payload")
    entries = simulator_state.fetch_commitments(1, block_hash=block_hash)
    decoded = bytes.fromhex(entries["hk"].removeprefix("0x")).decode("utf-8")
    assert decoded == "payload"


def test_commitments_scale_decoding(simulator_state: SimulatorState):
    simulator_state.set_head(5)
    block_hash = simulator_state.head().hash
    scale_payload = "0x040f313b623b313d31303b312e313d38"
    simulator_state.publish_commitment(1, block_hash=block_hash, hotkey="hk", payload=scale_payload)

    entries = simulator_state.fetch_commitments(1, block_hash=block_hash)
    decoded = bytes.fromhex(entries["hk"].removeprefix("0x")).decode("utf-8")
    assert decoded == "1;b;1=10;1.1=8"


# --- Mechanism Weights Tests ---


def test_mechanism_weights_empty_by_default(simulator_state: SimulatorState):
    """Test that mechanism weights are empty by default."""
    weights = simulator_state.get_mechanism_weights(netuid=1, mecid=0)
    assert weights == {}


def test_set_and_get_mechanism_weights(simulator_state: SimulatorState):
    """Test basic set and get mechanism weights."""
    weights_to_set = {0: 100, 1: 200, 2: 300}
    simulator_state.set_mechanism_weights(netuid=1, mecid=1, weights=weights_to_set)

    retrieved = simulator_state.get_mechanism_weights(netuid=1, mecid=1)
    assert retrieved == weights_to_set


def test_mechanism_weights_isolated_by_mecid(simulator_state: SimulatorState):
    """Test that different mechanisms have isolated weight storage."""
    simulator_state.set_mechanism_weights(netuid=1, mecid=0, weights={0: 100})
    simulator_state.set_mechanism_weights(netuid=1, mecid=1, weights={0: 200})

    weights_mec0 = simulator_state.get_mechanism_weights(netuid=1, mecid=0)
    weights_mec1 = simulator_state.get_mechanism_weights(netuid=1, mecid=1)

    assert weights_mec0 == {0: 100}
    assert weights_mec1 == {0: 200}


def test_mechanism_weights_backward_compat_with_legacy(simulator_state: SimulatorState):
    """Test that mechanism 0 syncs with legacy weights_by_uid."""
    weights = {0: 100, 1: 200}
    simulator_state.set_mechanism_weights(netuid=1, mecid=0, weights=weights)

    # Check mechanism 0 weights
    mec0_weights = simulator_state.get_mechanism_weights(netuid=1, mecid=0)
    assert mec0_weights == weights

    # Check legacy weights synced
    legacy_weights = simulator_state.weights(netuid=1)
    assert legacy_weights == weights


def test_legacy_set_weights_syncs_to_mechanism_0(simulator_state: SimulatorState):
    """Test that legacy set_weights syncs to mechanism 0."""
    weights = {0: 100, 1: 200}
    simulator_state.set_weights(netuid=1, weights=weights)

    # Check legacy interface
    legacy_weights = simulator_state.weights(netuid=1)
    assert legacy_weights == weights

    # Check mechanism 0 synced
    mec0_weights = simulator_state.get_mechanism_weights(netuid=1, mecid=0)
    assert mec0_weights == weights


def test_mechanism_weights_respect_block_history(simulator_state: SimulatorState):
    """Set weights at different blocks and verify historical lookups."""
    simulator_state.set_head(5)
    simulator_state.set_mechanism_weights(netuid=1, mecid=1, weights={0: 50})

    simulator_state.set_head(10)
    simulator_state.set_mechanism_weights(netuid=1, mecid=1, weights={0: 75})

    before_first = simulator_state.get_mechanism_weights(netuid=1, mecid=1, at_block=4)
    at_first = simulator_state.get_mechanism_weights(netuid=1, mecid=1, at_block=5)
    mid_epoch = simulator_state.get_mechanism_weights(netuid=1, mecid=1, at_block=7)
    latest = simulator_state.get_mechanism_weights(netuid=1, mecid=1)

    assert before_first == {}
    assert at_first == {0: 50}
    assert mid_epoch == {0: 50}
    assert latest == {0: 75}


def test_mechanism_count_defaults_to_one(simulator_state: SimulatorState):
    """Mechanism count should default to 1 for new subnets."""
    assert simulator_state.mechanism_count(1) == 1


def test_set_mechanism_count(simulator_state: SimulatorState):
    """Mechanism count updates and persists per subnet."""
    simulator_state.set_mechanism_count(1, 3)
    assert simulator_state.mechanism_count(1) == 3


def _split_values_for_testing(count: int) -> list[int]:
    if MECHANISM_SPLIT_VALUE_MAX is None:
        base = 1_000_000_000
    else:
        base = MECHANISM_SPLIT_VALUE_MAX
    step = max(1, base // (count + 1))
    return [step * (i + 1) for i in range(count)]


def test_set_mechanism_emission_split(simulator_state: SimulatorState):
    """Mechanism emission split can be configured per subnet."""
    split = _split_values_for_testing(3)
    simulator_state.set_mechanism_count(1, 3)
    simulator_state.set_mechanism_emission_split(1, split)
    assert simulator_state.mechanism_emission_split(1) == split


def test_clear_mechanism_emission_split(simulator_state: SimulatorState):
    """Clearing emission split should restore None."""
    simulator_state.set_mechanism_emission_split(1, _split_values_for_testing(2))
    assert simulator_state.mechanism_emission_split(1) is not None
    simulator_state.set_mechanism_emission_split(1, None)
    assert simulator_state.mechanism_emission_split(1) is None


@pytest.mark.asyncio
async def test_rpc_state_get_storage_mechanism_split(simulator_state: SimulatorState):
    if not simulator_server._MECH_EMISSION_SPLIT_PREFIX_HEX:
        pytest.skip("xxhash not available")

    key = simulator_server._twox64concat_storage_key(
        simulator_server._MECH_EMISSION_SPLIT_PREFIX_HEX,
        (1).to_bytes(2, "little"),
    )
    empty = await simulator_server.rpc_state_get_storage(simulator_state, [key])
    assert empty == "0x00"

    split = _split_values_for_testing(2)
    simulator_state.set_mechanism_emission_split(1, split)
    expected = simulator_server._codec.encode_hex(
        simulator_server._MECH_EMISSION_SPLIT_TYPE or "Option<Vec<u16>>",
        split,
    )
    encoded = await simulator_server.rpc_state_get_storage(simulator_state, [key])
    assert encoded == expected


@pytest.mark.asyncio
async def test_rpc_state_get_storage_mechanism_count(simulator_state: SimulatorState):
    if not simulator_server._MECH_COUNT_PREFIX_HEX:
        pytest.skip("xxhash not available")

    key = simulator_server._twox64concat_storage_key(
        simulator_server._MECH_COUNT_PREFIX_HEX,
        (1).to_bytes(2, "little"),
    )
    initial = await simulator_server.rpc_state_get_storage(simulator_state, [key])
    assert initial == simulator_server._codec.encode_hex("u8", 1)

    simulator_state.set_mechanism_count(1, 4)
    updated = await simulator_server.rpc_state_get_storage(simulator_state, [key])
    assert updated == simulator_server._codec.encode_hex("u8", 4)


# --- Commit-Reveal Tests ---


def test_commit_weights_stores_commit(simulator_state: SimulatorState):
    """Test that commit_weights stores a commit."""
    simulator_state.set_head(100)
    simulator_state.commit_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        commit_hash="0xabc123",
        reveal_round=150,
    )

    # Verify commit stored in subnet state
    subnet = simulator_state.ensure_subnet(1)
    commit_key = (1, "test_hotkey")
    assert commit_key in subnet.weight_commits
    commit = subnet.weight_commits[commit_key]
    assert commit.commit_hash == "0xabc123"
    assert commit.reveal_round == 150
    assert commit.committed_at_block == 100


def test_reveal_weights_validates_commit_hash(simulator_state: SimulatorState):
    """Test that reveal_weights validates the commit hash."""
    import hashlib

    simulator_state.set_head(100)
    weights = {0: 100, 1: 200}
    salt = b"test_salt"

    # Compute correct hash
    weights_bytes = str(sorted(weights.items())).encode("utf-8")
    correct_hash = "0x" + hashlib.sha256(weights_bytes + salt).hexdigest()

    # Commit with correct hash
    simulator_state.commit_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        commit_hash=correct_hash,
        reveal_round=150,
    )

    # Advance to reveal round
    simulator_state.set_head(150)

    # Reveal with correct weights and salt
    success = simulator_state.reveal_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        weights=weights,
        salt=salt,
    )

    assert success is True

    # Verify weights were applied
    applied_weights = simulator_state.get_mechanism_weights(netuid=1, mecid=1)
    assert applied_weights == weights

    # Verify commit was cleaned up
    subnet = simulator_state.ensure_subnet(1)
    commit_key = (1, "test_hotkey")
    assert commit_key not in subnet.weight_commits


def test_reveal_weights_fails_with_wrong_salt(simulator_state: SimulatorState):
    """Test that reveal fails with incorrect salt."""
    import hashlib

    simulator_state.set_head(100)
    weights = {0: 100, 1: 200}
    correct_salt = b"correct_salt"
    wrong_salt = b"wrong_salt"

    # Compute hash with correct salt
    weights_bytes = str(sorted(weights.items())).encode("utf-8")
    correct_hash = "0x" + hashlib.sha256(weights_bytes + correct_salt).hexdigest()

    simulator_state.commit_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        commit_hash=correct_hash,
        reveal_round=150,
    )

    simulator_state.set_head(150)

    # Try to reveal with wrong salt
    success = simulator_state.reveal_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        weights=weights,
        salt=wrong_salt,
    )

    assert success is False

    # Weights should not be applied
    applied_weights = simulator_state.get_mechanism_weights(netuid=1, mecid=1)
    assert applied_weights == {}


def test_reveal_weights_fails_before_reveal_round(simulator_state: SimulatorState):
    """Test that reveal fails before reveal_round."""
    import hashlib

    simulator_state.set_head(100)
    weights = {0: 100}
    salt = b"salt"

    weights_bytes = str(sorted(weights.items())).encode("utf-8")
    correct_hash = "0x" + hashlib.sha256(weights_bytes + salt).hexdigest()

    simulator_state.commit_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        commit_hash=correct_hash,
        reveal_round=150,
    )

    # Try to reveal before reveal_round (current block: 100, reveal_round: 150)
    success = simulator_state.reveal_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        weights=weights,
        salt=salt,
    )

    assert success is False


def test_reveal_weights_fails_without_commit(simulator_state: SimulatorState):
    """Test that reveal fails if no commit exists."""
    simulator_state.set_head(100)

    success = simulator_state.reveal_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        weights={0: 100},
        salt=b"salt",
    )

    assert success is False


def test_to_dict_includes_mechanism_weights_and_commits(simulator_state: SimulatorState):
    """Test that to_dict serializes mechanism weights and commits."""
    simulator_state.set_head(100)
    simulator_state.set_mechanism_weights(netuid=1, mecid=1, weights={0: 100})
    simulator_state.commit_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        commit_hash="0xabc",
        reveal_round=150,
    )

    state_dict = simulator_state.to_dict()

    # Verify structure
    assert "subnets" in state_dict
    assert 1 in state_dict["subnets"]
    subnet_dict = state_dict["subnets"][1]

    # Check mechanism weights
    assert "weights_by_mechanism" in subnet_dict
    assert 1 in subnet_dict["weights_by_mechanism"]
    assert subnet_dict["weights_by_mechanism"][1] == {0: 100}

    # Check commits
    assert "weight_commits" in subnet_dict
    assert "1:test_hotkey" in subnet_dict["weight_commits"]
    commit_data = subnet_dict["weight_commits"]["1:test_hotkey"]
    assert commit_data["commit_hash"] == "0xabc"
    assert commit_data["reveal_round"] == 150
