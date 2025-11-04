"""Tests for extrinsic decoding and application."""

import pytest

from infinite_hashes.testutils.simulator.extrinsics import (
    DecodedExtrinsic,
    apply_extrinsic,
    decode_extrinsic,
)
from infinite_hashes.testutils.simulator.state import SimulatorState


@pytest.fixture()
def simulator_state() -> SimulatorState:
    return SimulatorState()


# --- Extrinsic Decoding Tests ---


def test_decode_extrinsic_raises_on_invalid_hex():
    """Test that decode_extrinsic raises ValueError for invalid hex."""
    with pytest.raises(ValueError, match="Failed to decode extrinsic"):
        decode_extrinsic("0xinvalid")


def test_apply_set_mechanism_weights():
    """Test applying set_mechanism_weights extrinsic."""
    state = SimulatorState()
    extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="set_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "dests": [0, 1, 2],
            "weights": [100, 200, 300],
            "version_key": 0,
        },
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is True

    weights = state.get_mechanism_weights(netuid=1, mecid=1)
    assert weights == {0: 100, 1: 200, 2: 300}


def test_apply_set_mechanism_weights_validates_lengths():
    """Test that set_mechanism_weights validates dests/weights length mismatch."""
    state = SimulatorState()
    extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="set_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "dests": [0, 1],
            "weights": [100],  # Length mismatch
            "version_key": 0,
        },
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is False


def test_apply_commit_timelocked_mechanism_weights():
    """Test applying commit_timelocked_mechanism_weights extrinsic."""
    state = SimulatorState()
    state.set_head(100)

    extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="commit_timelocked_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "commit": "0xabc123",
            "reveal_round": 150,
            "commit_reveal_version": 4,
        },
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is True

    # Verify commit stored
    subnet = state.ensure_subnet(1)
    commit_key = (1, "test_signer")
    assert commit_key in subnet.weight_commits
    commit = subnet.weight_commits[commit_key]
    assert commit.commit_hash == "0xabc123"
    assert commit.reveal_round == 150


def test_apply_commit_without_0x_prefix():
    """Test that commit hash gets 0x prefix if missing."""
    state = SimulatorState()
    state.set_head(100)

    extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="commit_timelocked_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "commit": "abc123",  # No 0x prefix
            "reveal_round": 150,
            "commit_reveal_version": 4,
        },
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is True

    subnet = state.ensure_subnet(1)
    commit = subnet.weight_commits[(1, "test_signer")]
    assert commit.commit_hash == "0xabc123"


def test_apply_extrinsic_ignores_unsupported_pallet():
    """Test that unsupported pallets are ignored."""
    state = SimulatorState()
    extrinsic = DecodedExtrinsic(
        pallet="OtherPallet",
        call="some_call",
        params={},
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is False


def test_apply_extrinsic_ignores_unsupported_call():
    """Test that unsupported calls are ignored (but logged)."""
    state = SimulatorState()
    extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="unsupported_call",
        params={},
        signer="test_signer",
    )

    success = apply_extrinsic(state, extrinsic)
    assert success is False


# --- Integration Tests ---


def test_full_commit_reveal_workflow():
    """Test complete commit-reveal workflow."""
    import hashlib

    state = SimulatorState()
    state.set_head(100)

    # Step 1: Commit weights
    weights = {0: 100, 1: 200}
    salt = b"test_salt"
    weights_bytes = str(sorted(weights.items())).encode("utf-8")
    commit_hash = "0x" + hashlib.sha256(weights_bytes + salt).hexdigest()

    commit_extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="commit_timelocked_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "commit": commit_hash,
            "reveal_round": 150,
            "commit_reveal_version": 4,
        },
        signer="test_hotkey",
    )

    success = apply_extrinsic(state, commit_extrinsic)
    assert success is True

    # Step 2: Advance time to reveal round
    state.set_head(150)

    # Step 3: Reveal weights
    success = state.reveal_weights(
        netuid=1,
        mecid=1,
        hotkey="test_hotkey",
        weights=weights,
        salt=salt,
    )
    assert success is True

    # Step 4: Verify weights applied
    applied_weights = state.get_mechanism_weights(netuid=1, mecid=1)
    assert applied_weights == weights


def test_multiple_mechanisms_independent():
    """Test that multiple mechanisms can be updated independently."""
    state = SimulatorState()

    # Set weights for mechanism 0
    mec0_extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="set_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 0,
            "dests": [0, 1],
            "weights": [100, 200],
            "version_key": 0,
        },
        signer="signer1",
    )
    apply_extrinsic(state, mec0_extrinsic)

    # Set weights for mechanism 1
    mec1_extrinsic = DecodedExtrinsic(
        pallet="SubtensorModule",
        call="set_mechanism_weights",
        params={
            "netuid": 1,
            "mecid": 1,
            "dests": [0, 1],
            "weights": [300, 400],
            "version_key": 0,
        },
        signer="signer2",
    )
    apply_extrinsic(state, mec1_extrinsic)

    # Verify both mechanisms have independent weights
    mec0_weights = state.get_mechanism_weights(netuid=1, mecid=0)
    mec1_weights = state.get_mechanism_weights(netuid=1, mecid=1)

    assert mec0_weights == {0: 100, 1: 200}
    assert mec1_weights == {0: 300, 1: 400}
