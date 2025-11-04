"""Mock Drand implementation for testing without external dependencies.

Simplifies encryption to just JSON + compression since the real implementation
is in Rust and not easily accessible for parsing.
"""

import json
import time
import zlib

# Drand Quicknet constants (from bittensor_drand Rust code)
DRAND_GENESIS_TIME = 1692803367  # Quicknet genesis timestamp
DRAND_PERIOD = 3  # 3-second rounds
SECURITY_BLOCK_OFFSET = 3  # Blocks to add after reveal epoch starts


def calculate_reveal_round(
    tempo: int,
    current_block: int,
    netuid: int,
    subnet_reveal_period_epochs: int,
    block_time: float,
) -> tuple[int, int]:
    """Calculate reveal round and block using bittensor_drand logic.

    This matches the Rust implementation in bittensor_drand::generate_commit
    and subtensor's get_epoch_index function.

    Returns:
        (reveal_round, target_block): Drand round number and blockchain block number
    """
    # 1. Calculate reveal epoch using simple division (matches subtensor's get_epoch_index)
    tempo_plus_one = tempo + 1
    netuid_plus_one = netuid + 1

    current_epoch = (current_block + netuid_plus_one) // tempo_plus_one
    reveal_epoch = current_epoch + subnet_reveal_period_epochs

    # 2. First block inside the reveal epoch
    first_reveal_blk = (reveal_epoch * tempo_plus_one) - netuid_plus_one

    # 3. Target block (first reveal block + security offset)
    target_ingest_blk = first_reveal_blk + SECURITY_BLOCK_OFFSET
    blocks_until_ingest = target_ingest_blk - current_block
    secs_until_ingest = blocks_until_ingest * block_time

    # 4. Convert to Drand round (round down to never request future pulse)
    now_secs = time.time()
    target_secs = now_secs + secs_until_ingest
    reveal_round = int((target_secs - DRAND_GENESIS_TIME) / DRAND_PERIOD)

    if reveal_round < 1:
        reveal_round = 1

    return reveal_round, target_ingest_blk


def encrypt(data: bytes, n_blocks: int, block_time: int | float = 12.0) -> tuple[bytes, int]:
    """Mock version of bittensor_drand.encrypt.

    Simplified: just compresses data (no actual encryption for testing).

    Arguments:
        data: The binary data to encrypt.
        n_blocks: Number of blocks until the data should be revealed.
        block_time: Amount of time in seconds for one block. Defaults to 12 seconds.

    Returns:
        encrypted_data (bytes): Compressed data.
        target_round (int): Mock reveal round (calculated from blocks).
    """
    # Calculate mock target round
    mock_current_round = int(time.time() / DRAND_PERIOD)
    rounds_until_reveal = int((n_blocks * block_time) / DRAND_PERIOD)
    target_round = mock_current_round + rounds_until_reveal

    if target_round < 1:
        target_round = 1

    # Just compress (no encryption for testing)
    compressed = zlib.compress(data)
    return compressed, target_round


def decrypt(encrypted_data: bytes, no_errors: bool = True) -> bytes | None:
    """Mock version of bittensor_drand.decrypt.

    Simplified: just decompresses data (no actual decryption for testing).

    Arguments:
        encrypted_data: The encrypted data to decrypt.
        no_errors: If True, returns None instead of raising exceptions when decryption fails.

    Returns:
        decrypted_data (Optional[bytes]): The decrypted data if successful, None otherwise.
    """
    try:
        decompressed = zlib.decompress(encrypted_data)
        return decompressed
    except Exception as e:
        if no_errors:
            return None
        raise ValueError(f"Mock decryption failed: {e}")


def get_latest_round() -> int:
    """Mock version of bittensor_drand.get_latest_round.

    Returns a deterministic mock round number instead of fetching from Drand network.

    Returns:
        round (int): Mock Drand round number based on current time.
    """
    return int((time.time() - DRAND_GENESIS_TIME) / DRAND_PERIOD)


def get_encrypted_commit(
    uids: list[int],
    weights: list[int],
    version_key: int,
    tempo: int,
    current_block: int,
    netuid: int,
    subnet_reveal_period_epochs: int,
    block_time: int | float,
    hotkey: bytes,
) -> tuple[bytes, int]:
    """Simplified mock of bittensor_drand.get_encrypted_commit.

    Uses JSON + compression instead of complex Rust serialization.

    Arguments:
        uids: The uids to commit.
        weights: The weights associated with the uids.
        version_key: The version key to use for committing and revealing.
        tempo: Number of blocks in one epoch.
        current_block: The current block number in the network.
        netuid: The network unique identifier (NetUID) for the subnet.
        subnet_reveal_period_epochs: Number of epochs after which reveal will be performed.
        block_time: Amount of time in seconds for one block.
        hotkey: The hotkey of a neuron-committer (wallet.hotkey.public_key).

    Returns:
        commit (bytes): Compressed JSON payload.
        target_round (int): Mock reveal round.
    """
    # Calculate reveal round using proper logic
    reveal_round, target_block = calculate_reveal_round(
        tempo=tempo,
        current_block=current_block,
        netuid=netuid,
        subnet_reveal_period_epochs=subnet_reveal_period_epochs,
        block_time=block_time,
    )

    # Create payload as JSON
    payload = {
        "hotkey": hotkey.hex() if isinstance(hotkey, bytes) else hotkey,
        "uids": uids,
        "weights": weights,
        "version_key": version_key,
        "netuid": netuid,
    }

    # Serialize and compress
    json_bytes = json.dumps(payload).encode("utf-8")
    compressed = zlib.compress(json_bytes)

    return compressed, reveal_round


def parse_decrypted_commit(data: bytes) -> dict:
    """Parse decrypted commitment data from JSON.

    Arguments:
        data: Decrypted/decompressed commitment bytes.

    Returns:
        Dictionary with hotkey, uids, weights, version_key, netuid.
    """
    json_str = data.decode("utf-8")
    return json.loads(json_str)


def install_mock():
    """Install mock Drand by monkey-patching bittensor_drand module.

    Replaces functions with simplified JSON-based versions for testing.

    Call this at the start of test processes to enable offline testing.
    """
    import sys

    import bittensor_drand

    # Replace with our simplified mocks
    bittensor_drand.encrypt = encrypt
    bittensor_drand.decrypt = decrypt
    bittensor_drand.get_latest_round = get_latest_round
    bittensor_drand.get_encrypted_commit = get_encrypted_commit

    # Patch the internal compiled functions as well (from the .so file)
    if hasattr(bittensor_drand, "_encrypt"):
        bittensor_drand._encrypt = encrypt
    if hasattr(bittensor_drand, "_decrypt"):
        bittensor_drand._decrypt = decrypt
    if hasattr(bittensor_drand, "_get_latest_round"):
        bittensor_drand._get_latest_round = get_latest_round
    if hasattr(bittensor_drand, "_get_encrypted_commit"):
        bittensor_drand._get_encrypted_commit = get_encrypted_commit

    # Also update sys.modules to ensure all references use the mock
    sys.modules["bittensor_drand"] = bittensor_drand

    # Log what we've patched for debugging
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Mock Drand installed: get_encrypted_commit=%s, encrypt=%s, decrypt=%s",
        bittensor_drand.get_encrypted_commit.__name__,
        bittensor_drand.encrypt.__name__,
        bittensor_drand.decrypt.__name__,
    )
