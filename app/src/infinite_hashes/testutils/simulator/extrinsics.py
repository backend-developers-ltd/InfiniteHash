from __future__ import annotations

import dataclasses
import logging
from typing import Any

from scalecodec.base import RuntimeConfiguration, ScaleBytes

from .runtime import runtime_configuration
from .state import SimulatorState

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecodedExtrinsic:
    """Parsed extrinsic with extracted call information."""

    pallet: str
    call: str
    params: dict[str, Any]
    signer: str  # hotkey address


def decode_extrinsic(
    hex_data: str,
    runtime_config: RuntimeConfiguration | None = None,
    metadata: Any | None = None,
) -> DecodedExtrinsic:
    """
    Parse SCALE-encoded extrinsic to extract call info.

    Args:
        hex_data: Hex-encoded SCALE bytes (with or without 0x prefix)
        runtime_config: RuntimeConfiguration with portable registry loaded (optional)
        metadata: Metadata object for decoding call parameters (optional)

    Returns:
        DecodedExtrinsic with pallet, call, params, and signer

    Raises:
        ValueError: If decoding fails
    """
    default_metadata = None
    if runtime_config is None:
        runtime_config, default_metadata = runtime_configuration()
    else:
        _, default_metadata = runtime_configuration()

    if metadata is None:
        metadata = default_metadata

    if hex_data.startswith("0x"):
        hex_data = hex_data[2:]

    try:
        scale_bytes = ScaleBytes(bytes.fromhex(hex_data))

        extrinsic_decoder = runtime_config.create_scale_object(
            "Extrinsic",
            scale_bytes,
            metadata=metadata,
        )
        extrinsic_decoder.decode()

        # Extract signer address from extrinsic
        signer = extrinsic_decoder.value.get("address", {})
        if isinstance(signer, dict):
            signer_address = signer.get("Id") or signer.get("AccountId") or str(signer)
        else:
            signer_address = str(signer)

        # Extract call data
        call_data = extrinsic_decoder.value.get("call", {})
        if isinstance(call_data, dict):
            pallet = call_data.get("call_module", "")
            call_name = call_data.get("call_function", "")
            call_args_raw = call_data.get("call_args", [])

            # Convert call_args from list to dict
            # call_args is a list of {'name': ..., 'value': ...} dicts
            if isinstance(call_args_raw, list):
                call_args = {arg["name"]: arg["value"] for arg in call_args_raw}
            elif isinstance(call_args_raw, dict):
                call_args = call_args_raw
            else:
                call_args = {}
        else:
            raise ValueError(f"Unexpected call data format: {call_data}")

        return DecodedExtrinsic(
            pallet=pallet,
            call=call_name,
            params=call_args,
            signer=signer_address,
        )
    except Exception as exc:
        raise ValueError(f"Failed to decode extrinsic: {exc}") from exc


def apply_extrinsic(state: SimulatorState, extrinsic: DecodedExtrinsic) -> bool:
    """
    Apply extrinsic effects to simulator state.

    Args:
        state: SimulatorState to modify
        extrinsic: Decoded extrinsic to apply

    Returns:
        True if extrinsic was successfully applied, False otherwise
    """
    try:
        if extrinsic.pallet == "SubtensorModule":
            if extrinsic.call == "set_mechanism_weights":
                return _apply_set_mechanism_weights(state, extrinsic)
            elif extrinsic.call == "commit_timelocked_mechanism_weights":
                return _apply_commit_timelocked_mechanism_weights(state, extrinsic)
            else:
                return False
        elif extrinsic.pallet == "Commitments":
            if extrinsic.call == "set_commitment":
                return _apply_set_commitment(state, extrinsic)
            else:
                return False
        else:
            return False
    except Exception as exc:
        logger.error(
            "Failed to apply extrinsic %s.%s: %s",
            extrinsic.pallet,
            extrinsic.call,
            exc,
            exc_info=True,
        )
        return False


def _apply_set_mechanism_weights(state: SimulatorState, extrinsic: DecodedExtrinsic) -> bool:
    """
    Apply set_mechanism_weights extrinsic.

    Expected params:
        - netuid: int
        - mecid: int (mechanism ID)
        - dests: list[int] (UIDs)
        - weights: list[int]
        - version_key: int
    """
    params = extrinsic.params
    netuid = int(params.get("netuid", 0))
    mecid = int(params.get("mecid", 0))
    dests = params.get("dests", [])
    weights = params.get("weights", [])

    if not isinstance(dests, list) or not isinstance(weights, list):
        logger.error("Invalid dests or weights format")
        return False

    if len(dests) != len(weights):
        logger.error("Mismatched dests/weights length")
        return False

    # Convert to dict[uid -> weight]
    weights_dict = {int(uid): int(weight) for uid, weight in zip(dests, weights)}

    state.set_mechanism_weights(netuid, mecid, weights_dict)

    logger.info(
        "Applied set_mechanism_weights: netuid=%s mecid=%s signer=%s weights=%s",
        netuid,
        mecid,
        extrinsic.signer,
        weights_dict,
    )

    return True


def _apply_commit_timelocked_mechanism_weights(
    state: SimulatorState,
    extrinsic: DecodedExtrinsic,
) -> bool:
    """
    Apply commit_timelocked_mechanism_weights extrinsic.

    Expected params:
        - netuid: int
        - mecid: int (mechanism ID)
        - commit: str or bytes (encrypted weight data from bittensor_drand)
        - reveal_round: int (block number when reveal is allowed)
        - commit_reveal_version: int
    """
    params = extrinsic.params
    netuid = int(params.get("netuid", 0))
    mecid = int(params.get("mecid", 0))
    commit_data = params.get("commit", "")
    reveal_round = int(params.get("reveal_round", 0))

    if not commit_data:
        logger.error("Missing commit data")
        return False

    if isinstance(commit_data, str):
        commit_hex = commit_data if commit_data.startswith("0x") else f"0x{commit_data}"
        try:
            encrypted_bytes = bytes.fromhex(commit_hex.removeprefix("0x"))
        except ValueError:
            logger.error("Commit data is not valid hex")
            return False
    elif isinstance(commit_data, bytes | bytearray):
        encrypted_bytes = bytes(commit_data)
        commit_hex = "0x" + encrypted_bytes.hex()
    else:
        logger.error("Invalid commit data type: %s", type(commit_data))
        return False

    # Get current block and subnet info for target_block calculation
    current_block = state.head().number
    subnet = state.ensure_subnet(netuid)
    tempo = subnet.tempo

    # Calculate target_block from reveal_round (drand round -> blockchain block)
    # We need to recalculate to get the target_block that corresponds to this reveal_round
    target_block = None
    try:
        from infinite_hashes.testutils.integration.mock_drand import calculate_reveal_round

        # Recalculate to get the target_block that matches this reveal_round
        # Use subnet_reveal_period_epochs=3 (hardcoded in hyperparams)
        _, calculated_target_block = calculate_reveal_round(
            tempo=tempo,
            current_block=current_block,
            netuid=netuid,
            subnet_reveal_period_epochs=3,
            block_time=12.0,
        )
        target_block = calculated_target_block
    except Exception as exc:
        logger.warning(f"Failed to calculate target_block from reveal_round: {exc}")
        # Fallback: assume reveal_round is close to block number (wrong but better than nothing)
        target_block = None

    # Decrypt the commit using mock_drand
    decrypted_weights = None
    try:
        from infinite_hashes.testutils.integration.mock_drand import decrypt, parse_decrypted_commit

        # Decrypt (decompress) the data
        decrypted_data = decrypt(encrypted_bytes, no_errors=False)
        if decrypted_data is None:
            logger.warning("Failed to decrypt commit data")
        else:
            # Parse the JSON payload
            parsed = parse_decrypted_commit(decrypted_data)

            # Extract weights
            uids = parsed.get("uids", [])
            weights = parsed.get("weights", [])

            # Convert to dict
            decrypted_weights = {int(uid): int(weight) for uid, weight in zip(uids, weights)}
    except Exception as exc:
        logger.warning(f"Failed to decrypt/parse commit data: {exc}")

    # Convert signer from hex to SS58 if needed (for consistency with other extrinsics)
    from scalecodec.utils import ss58

    if extrinsic.signer.startswith("0x"):
        # Signer is in hex format, convert to SS58
        signer_bytes = bytes.fromhex(extrinsic.signer.removeprefix("0x"))
        hotkey_ss58 = ss58.ss58_encode(signer_bytes)
    else:
        # Already in SS58 format
        hotkey_ss58 = extrinsic.signer

    state.commit_weights(
        netuid=netuid,
        mecid=mecid,
        hotkey=hotkey_ss58,
        commit_hash=commit_hex,
        reveal_round=reveal_round,
        target_block=target_block,
        decrypted_weights=decrypted_weights,
        encrypted_data=encrypted_bytes,
    )

    return True


def _apply_set_commitment(state: SimulatorState, extrinsic: DecodedExtrinsic) -> bool:
    """
    Apply Commitments::set_commitment extrinsic.

    Expected params:
        - netuid: int
        - info: bytes or list (commitment data as Vec<u8>)
    """
    params = extrinsic.params
    netuid = int(params.get("netuid", 0))
    info = params.get("info", b"")

    # Convert info to bytes if needed
    if isinstance(info, dict):
        # SCALE decoder returned CommitmentInfo struct: {'fields': [{'RawN': '0x...'}]}
        # Extract hex data from the Data enum variants
        fields = info.get("fields", [])
        if not fields:
            logger.error("Empty fields in commitment info dict")
            return False

        # Concatenate all Raw data fields
        data_parts = []
        for field in fields:
            if isinstance(field, dict):
                # Each field is {'Raw14': '0x...'} or similar
                for variant_name, hex_value in field.items():
                    if variant_name.startswith("Raw") and hex_value:
                        # Extract hex data
                        hex_str = hex_value.removeprefix("0x") if isinstance(hex_value, str) else hex_value
                        data_parts.append(hex_str)

        if not data_parts:
            logger.error("No Raw data found in commitment fields")
            return False

        # Join all parts and convert to bytes
        combined_hex = "".join(data_parts)
        commitment_bytes = bytes.fromhex(combined_hex)

    elif isinstance(info, list):
        # List of u8 values
        commitment_bytes = bytes(info)
    elif isinstance(info, bytes | bytearray):
        commitment_bytes = bytes(info)
    elif isinstance(info, str):
        # Hex string
        commitment_bytes = bytes.fromhex(info.removeprefix("0x"))
    else:
        logger.error(f"Invalid commitment info type: {type(info)}")
        return False

    if not commitment_bytes:
        logger.error("Empty commitment data")
        return False

    # Convert to hex string for storage
    commitment_hex = "0x" + commitment_bytes.hex()

    # Get current block hash
    block = state.head()
    block_hash = block.hash

    # Convert signer from hex to SS58 if needed
    from scalecodec.utils import ss58

    if extrinsic.signer.startswith("0x"):
        # Signer is in hex format, convert to SS58
        signer_bytes = bytes.fromhex(extrinsic.signer.removeprefix("0x"))
        hotkey_ss58 = ss58.ss58_encode(signer_bytes)
    else:
        # Already in SS58 format
        hotkey_ss58 = extrinsic.signer

    # Store commitment
    state.publish_commitment(
        netuid=netuid,
        block_hash=block_hash,
        hotkey=hotkey_ss58,
        payload=commitment_hex,
    )

    return True
