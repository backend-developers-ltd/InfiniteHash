from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import structlog
import websockets.asyncio.server
from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.utils import ss58
from websockets.exceptions import ConnectionClosed

from .extrinsics import apply_extrinsic, decode_extrinsic
from .http_api import HTTPError, SimulatorHTTPAPI
from .runtime import METADATA_V15_HEX, RUNTIME_VERSION, runtime_configuration
from .state import SimulatorState

try:
    import xxhash
except ImportError:  # pragma: no cover - ensure graceful fallback if xxhash missing
    xxhash = None

logger = structlog.get_logger(__name__)


# --- Metadata & Runtime -------------------------------------------------


def get_runtime_config_with_metadata() -> tuple[RuntimeConfiguration, Any]:
    return runtime_configuration()


_runtime_config, _metadata = get_runtime_config_with_metadata()


def _lookup_storage_type(pallet_name: str, storage_name: str) -> str | None:
    if _metadata is None:
        raise RuntimeError("Metadata not loaded; cannot resolve storage type")  # explicit failure
    pallet = _metadata.get_metadata_pallet(pallet_name)
    storage_fn = pallet.get_storage_function(storage_name)
    return storage_fn.get_value_type_string()


class ScaleCodec:
    def __init__(self, runtime_config: RuntimeConfiguration):
        self._runtime_config = runtime_config

    def encode_hex(self, type_string: str, value: Any) -> str:
        obj = self._runtime_config.create_scale_object(type_string)
        obj.encode(value)
        data = obj.data.data
        return "0x" + data.hex()

    def decode_u16(self, data_hex: str | bytes | None, *, default: int = 1) -> int:
        if not data_hex:
            return default
        if isinstance(data_hex, str):
            data_bytes = bytes.fromhex(data_hex.removeprefix("0x"))
        else:
            data_bytes = bytes(data_hex)
        decoder = self._runtime_config.create_scale_object("u16", ScaleBytes(data_bytes))
        decoder.decode()
        return int(decoder.value)


_codec = ScaleCodec(_runtime_config)


def _require_xxhash() -> None:
    if xxhash is None:
        raise RuntimeError("xxhash module is required for Twox hashing but is not installed")


def _twox64(data: bytes, seed: int) -> bytes:
    _require_xxhash()
    # xxhash digests are big-endian; reverse to match SCALE little-endian encoding
    return xxhash.xxh64(data, seed=seed).digest()[::-1]


def _twox128(data: bytes) -> bytes:
    return _twox64(data, 0) + _twox64(data, 1)


def _twox64_concat(data: bytes) -> bytes:
    return _twox64(data, 0) + data


_MECH_EMISSION_SPLIT_TYPE = _lookup_storage_type("SubtensorModule", "MechanismEmissionSplit")

if xxhash is not None:
    _SUBTENSOR_PREFIX = _twox128(b"SubtensorModule")
    _MECH_EMISSION_SPLIT_PREFIX_HEX = (_SUBTENSOR_PREFIX + _twox128(b"MechanismEmissionSplit")).hex()
    _MECH_COUNT_PREFIX_HEX = (_SUBTENSOR_PREFIX + _twox128(b"MechanismCountCurrent")).hex()
else:  # pragma: no cover - only when xxhash missing
    _SUBTENSOR_PREFIX = b""
    _MECH_EMISSION_SPLIT_PREFIX_HEX = ""
    _MECH_COUNT_PREFIX_HEX = ""


def _twox64concat_storage_key(prefix_hex: str, key_bytes: bytes) -> str:
    if not prefix_hex:
        return ""
    hashed = _twox64_concat(key_bytes)
    return "0x" + prefix_hex + hashed.hex()


def _extract_twox64concat_netuid(storage_hex: str, prefix_hex: str) -> int | None:
    key_hex = storage_hex.removeprefix("0x")
    if not prefix_hex or not key_hex.startswith(prefix_hex):
        return None
    remainder = key_hex[len(prefix_hex) :]
    if len(remainder) < 20:
        return None
    key_bytes_hex = remainder[16:20]
    try:
        key_bytes = bytes.fromhex(key_bytes_hex)
    except ValueError:
        return None
    if len(key_bytes) != 2:
        return None
    return int.from_bytes(key_bytes, "little")


def _account_to_hex(value: Any) -> str:
    if not value:
        return "0x" + ("00" * 32)
    try:
        if isinstance(value, bytes):
            data = bytes(value)
        elif isinstance(value, str):
            if value.startswith("0x"):
                data = bytes.fromhex(value[2:])
            else:
                decoded = ss58.ss58_decode(value)
                if isinstance(decoded, bytes):
                    data = decoded
                else:
                    hex_value = str(decoded)
                    data = bytes.fromhex(hex_value.removeprefix("0x"))
        else:
            raise TypeError
    except Exception:  # noqa: BLE001 - fallback to zero account for invalid inputs
        logger.warning("Invalid account identifier provided", account=value)
        return "0x" + ("00" * 32)

    if len(data) != 32:
        data = (data + b"\x00" * 32)[:32]
    return "0x" + data.hex()


def _coerce_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return int(value)
    raise TypeError(f"Cannot coerce {value!r} to int")


def _extract_param(params: Any, key: str, index: int | None = None) -> Any:
    if isinstance(params, dict):
        return params.get(key)
    if index is not None and isinstance(params, list | tuple) and len(params) > index:
        return params[index]
    return None


def _lookup_block(state: SimulatorState, block_hash: str | None) -> tuple[int | None, Any]:
    block_number = state.block_number_for_hash(block_hash)
    if block_number is None:
        return None, state.head()
    try:
        return block_number, state.get_block(block_number)
    except KeyError:
        return None, state.head()


# --- Runtime API builders -----------------------------------------------


@dataclass(frozen=True)
class RuntimeApiSpec:
    type_string: str
    builder: Callable[[SimulatorState, int, str | None], Any]


_DEFAULT_SUBNET_HYPERPARAMS = {
    "rho": 13,
    "kappa": 32767,
    "immunity_period": 7200,
    "min_allowed_weights": 8,
    "max_weights_limit": 455,
    "tempo": 30,
    "min_difficulty": 10_000_000,
    "max_difficulty": 0xFFFFFFFFFFFFFFFF,
    "weights_version": 2013,
    "weights_rate_limit": 0,
    "adjustment_interval": 100,
    "activity_cutoff": 5000,
    "registration_allowed": True,
    "target_regs_per_interval": 1,
    "min_burn": 1_000_000_000,
    "max_burn": 100_000_000_000,
    "bonds_moving_avg": 900_000,
    "max_regs_per_block": 1,
    "serving_rate_limit": 0,
    "max_validators": 128,
    "adjustment_alpha": 0,
    "difficulty": 10_000_000,
    "commit_reveal_period": 3,
    "commit_reveal_weights_enabled": True,
    "alpha_high": 58_982,
    "alpha_low": 45_875,
    "liquid_alpha_enabled": False,
}


def _build_hyperparams(state: SimulatorState, netuid: int, _block_hash: str | None) -> dict[str, Any]:
    subnet = state.ensure_subnet(netuid)
    payload = dict(_DEFAULT_SUBNET_HYPERPARAMS)
    payload["tempo"] = subnet.tempo
    return payload


def _build_dynamic_info(state: SimulatorState, netuid: int, _block_hash: str | None) -> dict[str, Any]:
    subnet = state.ensure_subnet(netuid)
    owner_hotkey, owner_coldkey = state.get_subnet_owner(netuid)

    return {
        "netuid": netuid,
        "owner_hotkey": _account_to_hex(owner_hotkey),
        "owner_coldkey": _account_to_hex(owner_coldkey),
        "subnet_name": [],
        "token_symbol": [],
        "tempo": subnet.tempo,
        "last_step": 0,
        "blocks_since_last_step": 0,
        "emission": 1_000_000_000,
        "alpha_in": 0,
        "alpha_out": 0,
        "tao_in": 0,
        "alpha_out_emission": 0,
        "alpha_in_emission": 0,
        "tao_in_emission": 0,
        "pending_alpha_emission": 0,
        "pending_root_emission": 0,
        "subnet_volume": 0,
        "network_registered_at": 0,
        "subnet_identity": None,
        "moving_price": 0,
    }


def _build_neurons_lite(state: SimulatorState, netuid: int, block_hash: str | None) -> list[dict[str, Any]]:
    block_number = state.block_number_for_hash(block_hash)
    neurons = state.neurons(netuid, block_number=block_number)
    encoded: list[dict[str, Any]] = []

    for idx, neuron in enumerate(neurons):
        hotkey_hex = _account_to_hex(neuron.get("hotkey"))
        coldkey_hex = _account_to_hex(neuron.get("coldkey", neuron.get("hotkey")))
        stake_value = neuron.get("stake", 0)
        stake_rao = int(stake_value * 1_000_000_000) if isinstance(stake_value, float) else int(stake_value)

        axon_defaults = {
            "block": 0,
            "version": 0,
            "ip": 0,
            "port": 0,
            "ip_type": 0,
            "protocol": 0,
            "placeholder1": 0,
            "placeholder2": 0,
        }
        axon_info = {**axon_defaults, **neuron.get("axon_info", {})}

        prometheus_defaults = {
            "block": 0,
            "version": 0,
            "ip": 0,
            "port": 0,
            "ip_type": 0,
        }
        prometheus_info = {**prometheus_defaults, **neuron.get("prometheus_info", {})}

        encoded.append(
            {
                "hotkey": hotkey_hex,
                "coldkey": coldkey_hex,
                "uid": int(neuron.get("uid", idx)),
                "netuid": netuid,
                "active": bool(neuron.get("active", True)),
                "axon_info": axon_info,
                "prometheus_info": prometheus_info,
                "stake": [[coldkey_hex, stake_rao]],
                "rank": int(neuron.get("rank", 0)),
                "emission": int(neuron.get("emission", 0)),
                "incentive": int(neuron.get("incentive", 0)),
                "consensus": int(neuron.get("consensus", 0)),
                "trust": int(neuron.get("trust", 0)),
                "validator_trust": int(neuron.get("validator_trust", 0)),
                "dividends": int(neuron.get("dividends", 0)),
                "last_update": int(neuron.get("last_update", 0)),
                "validator_permit": bool(neuron.get("validator_permit", False)),
                "pruning_score": int(neuron.get("pruning_score", 0)),
            }
        )

    return encoded


_RUNTIME_API_HANDLERS: dict[str, RuntimeApiSpec] = {
    "SubnetInfoRuntimeApi_get_subnet_hyperparams": RuntimeApiSpec("scale_info::604", _build_hyperparams),
    "SubnetInfoRuntimeApi_get_dynamic_info": RuntimeApiSpec("scale_info::610", _build_dynamic_info),
    "NeuronInfoRuntimeApi_get_neurons_lite": RuntimeApiSpec("scale_info::592", _build_neurons_lite),
}


# --- Commitment helpers -------------------------------------------------

COMMITMENT_STORAGE_PREFIX = "ca407206ec1ab726b2636c4b145ac287419a60ae8b01e6dcaebd7317e43c69bf"


def _is_commitment_key(key: str) -> bool:
    return COMMITMENT_STORAGE_PREFIX in key


def _extract_netuid_from_key(key: str) -> int | None:
    key_hex = key.removeprefix("0x")
    if len(key_hex) < 68:
        return None
    netuid_bytes = bytes.fromhex(key_hex[64:68])
    return int.from_bytes(netuid_bytes, "little")


def _decode_commitment_storage_key(storage_key: str) -> tuple[int, str] | None:
    try:
        key_bytes = bytes.fromhex(storage_key.removeprefix("0x"))
        expected_len = 32 + 2 + 8 + 32
        if len(key_bytes) != expected_len:
            return None
        netuid = int.from_bytes(key_bytes[32:34], "little")
        account_bytes = key_bytes[42:74]
        return netuid, ss58.ss58_encode(account_bytes)
    except Exception:
        logger.exception("Failed to decode commitment storage key", key=storage_key)
        return None


def _encode_commitment_payload(payload: bytes, block_number: int) -> str:
    if len(payload) > 136:
        raise ValueError("Commitment payload too large")
    result = bytearray()
    result.extend((0).to_bytes(8, "little"))  # deposit
    result.extend(int(block_number).to_bytes(4, "little"))  # block
    result.append(1 << 2)  # Vec length = 1
    variant_index = 1 + len(payload)
    result.append(variant_index)
    result.extend(payload)
    return "0x" + result.hex()


def _get_commitment_entry(
    state: SimulatorState,
    *,
    netuid: int,
    hotkey: str,
    block_number: int,
) -> tuple[int, bytes] | None:
    subnet = state.ensure_subnet(netuid)
    history = subnet.commitments_by_hotkey.get(hotkey)
    if not history:
        return None
    for commit_block, payload in reversed(history):
        if commit_block <= block_number:
            return commit_block, payload
    return None


# --- RPC registration ---------------------------------------------------

RPC_METHODS: dict[str, Callable[[SimulatorState, Any], Awaitable[Any]]] = {}


def rpc_method(name: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable[[SimulatorState, Any], Coroutine[Any, Any, Any]]):
        RPC_METHODS[name] = func
        return func

    return decorator


@rpc_method("chain_getHead")
async def rpc_chain_get_head(state: SimulatorState, params: Any) -> Any:  # noqa: ARG001
    return state.head().hash


@rpc_method("chain_getHeader")
async def rpc_chain_get_header(state: SimulatorState, params: Any) -> Any:
    block_hash = _extract_param(params, "hash", index=0)
    block_number, block = _lookup_block(state, block_hash)
    timestamp_ms = int(block.timestamp.timestamp() * 1000)
    return {
        "number": f"0x{block.number:x}",
        "parentHash": "0x" + "00" * 32,
        "stateRoot": "0x" + "00" * 32,
        "extrinsicsRoot": "0x" + "00" * 32,
        "digest": {"logs": []},
        "timestamp": timestamp_ms,
    }


@rpc_method("chain_getBlockHash")
async def rpc_chain_get_block_hash(state: SimulatorState, params: Any) -> Any:
    block_identifier = _extract_param(params, "hash", index=0)
    if block_identifier is None:
        return state.head().hash
    try:
        number = _coerce_int(block_identifier)
        return state.get_block(number).hash
    except Exception:
        import hashlib

        hash_input = f"block_{block_identifier}".encode()
        return "0x" + hashlib.sha256(hash_input).hexdigest()[:64]


@rpc_method("state_call")
async def rpc_state_call(state: SimulatorState, params: Any) -> Any:
    name = _extract_param(params, "name", index=0)
    if name == "Metadata_metadata_at_version":
        return METADATA_V15_HEX

    bytes_param = _extract_param(params, "bytes", index=1)
    block_hash = _extract_param(params, "hash", index=2)

    spec = _RUNTIME_API_HANDLERS.get(name)
    if not spec:
        return "0x00"

    netuid = _codec.decode_u16(bytes_param, default=1)
    payload = spec.builder(state, netuid, block_hash)
    return _codec.encode_hex(spec.type_string, payload)


@rpc_method("state_getRuntimeVersion")
async def rpc_state_get_runtime_version(state: SimulatorState, params: Any) -> Any:  # noqa: ARG001
    return RUNTIME_VERSION


@rpc_method("system_accountNextIndex")
async def rpc_system_account_next_index(state: SimulatorState, params: Any) -> Any:  # noqa: ARG001
    return 0


async def _commitment_keys_for_prefix(state: SimulatorState, key_prefix: str, block_hash: str | None) -> list[str]:
    netuid = _extract_netuid_from_key(key_prefix)
    if netuid is None:
        return []

    block_hash = block_hash or state.head().hash
    commitments = state.fetch_commitments(netuid, block_hash=block_hash)

    try:
        import xxhash
    except ImportError:  # pragma: no cover - xxhash should be available but guard regardless
        logger.warning("xxhash not available, skipping commitment key generation")
        return []

    prefix_hex = key_prefix.removeprefix("0x")
    keys: list[str] = []
    for hotkey in commitments.keys():
        account_hex = _account_to_hex(hotkey).removeprefix("0x")
        account_bytes = bytes.fromhex(account_hex)
        hashed = xxhash.xxh64(account_bytes).digest().hex()
        keys.append("0x" + prefix_hex + hashed + account_hex)
    return keys


@rpc_method("state_getKeys")
async def rpc_state_get_keys(state: SimulatorState, params: Any) -> Any:
    key_prefix = _extract_param(params, "prefix", index=0)
    block_hash = _extract_param(params, "hash", index=1)
    if not key_prefix:
        return []
    if _is_commitment_key(key_prefix):
        return await _commitment_keys_for_prefix(state, key_prefix, block_hash)
    if _MECH_EMISSION_SPLIT_PREFIX_HEX:
        prefix_hex = key_prefix.removeprefix("0x")
        if prefix_hex == _MECH_EMISSION_SPLIT_PREFIX_HEX:
            keys: list[str] = []
            for netuid in state.netuids():
                if state.mechanism_emission_split(netuid) is None:
                    continue
                key = _twox64concat_storage_key(_MECH_EMISSION_SPLIT_PREFIX_HEX, netuid.to_bytes(2, "little"))
                if key:
                    keys.append(key)
            return keys
        netuid = _extract_twox64concat_netuid(key_prefix, _MECH_EMISSION_SPLIT_PREFIX_HEX)
        if netuid is not None and state.mechanism_emission_split(netuid) is not None:
            key = _twox64concat_storage_key(_MECH_EMISSION_SPLIT_PREFIX_HEX, netuid.to_bytes(2, "little"))
            return [key] if key else []
    return []


@rpc_method("state_getKeysPaged")
async def rpc_state_get_keys_paged(state: SimulatorState, params: Any) -> Any:
    key_prefix = _extract_param(params, "key", index=0)
    block_hash = _extract_param(params, "hash", index=3)
    if not key_prefix:
        return []
    if _is_commitment_key(key_prefix):
        return await _commitment_keys_for_prefix(state, key_prefix, block_hash)
    if _MECH_EMISSION_SPLIT_PREFIX_HEX:
        prefix_hex = key_prefix.removeprefix("0x")
        if prefix_hex == _MECH_EMISSION_SPLIT_PREFIX_HEX:
            keys: list[str] = []
            for netuid in state.netuids():
                if state.mechanism_emission_split(netuid) is None:
                    continue
                key = _twox64concat_storage_key(_MECH_EMISSION_SPLIT_PREFIX_HEX, netuid.to_bytes(2, "little"))
                if key:
                    keys.append(key)
            return keys
        netuid = _extract_twox64concat_netuid(key_prefix, _MECH_EMISSION_SPLIT_PREFIX_HEX)
        if netuid is not None and state.mechanism_emission_split(netuid) is not None:
            key = _twox64concat_storage_key(_MECH_EMISSION_SPLIT_PREFIX_HEX, netuid.to_bytes(2, "little"))
            return [key] if key else []
    return []


@rpc_method("state_queryStorageAt")
async def rpc_state_query_storage_at(state: SimulatorState, params: Any) -> Any:
    keys = _extract_param(params, "keys", index=0) or []
    block_hash = _extract_param(params, "hash", index=1) or state.head().hash
    if not keys:
        return []

    key = keys[0]
    if not _is_commitment_key(key):
        return []

    netuid = _extract_netuid_from_key(key)
    if netuid is None:
        return []

    block_number, _ = _lookup_block(state, block_hash)
    if block_number is None:
        block_number = state.head().number
    changes: list[list[str]] = []
    for storage_key in keys:
        decoded = _decode_commitment_storage_key(storage_key)
        if not decoded:
            continue
        _, hotkey = decoded
        entry = _get_commitment_entry(state, netuid=netuid, hotkey=hotkey, block_number=block_number)
        if not entry:
            continue
        commit_block, payload_bytes = entry
        value_hex = _encode_commitment_payload(payload_bytes, commit_block)
        changes.append([storage_key, value_hex])

    if not changes:
        return []
    return [{"block": block_hash, "changes": changes}]


@rpc_method("state_getStorage")
async def rpc_state_get_storage(state: SimulatorState, params: Any) -> Any:
    storage_key = _extract_param(params, "key", index=0)
    block_hash = _extract_param(params, "hash", index=1)
    if not storage_key:
        return None

    TIMESTAMP_NOW_KEY = "0xf0c365c3cf59d671eb72da0e7a4113c49f1f0515f462cdcf84e0f1d6045dfcbb"
    SYSTEM_EVENTS_KEY = "0x26aa394eea5630e07c48ae0c9558cef780d41e5e16056765bc8461851072c9d7"

    if storage_key == TIMESTAMP_NOW_KEY:
        block_hash = block_hash or state.head().hash
        _, block = _lookup_block(state, block_hash)
        timestamp_ms = int(block.timestamp.timestamp() * 1000)
        return "0x" + timestamp_ms.to_bytes(8, "little").hex()

    if storage_key == SYSTEM_EVENTS_KEY:
        return "0x00"

    if _is_commitment_key(storage_key):
        decoded = _decode_commitment_storage_key(storage_key)
        if not decoded:
            return None
        netuid, hotkey = decoded
        block_hash = block_hash or state.head().hash
        block_number, _ = _lookup_block(state, block_hash)
        if block_number is None:
            block_number = state.head().number
        entry = _get_commitment_entry(state, netuid=netuid, hotkey=hotkey, block_number=block_number)
        if not entry:
            return None
        commit_block, payload_bytes = entry
        return _encode_commitment_payload(payload_bytes, commit_block)
    if _MECH_EMISSION_SPLIT_PREFIX_HEX and _MECH_EMISSION_SPLIT_TYPE:
        netuid = _extract_twox64concat_netuid(storage_key, _MECH_EMISSION_SPLIT_PREFIX_HEX)
        if netuid is not None:
            split = state.mechanism_emission_split(netuid)
            # Handle None explicitly - SCALE encodes Option<T> None as 0x00
            if split is None:
                return "0x00"
            try:
                return _codec.encode_hex(_MECH_EMISSION_SPLIT_TYPE, split)
            except Exception:  # pragma: no cover - defensive guard for encoding issues
                logger.exception("Failed to encode mechanism emission split", netuid=netuid, split=split)
                return None
    if _MECH_COUNT_PREFIX_HEX:
        netuid = _extract_twox64concat_netuid(storage_key, _MECH_COUNT_PREFIX_HEX)
        if netuid is not None:
            count = state.mechanism_count(netuid)
            try:
                return _codec.encode_hex("u8", count)
            except Exception:  # pragma: no cover
                logger.exception("Failed to encode mechanism count", netuid=netuid, count=count)
                return None

    logger.warning("Unknown storage key requested", storage_key=storage_key[:80], block_hash=block_hash)
    return None


async def _send_extrinsic_events(websocket: Any, subscription_id: str, block_hash: str) -> None:
    await asyncio.sleep(0.01)
    events = [
        {"result": {"inBlock": block_hash}},
        {"result": {"finalized": block_hash}},
    ]
    for entry in events:
        event = {
            "jsonrpc": "2.0",
            "method": "author_extrinsicUpdate",
            "params": {"subscription": subscription_id, **entry},
        }
        await websocket.send(json.dumps(event))
        await asyncio.sleep(0.01)


@rpc_method("author_submitAndWatchExtrinsic")
async def rpc_author_submit_and_watch_extrinsic(
    state: SimulatorState,
    params: Any,
    websocket: Any | None = None,
    request_id: Any | None = None,  # noqa: ARG001
) -> Any:
    extrinsic_hex = _extract_param(params, "bytes", index=0) or _extract_param(params, "extrinsic", index=0)
    if isinstance(params, list | tuple) and not extrinsic_hex:
        extrinsic_hex = params[0]

    if extrinsic_hex is None:
        raise ValueError("Missing extrinsic bytes")

    state.submitted_extrinsics.append(extrinsic_hex)

    try:
        decoded = decode_extrinsic(extrinsic_hex, runtime_config=_runtime_config, metadata=_metadata)
        apply_extrinsic(state, decoded)
    except Exception:
        logger.exception("Failed to process extrinsic")

    subscription_id = "0x01"
    if websocket is not None:
        asyncio.create_task(_send_extrinsic_events(websocket, subscription_id, state.head().hash))
    return subscription_id


@rpc_method("author_unwatchExtrinsic")
async def rpc_author_unwatch_extrinsic(state: SimulatorState, params: Any) -> Any:  # noqa: ARG001
    return True


@rpc_method("chain_getBlock")
async def rpc_chain_get_block(state: SimulatorState, params: Any) -> Any:
    block_hash = _extract_param(params, "hash", index=0)
    _, block = _lookup_block(state, block_hash)
    extrinsics = list(state.submitted_extrinsics)
    return {
        "block": {
            "header": {
                "number": f"0x{block.number:x}",
                "parentHash": "0x" + "00" * 32,
                "stateRoot": "0x" + "00" * 32,
                "extrinsicsRoot": "0x" + "00" * 32,
                "digest": {"logs": []},
            },
            "extrinsics": extrinsics,
        }
    }


# --- HTTP API -----------------------------------------------------------


class SimulatorHTTPRequestHandler(BaseHTTPRequestHandler):
    state: SimulatorState
    api: SimulatorHTTPAPI

    server_version = "SubtensorSim/0.1"

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return {}
        length = int(length_header)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPError(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        segments = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        try:
            payload = self.api.handle_get(segments, query)
        except HTTPError as exc:
            self._send_json(exc.payload, status=exc.status)
            return
        self._send_json(payload)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        segments = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        try:
            body = self._read_json()
        except HTTPError as exc:
            self._send_json(exc.payload, status=exc.status)
            return
        try:
            payload = self.api.handle_post(segments, query, body)
        except HTTPError as exc:
            self._send_json(exc.payload, status=exc.status)
            return
        self._send_json(payload)

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        self._send_json({"error": "not implemented"}, status=HTTPStatus.NOT_FOUND)


def _build_handler(state: SimulatorState) -> Callable[[Any, Any, Any], SimulatorHTTPRequestHandler]:
    api = SimulatorHTTPAPI(state)

    class _Handler(SimulatorHTTPRequestHandler):
        pass

    _Handler.state = state
    _Handler.api = api
    return _Handler


class RPCServer:
    def __init__(self, state: SimulatorState):
        self.state = state

    async def handler(self, websocket):
        logger.info("RPC client connected", address=websocket.remote_address)
        try:
            async for message in websocket:
                response = await self._handle_message(message, websocket)
                if response is not None:
                    await websocket.send(json.dumps(response))
        except ConnectionClosed:
            logger.info("RPC client disconnected", address=websocket.remote_address)
        except Exception:
            logger.exception("RPC server handler error")

    async def _handle_message(self, message: str, websocket) -> dict[str, Any] | None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            }

        method = data.get("method")
        params = data.get("params")
        request_id = data.get("id")

        handler = RPC_METHODS.get(method)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method not found: {method}"},
                "id": request_id,
            }

        try:
            if method == "author_submitAndWatchExtrinsic":
                result = await handler(self.state, params, websocket, request_id)
            else:
                result = await handler(self.state, params)
            return {"jsonrpc": "2.0", "result": result, "id": request_id}
        except Exception as exc:  # noqa: BLE001 - surface handler errors as JSON-RPC failures
            logger.exception("Error processing RPC message", method=method)
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {exc}"},
                "id": request_id,
            }


async def run_http_server(host: str, port: int, state: SimulatorState, stop_event: asyncio.Event) -> None:
    import threading

    handler = _build_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True

    def serve() -> None:
        logger.info("Simulator HTTP server running", host=host, port=port)
        httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    await asyncio.sleep(0.1)
    await stop_event.wait()

    logger.info("Shutting down HTTP server")
    httpd.shutdown()
    thread.join(timeout=2.0)


async def run_rpc_server(host: str, port: int, state: SimulatorState, stop_event: asyncio.Event) -> None:
    rpc_server = RPCServer(state)
    logger.info("Simulator RPC server running", host=host, port=port)
    logger.info("Registered RPC methods", methods=sorted(RPC_METHODS.keys()))
    async with websockets.asyncio.server.serve(
        rpc_server.handler,
        host,
        port,
        open_timeout=60,
        close_timeout=60,
        ping_interval=None,
    ) as server:
        await stop_event.wait()
        server.close()
        await server.wait_closed()


async def run_servers(
    http_host: str,
    http_port: int,
    rpc_host: str,
    rpc_port: int,
    state: SimulatorState | None = None,
) -> None:
    state = state or SimulatorState()
    stop_event = asyncio.Event()
    http_server = run_http_server(http_host, http_port, state, stop_event)
    rpc_server = run_rpc_server(rpc_host, rpc_port, state, stop_event)

    try:
        await asyncio.gather(http_server, rpc_server)
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        logger.info("Stopping simulator servers")
    finally:
        stop_event.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Subtensor simulator server")
    parser.add_argument("--http-host", default="127.0.0.1", help="Host interface for HTTP server")
    parser.add_argument("--http-port", type=int, default=8090, help="Listening port for HTTP server")
    parser.add_argument("--rpc-host", default="127.0.0.1", help="Host interface for RPC server")
    parser.add_argument("--rpc-port", type=int, default=9944, help="Listening port for RPC server")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        asyncio.run(
            run_servers(
                http_host=args.http_host,
                http_port=args.http_port,
                rpc_host=args.rpc_host,
                rpc_port=args.rpc_port,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Simulator servers stopped.")


if __name__ == "__main__":  # pragma: no cover
    main()
