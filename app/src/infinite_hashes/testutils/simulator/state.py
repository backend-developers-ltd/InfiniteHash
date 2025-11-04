from __future__ import annotations

import dataclasses
import datetime as dt
import threading
from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from .runtime import runtime_configuration

_RUNTIME_CONFIG, _METADATA = runtime_configuration()


def _parse_scale_info_type_id(type_string: str) -> int | None:
    if not type_string or not type_string.startswith("scale_info::"):
        return None
    try:
        return int(type_string.split("::", 1)[1])
    except ValueError:
        return None


def _infer_mechanism_split_value_max() -> int:
    if _METADATA is None:
        raise RuntimeError("Metadata not loaded; cannot validate mechanism split type")

    pallet = _METADATA.get_metadata_pallet("SubtensorModule")
    storage_fn = pallet.get_storage_function("MechanismEmissionSplit")
    type_string = storage_fn.get_value_type_string()

    type_id = _parse_scale_info_type_id(type_string)
    if type_id is None:
        raise AssertionError(f"Unexpected MechanismEmissionSplit type string: {type_string!r}")

    types = _METADATA.value[1]["V15"]["types"]["types"]
    sequence_entry = types[type_id]
    seq_def = sequence_entry["type"]["def"].get("sequence")
    if seq_def is None:
        raise AssertionError("MechanismEmissionSplit value is not a sequence")

    element_id = seq_def["type"]
    element_entry = types[element_id]
    element_def = element_entry["type"]["def"]
    element_path = element_entry["type"]["path"]

    if "primitive" in element_def:
        primitive = element_def["primitive"]
        if primitive == "u16":
            result = 0xFFFF
        elif primitive == "u32":
            result = 0xFFFFFFFF
        else:
            raise AssertionError(f"Unsupported primitive mechanism split type: {primitive}")
    elif element_path and element_path[-1].lower() == "perbill":
        result = 1_000_000_000
    else:
        raise AssertionError(f"Unsupported mechanism split element type: {element_path or element_def}")

    if result != 0xFFFF:
        raise AssertionError(f"Mechanism split max {result} does not match expected 65535")

    return result


MECHANISM_SPLIT_VALUE_MAX = _infer_mechanism_split_value_max()

UTC = dt.UTC
DEFAULT_BLOCK_INTERVAL = dt.timedelta(seconds=12)

logger = structlog.get_logger(__name__)


def _parse_timestamp(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return dt.datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty timestamp string")
        if text.endswith("Z"):
            text = text[:-1]
            parsed = dt.datetime.fromisoformat(text)
            return parsed.replace(tzinfo=UTC)
        parsed = dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
        return parsed
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _format_timestamp(ts: dt.datetime) -> str:
    ts = ts.astimezone(UTC)
    return ts.isoformat().replace("+00:00", "Z")


def _default_block_hash(number: int) -> str:
    return f"0x{int(number).to_bytes(32, byteorder='big', signed=False).hex()}"


@dataclasses.dataclass(slots=True)
class BlockRecord:
    number: int
    hash: str
    timestamp: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "hash": self.hash,
            "timestamp": _format_timestamp(self.timestamp),
        }


@dataclasses.dataclass(slots=True)
class NeuronRecord:
    hotkey: str
    uid: int
    stake: float | int = 0
    registration_block: int = 0  # Block number when this neuron was registered
    extras: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "hotkey": self.hotkey,
            "uid": self.uid,
            "stake": self.stake,
            "registration_block": self.registration_block,
        }
        if self.extras:
            data.update(self.extras)
        return data


@dataclasses.dataclass
class WeightCommit:
    """Stores a weight commit for commit-reveal scheme."""

    commit_hash: str
    reveal_round: int  # Drand round number (stored on-chain)
    committed_at_block: int
    commit_reveal_version: int = 4  # CRV3
    target_block: int | None = None  # Blockchain block when weights should be revealed
    # Decrypted weights (stored immediately but only visible after target_block)
    decrypted_weights: dict[int, int] | None = None
    encrypted_data: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "reveal_round": self.reveal_round,
            "target_block": self.target_block,
            "committed_at_block": self.committed_at_block,
            "commit_reveal_version": self.commit_reveal_version,
            "has_decrypted_weights": self.decrypted_weights is not None,
        }


@dataclasses.dataclass
class SubnetState:
    netuid: int
    tempo: int = 29
    mechanism_count: int = 1
    mechanism_emission_split: list[int] | None = None
    # Subnet owner (for burn mechanism)
    owner_hotkey: str | None = None
    owner_coldkey: str | None = None
    # Legacy single weight set (kept for backward compatibility, points to mechanism 0)
    weights_by_uid: dict[int, int] = dataclasses.field(default_factory=dict)
    # Mechanism-specific weights: mecid -> {uid -> weight}
    weights_by_mechanism: dict[int, dict[int, int]] = dataclasses.field(default_factory=dict)
    # Mechanism weight history: mecid -> [(block_number, {uid -> weight})]
    weights_history_by_mechanism: dict[int, list[tuple[int, dict[int, int]]]] = dataclasses.field(default_factory=dict)
    # Commit-reveal storage: (mecid, hotkey) -> commit data
    weight_commits: dict[tuple[int, str], WeightCommit] = dataclasses.field(default_factory=dict)
    # Store commitments history per hotkey: hotkey -> list[(block_number, payload_bytes)]
    commitments_by_hotkey: dict[str, list[tuple[int, bytes]]] = dataclasses.field(default_factory=dict)
    # Block hash to block number mapping for commitment queries
    _block_hash_to_number: dict[str, int] = dataclasses.field(default_factory=dict)
    # Store ALL neuron registrations (history), not just latest
    neuron_registrations: list[NeuronRecord] = dataclasses.field(default_factory=list)

    @property
    def blocks_per_epoch(self) -> int:
        return self.tempo + 1

    def epoch_for(self, block_number: int) -> dict[str, int]:
        """Calculate epoch index using subtensor's get_epoch_index algorithm.

        This matches the logic from:
        https://github.com/opentensor/subtensor/blob/main/pallets/subtensor/src/subnets/weights.rs

        Note: This is different from the coinbase run algorithm (which determines
        when to run coinbase), but matches what bittensor_drand uses for commit/reveal.
        """
        netuid_plus_one = self.netuid + 1
        tempo_plus_one = self.tempo + 1
        block_with_offset = block_number + netuid_plus_one

        epoch_index = block_with_offset // tempo_plus_one if tempo_plus_one > 0 else 0
        remainder = block_with_offset % tempo_plus_one if tempo_plus_one > 0 else 0
        start = max(block_number - remainder - 1, 0)
        end = block_number - remainder + self.tempo

        return {"index": epoch_index, "start": start, "end": end}

    def to_dict(self) -> dict[str, Any]:
        # Convert commitment bytes to hex strings for JSON serialization
        commitments_serializable = {}
        for hotkey, history in self.commitments_by_hotkey.items():
            if not history:
                continue
            for blk, payload in history:
                serialize_history = commitments_serializable.setdefault(hotkey, [])
                if isinstance(payload, bytes):
                    serialize_history.append((blk, "0x" + payload.hex()))
                else:
                    serialize_history.append((blk, "0x" + str(payload)))

        return {
            "netuid": self.netuid,
            "tempo": self.tempo,
            "blocks_per_epoch": self.blocks_per_epoch,
            "mechanism_count": self.mechanism_count,
            "mechanism_emission_split": list(self.mechanism_emission_split) if self.mechanism_emission_split else None,
            "weights": dict(self.weights_by_uid),  # legacy mechanism 0
            "weights_by_mechanism": {mecid: dict(weights) for mecid, weights in self.weights_by_mechanism.items()},
            "weight_commits": {
                f"{mecid}:{hotkey}": commit.to_dict() for (mecid, hotkey), commit in self.weight_commits.items()
            },
            "commitments": commitments_serializable,
            "neurons": [n.to_dict() for n in self.neurons()],
        }

    def set_neurons(self, neurons: Iterable[Mapping[str, Any]], registration_block: int = 0) -> None:
        """Register neurons with registration block tracking.

        Args:
            neurons: Neurons to register
            registration_block: Block number when registration happens (default: 0 for initialization)
        """
        added = 0
        for idx, neuron in enumerate(neurons):
            hotkey = str(neuron["hotkey"])
            uid = int(neuron.get("uid", idx))
            stake = neuron.get("stake", 0)
            extras = {k: v for k, v in neuron.items() if k not in {"hotkey", "uid", "stake", "registration_block"}}

            # Append to history
            self.neuron_registrations.append(
                NeuronRecord(
                    hotkey=hotkey,
                    uid=uid,
                    stake=stake,
                    registration_block=registration_block,
                    extras=extras,
                )
            )
            added += 1

        logger.info(
            "subnet.set_neurons",
            netuid=self.netuid,
            added=added,
            registration_block=registration_block,
        )

    def neurons_at_block(self, block_number: int | None = None) -> list[NeuronRecord]:
        """Get active neurons at a specific block.

        For each UID, returns the neuron with highest registration_block <= block_number.
        Multiple registrations of same hotkey can happen; latest wins.

        Args:
            block_number: Block to query (None = all registered neurons, no filtering)

        Returns:
            List of active neurons at that block
        """
        if block_number is None:
            # No filtering - return latest for each UID
            block_number = float("inf")

        # Group by UID, taking highest registration_block <= block_number
        active_by_uid: dict[int, NeuronRecord] = {}

        for neuron in self.neuron_registrations:
            if neuron.registration_block > block_number:
                continue  # Future registration

            uid = neuron.uid
            if uid not in active_by_uid or neuron.registration_block > active_by_uid[uid].registration_block:
                active_by_uid[uid] = neuron

        return list(active_by_uid.values())

    def neurons(self, block_number: int | None = None) -> list[NeuronRecord]:
        """Get neurons at current state or specific block."""
        return self.neurons_at_block(block_number)

    @staticmethod
    def _decode_compact_u32(data: memoryview, pos: int) -> tuple[int, int]:
        if pos >= len(data):
            raise ValueError("compact integer missing data")

        first = data[pos]
        mode = first & 0b11
        if mode == 0:
            return first >> 2, pos + 1
        if mode == 1:
            if pos + 1 >= len(data):
                raise ValueError("compact integer truncated (mode=1)")
            value = (first >> 2) | (data[pos + 1] << 6)
            return value, pos + 2
        if mode == 2:
            if pos + 3 >= len(data):
                raise ValueError("compact integer truncated (mode=2)")
            value = (first >> 2) | (data[pos + 1] << 6) | (data[pos + 2] << 14) | (data[pos + 3] << 22)
            return value, pos + 4

        byte_len = (first >> 2) + 4
        end = pos + 1 + byte_len
        if end > len(data):
            raise ValueError("compact integer truncated (mode=3)")
        value = int.from_bytes(data[pos + 1 : end], "little")
        return value, end

    @staticmethod
    def _decode_commitment_chunks(payload: bytes) -> bytes:
        data_view = memoryview(payload)
        pos = 0
        fields, pos = SubnetState._decode_compact_u32(data_view, pos)
        if fields == 0:
            return b""

        parts: list[bytes] = []
        for _ in range(fields):
            if pos >= len(data_view):
                raise ValueError("commitment Data variant missing")
            variant = data_view[pos]
            pos += 1

            if variant == 0:
                # Data::None, nothing to append
                continue
            if variant <= 1:
                expected_len = 0
            else:
                expected_len = variant - 1

            end = pos + expected_len
            if end > len(data_view):
                raise ValueError("commitment Data payload truncated")
            if expected_len:
                parts.append(bytes(data_view[pos:end]))
            pos = end

        # If non-zero bytes remain we likely decoded the wrong structure.
        if pos < len(data_view) and any(data_view[pos:]):
            raise ValueError("unexpected trailing bytes after commitment decode")

        return b"".join(parts)

    @staticmethod
    def _maybe_decode_commitment(payload: bytes) -> bytes:
        # Try decoding as SCALE-encoded CommitmentInfo Vec<Data> (what the chain stores).
        try:
            return SubnetState._decode_commitment_chunks(payload)
        except ValueError:
            # Some call sites hand in raw UTF-8 already; fall back to original bytes.
            return payload

    @staticmethod
    def _coerce_commitment_bytes(payload: Any) -> bytes:
        if isinstance(payload, bytes | bytearray):
            data = bytes(payload)
        else:
            text = str(payload)
            if text.startswith("0x"):
                try:
                    data = bytes.fromhex(text[2:])
                except ValueError:
                    data = text.encode("utf-8", errors="ignore")
            else:
                data = text.encode("utf-8", errors="ignore")

        return SubnetState._maybe_decode_commitment(data)

    def publish_commitment(self, *, block_hash: str, block_number: int, hotkey: str, payload: Any) -> None:
        data = self._coerce_commitment_bytes(payload)
        history = self.commitments_by_hotkey.setdefault(hotkey, [])
        history.append((block_number, data))
        history.sort(key=lambda item: item[0])
        self._block_hash_to_number[block_hash] = block_number
        logger.info(
            "subnet.publish_commitment",
            netuid=self.netuid,
            hotkey=hotkey,
            block_hash=block_hash,
            block_number=block_number,
            payload_type=type(payload).__name__,
        )

    def set_commitments(self, block_hash: str, block_number: int, entries: Mapping[str, Any]) -> None:
        for hotkey, payload in entries.items():
            data = self._coerce_commitment_bytes(payload)
            self.commitments_by_hotkey[str(hotkey)] = [(block_number, data)]
        self._block_hash_to_number[block_hash] = block_number
        logger.info(
            "subnet.set_commitments",
            netuid=self.netuid,
            block_hash=block_hash,
            block_number=block_number,
            entries=len(entries),
        )

    def fetch_commitments(self, block_hash: str, block_number: int) -> dict[str, str]:
        """Fetch commitments from neurons active at the given block.

        Only returns commitments from neurons that are active at block_number.
        This handles UID replacement - old miner's commitment is ignored.
        """
        # Register this block hash -> number mapping
        self._block_hash_to_number[block_hash] = block_number

        # Get active neurons at this block
        active_neurons = self.neurons_at_block(block_number)
        active_hotkeys = {n.hotkey for n in active_neurons}

        if not active_hotkeys:
            fallback: dict[str, str] = {}
            for hotkey, history in self.commitments_by_hotkey.items():
                if not history:
                    continue
                payload = history[-1][1]
                if isinstance(payload, bytes):
                    fallback[hotkey] = "0x" + payload.hex()
                else:
                    fallback[hotkey] = str(payload)
            return fallback

        # Return commitments only from active neurons
        result = {}
        for hotkey, history in self.commitments_by_hotkey.items():
            if hotkey not in active_hotkeys or not history:
                continue

            # Find the latest commitment at or before block_number
            selected_payload = None
            for commit_block, payload in reversed(history):
                if commit_block <= block_number:
                    selected_payload = payload
                    break

            if selected_payload is None:
                continue

            if isinstance(selected_payload, bytes):
                result[hotkey] = "0x" + selected_payload.hex()
            else:
                result[hotkey] = str(selected_payload)

        return result

    def set_weights(self, weights: Mapping[int, int], *, block_number: int) -> None:
        """Legacy method - sets weights for mechanism 0."""
        normalized = {int(k): int(v) for k, v in weights.items()}
        self.weights_by_uid = dict(normalized)
        self.set_mechanism_weights(0, normalized, block_number=block_number)
        logger.info(
            "subnet.set_weights_legacy",
            netuid=self.netuid,
            entries=len(weights),
            block=block_number,
        )

    def set_mechanism_count(self, count: int) -> None:
        count = int(count)
        if count < 1 or count > 255:
            raise ValueError("mechanism_count must be between 1 and 255")
        previous = self.mechanism_count
        self.mechanism_count = count
        if self.mechanism_emission_split and len(self.mechanism_emission_split) > count:
            self.mechanism_emission_split = self.mechanism_emission_split[:count]
        logger.info(
            "subnet.set_mechanism_count",
            netuid=self.netuid,
            previous=previous,
            count=count,
        )

    def get_mechanism_count(self) -> int:
        return self.mechanism_count

    def set_mechanism_emission_split(self, split: Iterable[int] | None) -> None:
        if split is None:
            self.mechanism_emission_split = None
            logger.info("subnet.clear_mechanism_emission_split", netuid=self.netuid)
            return

        values: list[int] = []
        total = 0
        for value in split:
            ivalue = int(value)
            if ivalue < 0:
                raise ValueError("split values must be non-negative")
            if MECHANISM_SPLIT_VALUE_MAX is not None and ivalue > MECHANISM_SPLIT_VALUE_MAX:
                raise ValueError(f"split values must be <= {MECHANISM_SPLIT_VALUE_MAX}")
            values.append(ivalue)
            total += ivalue

        if not values:
            raise ValueError("split must contain at least one entry")
        self.mechanism_emission_split = values
        logger.info(
            "subnet.set_mechanism_emission_split",
            netuid=self.netuid,
            entries=len(values),
            total=total,
        )

    def get_mechanism_emission_split(self) -> list[int] | None:
        if self.mechanism_emission_split is None:
            return None
        return list(self.mechanism_emission_split)

    def get_mechanism_weights(
        self, mecid: int, at_block: int | None = None, hotkey: str | None = None
    ) -> dict[int, int]:
        """Get weights for a specific mechanism.

        Args:
            mecid: Mechanism ID
            at_block: If provided, includes auto-revealed weights from commits where target_block <= at_block.
            hotkey: When set together with at_block, overlays auto-revealed weights for this validator.

        Returns:
            Aggregate dict[uid, weight] representing the latest applied weights.
        """
        history = self.weights_history_by_mechanism.get(mecid, [])
        base_weights: dict[int, int] = {}
        if at_block is None:
            if history:
                base_weights = dict(history[-1][1])
        else:
            for block_number, entry_weights in history:
                if block_number > at_block:
                    break
                base_weights = dict(entry_weights)

        weights = dict(base_weights)

        # Collect auto-revealed commits that should be visible at the requested block
        revealed_by_hotkey: dict[str, dict[int, int]] = {}
        if at_block is not None:
            for (commit_mecid, commit_hotkey), commit in self.weight_commits.items():
                if commit_mecid != mecid:
                    continue
                target = commit.target_block
                if target is None or target > at_block:
                    continue
                if commit.decrypted_weights:
                    revealed_by_hotkey[commit_hotkey] = dict(commit.decrypted_weights)

        if hotkey is not None:
            override = revealed_by_hotkey.get(hotkey)
            if override:
                weights.update(override)
            return weights

        # Aggregate all revealed weights on top of the base state
        for override in revealed_by_hotkey.values():
            weights.update(override)
        return weights

    def set_mechanism_weights(self, mecid: int, weights: Mapping[int, int], *, block_number: int) -> None:
        """Set weights for a specific mechanism at a given block."""
        normalized = {int(k): int(v) for k, v in weights.items()}
        entry_weights = dict(normalized)
        self.weights_by_mechanism[mecid] = entry_weights

        history = self.weights_history_by_mechanism.setdefault(mecid, [])
        entry = (block_number, entry_weights)
        for idx, (existing_block, _) in enumerate(history):
            if block_number < existing_block:
                history.insert(idx, entry)
                break
            if block_number == existing_block:
                history[idx] = entry
                break
        else:
            history.append(entry)

        # Sync mechanism 0 to legacy weights_by_uid for backward compatibility
        if mecid == 0:
            self.weights_by_uid = dict(entry_weights)
        logger.info(
            "subnet.set_mechanism_weights",
            netuid=self.netuid,
            mecid=mecid,
            entries=len(weights),
            block=block_number,
        )

    def commit_weights(
        self,
        mecid: int,
        hotkey: str,
        commit_hash: str,
        reveal_round: int,
        committed_at_block: int,
        commit_reveal_version: int = 4,
        target_block: int | None = None,
        decrypted_weights: dict[int, int] | None = None,
        encrypted_data: bytes | None = None,
    ) -> None:
        """Store a weight commit with optional decrypted weights (for auto-reveal).

        Args:
            target_block: Blockchain block number when weights should be revealed.
                         If None, reveal_round is used as fallback (assumes it's a block number).
        """
        commit = WeightCommit(
            commit_hash=commit_hash,
            reveal_round=reveal_round,
            committed_at_block=committed_at_block,
            commit_reveal_version=commit_reveal_version,
            target_block=target_block if target_block is not None else reveal_round,
            decrypted_weights=decrypted_weights,
            encrypted_data=encrypted_data,
        )
        self.weight_commits[(mecid, hotkey)] = commit
        logger.info(
            "subnet.commit_weights",
            netuid=self.netuid,
            mecid=mecid,
            hotkey=hotkey,
            target_block=commit.target_block,
            reveal_round=reveal_round,
            decrypted=decrypted_weights is not None,
        )

    def reveal_weights(
        self,
        mecid: int,
        hotkey: str,
        weights: Mapping[int, int],
        salt: bytes,
        current_block: int,
    ) -> bool:
        """
        Reveal committed weights by validating hash and applying them.

        Returns True if reveal was valid and weights were applied, False otherwise.
        """
        key = (mecid, hotkey)
        commit = self.weight_commits.get(key)

        if commit is None:
            return False

        # Check if we're in the correct reveal round
        if current_block < commit.reveal_round:
            return False

        # Validate commit hash
        import hashlib

        weights_bytes = str(sorted(weights.items())).encode("utf-8")
        computed_hash = hashlib.sha256(weights_bytes + salt).hexdigest()

        if f"0x{computed_hash}" != commit.commit_hash.lower():
            return False

        # Apply the revealed weights
        self.set_mechanism_weights(mecid, weights, block_number=current_block)

        # Clean up the commit
        del self.weight_commits[key]

        logger.info(
            "subnet.reveal_weights",
            netuid=self.netuid,
            mecid=mecid,
            hotkey=hotkey,
            applied=True,
            entries=len(weights),
        )
        return True


class SimulatorState:
    def __init__(self):
        self._lock = threading.RLock()
        self._blocks: dict[int, BlockRecord] = {}
        self._head: int = 0
        self._subnets: dict[int, SubnetState] = {}
        self.submitted_extrinsics: list[str] = []
        self.set_head(0)

    # -- Block helpers -----------------------------------------------------

    def _last_timestamp(self) -> dt.datetime:
        if not self._blocks:
            return dt.datetime.now(tz=UTC)
        return max(self._blocks.values(), key=lambda b: b.number).timestamp

    def set_block(
        self,
        number: int,
        *,
        timestamp: Any | None = None,
        block_hash: str | None = None,
    ) -> BlockRecord:
        with self._lock:
            if timestamp is None:
                if self._blocks:
                    timestamp = self._blocks.get(number - 1, None)
                    if isinstance(timestamp, BlockRecord):
                        timestamp = timestamp.timestamp + DEFAULT_BLOCK_INTERVAL
                    else:
                        timestamp = self._last_timestamp() + DEFAULT_BLOCK_INTERVAL
                else:
                    timestamp = dt.datetime.now(tz=UTC)
            else:
                timestamp = _parse_timestamp(timestamp)
            if block_hash is None:
                block_hash = _default_block_hash(number)
            record = BlockRecord(number=number, hash=block_hash, timestamp=timestamp)
            self._blocks[number] = record
            if number >= self._head:
                self._head = number
            logger.info(
                "state.set_block",
                number=number,
                hash=block_hash,
                timestamp=record.timestamp.isoformat(),
            )
            return record

    def get_block(self, number: int) -> BlockRecord:
        try:
            return self._blocks[number]
        except KeyError as exc:
            raise KeyError(f"Unknown block: {number}") from exc

    def block_number_for_hash(self, block_hash: str | None) -> int | None:
        """Return the block number for a given hash, or None if not found."""
        if not block_hash:
            return None
        for number, record in self._blocks.items():
            if record.hash == block_hash:
                return number
        return None

    def set_head(self, number: int, *, timestamp: Any | None = None, block_hash: str | None = None) -> BlockRecord:
        previous = self._head
        record = self.set_block(number, timestamp=timestamp, block_hash=block_hash)
        with self._lock:
            self._head = number
        logger.info(
            "state.set_head",
            number=number,
            previous=previous,
            hash=record.hash,
        )
        return record

    def head(self) -> BlockRecord:
        with self._lock:
            return self._blocks[self._head]

    def advance_head(
        self,
        steps: int = 1,
        *,
        timestamps: Iterable[Any] | None = None,
        step_seconds: float | None = None,
    ) -> BlockRecord:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        timestamps_list = list(timestamps or [])
        last = self.head()
        for idx in range(steps):
            number = last.number + 1
            if idx < len(timestamps_list):
                ts = _parse_timestamp(timestamps_list[idx])
            elif step_seconds is not None:
                ts = last.timestamp + dt.timedelta(seconds=step_seconds)
            else:
                ts = last.timestamp + DEFAULT_BLOCK_INTERVAL
            last = self.set_block(number, timestamp=ts)
        logger.info(
            "state.advance_head",
            steps=steps,
            final_block=last.number,
            step_seconds=step_seconds,
            timestamps_count=len(timestamps_list),
        )
        return last

    # -- Subnet helpers ----------------------------------------------------

    def ensure_subnet(self, netuid: int) -> SubnetState:
        with self._lock:
            subnet = self._subnets.get(netuid)
            if subnet is None:
                subnet = SubnetState(netuid=netuid)
                self._subnets[netuid] = subnet
                logger.info("state.create_subnet", netuid=netuid)
            return subnet

    def netuids(self) -> list[int]:
        with self._lock:
            return sorted(self._subnets.keys())

    def update_subnet(self, netuid: int, *, tempo: int | None = None) -> SubnetState:
        subnet = self.ensure_subnet(netuid)
        if tempo is not None:
            new_tempo = int(tempo)
            if new_tempo != subnet.tempo:
                old_tempo = subnet.tempo
                subnet.tempo = new_tempo
                logger.info("state.update_subnet", netuid=netuid, tempo=new_tempo, previous_tempo=old_tempo)
        return subnet

    def set_subnet_owner(self, netuid: int, owner_hotkey: str, owner_coldkey: str) -> None:
        """Set the subnet owner (for burn mechanism)."""
        subnet = self.ensure_subnet(netuid)
        subnet.owner_hotkey = owner_hotkey
        subnet.owner_coldkey = owner_coldkey
        logger.info(
            "state.set_subnet_owner",
            netuid=netuid,
            owner_hotkey=owner_hotkey,
            owner_coldkey=owner_coldkey,
        )

    def get_subnet_owner(self, netuid: int) -> tuple[str | None, str | None]:
        """Get subnet owner hotkey and coldkey."""
        subnet = self.ensure_subnet(netuid)
        return subnet.owner_hotkey, subnet.owner_coldkey

    def subnet_epoch(self, netuid: int, block_number: int | None = None) -> dict[str, int]:
        subnet = self.ensure_subnet(netuid)
        if block_number is None:
            block_number = self.head().number
        return subnet.epoch_for(block_number)

    def set_neurons(self, netuid: int, neurons: Iterable[Mapping[str, Any]]) -> None:
        subnet = self.ensure_subnet(netuid)
        current_block = self.head().number
        subnet.set_neurons(neurons, registration_block=current_block)
        # Subnet logging already handles detailed reporting.
        logger.info(
            "state.set_neurons",
            netuid=netuid,
            registration_block=current_block,
        )

    def neurons(self, netuid: int, block_number: int | None = None) -> list[dict[str, Any]]:
        """Get neurons for a subnet, optionally at a specific block.

        Args:
            netuid: Subnet ID
            block_number: Block number to query (None = current state)

        Returns:
            List of neuron dictionaries
        """
        subnet = self.ensure_subnet(netuid)
        return [n.to_dict() for n in subnet.neurons(block_number)]

    def publish_commitment(
        self,
        netuid: int,
        *,
        block_hash: str,
        hotkey: str,
        payload: str,
    ) -> None:
        # Find block number from hash
        block_number = None
        for blk_num, blk_rec in self._blocks.items():
            if blk_rec.hash == block_hash:
                block_number = blk_num
                break
        if block_number is None:
            # Default to head if not found
            block_number = self.head().number

        subnet = self.ensure_subnet(netuid)
        subnet.publish_commitment(block_hash=block_hash, block_number=block_number, hotkey=hotkey, payload=payload)
        logger.info(
            "state.publish_commitment",
            netuid=netuid,
            block_hash=block_hash,
            block_number=block_number,
            hotkey=hotkey,
        )

    def set_commitments(self, netuid: int, *, block_hash: str, entries: Mapping[str, str]) -> None:
        # Find block number from hash
        block_number = None
        for blk_num, blk_rec in self._blocks.items():
            if blk_rec.hash == block_hash:
                block_number = blk_num
                break
        if block_number is None:
            # Default to head if not found
            block_number = self.head().number

        subnet = self.ensure_subnet(netuid)
        subnet.set_commitments(block_hash, block_number, entries)
        logger.info(
            "state.set_commitments",
            netuid=netuid,
            block_hash=block_hash,
            block_number=block_number,
            entries=len(entries),
        )

    def fetch_commitments(self, netuid: int, *, block_hash: str) -> dict[str, str]:
        # Find block number from hash
        block_number = None
        for blk_num, blk_rec in self._blocks.items():
            if blk_rec.hash == block_hash:
                block_number = blk_num
                break
        if block_number is None:
            # Default to head if not found
            block_number = self.head().number

        subnet = self.ensure_subnet(netuid)
        return subnet.fetch_commitments(block_hash, block_number)

    def set_weights(self, netuid: int, weights: Mapping[int, int], *, block_number: int | None = None) -> None:
        subnet = self.ensure_subnet(netuid)
        if block_number is None:
            block_number = self.head().number
        subnet.set_weights(weights, block_number=block_number)

    def weights(self, netuid: int) -> dict[int, int]:
        subnet = self.ensure_subnet(netuid)
        return dict(subnet.weights_by_uid)

    def get_mechanism_weights(
        self, netuid: int, mecid: int, at_block: int | None = None, hotkey: str | None = None
    ) -> dict[int, int]:
        subnet = self.ensure_subnet(netuid)
        if at_block is None:
            at_block = self.head().number
        return subnet.get_mechanism_weights(mecid, at_block=at_block, hotkey=hotkey)

    def set_mechanism_weights(
        self, netuid: int, mecid: int, weights: Mapping[int, int], *, block_number: int | None = None
    ) -> None:
        subnet = self.ensure_subnet(netuid)
        if block_number is None:
            block_number = self.head().number
        subnet.set_mechanism_weights(mecid, weights, block_number=block_number)

    def commit_weights(
        self,
        netuid: int,
        mecid: int,
        hotkey: str,
        commit_hash: str,
        reveal_round: int,
        target_block: int | None = None,
        decrypted_weights: dict[int, int] | None = None,
        encrypted_data: bytes | None = None,
    ) -> None:
        subnet = self.ensure_subnet(netuid)
        current_block = self.head().number
        subnet.commit_weights(
            mecid,
            hotkey,
            commit_hash,
            reveal_round,
            current_block,
            target_block=target_block,
            decrypted_weights=decrypted_weights,
            encrypted_data=encrypted_data,
        )
        logger.info(
            "state.commit_weights",
            netuid=netuid,
            mecid=mecid,
            hotkey=hotkey,
            commit_hash=commit_hash,
            reveal_round=reveal_round,
            target_block=target_block,
            decrypted=decrypted_weights is not None,
        )

    def set_mechanism_count(self, netuid: int, count: int) -> None:
        subnet = self.ensure_subnet(netuid)
        subnet.set_mechanism_count(count)

    def mechanism_count(self, netuid: int) -> int:
        subnet = self.ensure_subnet(netuid)
        return subnet.get_mechanism_count()

    def set_mechanism_emission_split(self, netuid: int, split: Iterable[int] | None) -> None:
        subnet = self.ensure_subnet(netuid)
        subnet.set_mechanism_emission_split(split)

    def mechanism_emission_split(self, netuid: int) -> list[int] | None:
        subnet = self.ensure_subnet(netuid)
        return subnet.get_mechanism_emission_split()

    def reveal_weights(
        self,
        netuid: int,
        mecid: int,
        hotkey: str,
        weights: Mapping[int, int],
        salt: bytes,
    ) -> bool:
        subnet = self.ensure_subnet(netuid)
        current_block = self.head().number
        success = subnet.reveal_weights(mecid, hotkey, weights, salt, current_block)
        logger.info(
            "state.reveal_weights",
            netuid=netuid,
            mecid=mecid,
            hotkey=hotkey,
            success=success,
            entries=len(weights),
        )
        return success

    # -- Rendering ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "head": self.head().to_dict(),
                "blocks": [b.to_dict() for b in sorted(self._blocks.values(), key=lambda b: b.number)],
                "subnets": {netuid: subnet.to_dict() for netuid, subnet in self._subnets.items()},
            }
