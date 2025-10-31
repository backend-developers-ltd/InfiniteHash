from __future__ import annotations

import json
import pathlib

import turbobt

from .transport import MockedTransport


class SubtensorSimulator:
    """
    Lightweight simulator over MockedTransport that uses turbobt's registry
    to encode SCALE from human-readable Python data.
    """

    def __init__(self):
        self.transport = MockedTransport()
        self.bt = turbobt.Bittensor(
            "ws://127.0.0.1:9944",
            transport=self.transport,
            verify=None,
        )

    async def __aenter__(self) -> SubtensorSimulator:
        await self.bt.__aenter__()
        await self._bootstrap_runtime()
        return self

    async def __aexit__(self, *args):
        await self.bt.__aexit__(*args)

    async def _bootstrap_runtime(self):
        """
        Initialize turbobt runtime by mocking metadata responses from the encoded
        hex file tracked under this repository.
        """
        encoded_path = self._find_metadata_encoded()

        if not encoded_path or not encoded_path.exists():
            raise FileNotFoundError(
                "Missing metadata file. Expected infinite_hashes/consensus/tests/data/metadata_at_version.json."
            )

        meta_hex = json.loads(encoded_path.read_text())

        self.transport.responses.setdefault("state_call", {})["Metadata_metadata_at_version"] = {"result": meta_hex}
        # Header number should be hex string per Substrate RPC conventions
        self.transport.responses["chain_getHeader"] = {"result": {"number": "0x1"}}
        # Provide minimal runtime/chain info used during signing
        self.transport.responses["state_getRuntimeVersion"] = {"result": {"specVersion": 1, "transactionVersion": 1}}
        self.transport.responses["chain_getRuntimeVersion"] = {"result": {"specVersion": 1, "transactionVersion": 1}}
        self.transport.responses["chain_getGenesisHash"] = {"result": "0x" + ("11" * 32)}
        # Default next nonce for any account queried
        self.transport.responses["system_accountNextIndex"] = {"result": 0}

        # Initialize turbobt registry/metadata (builds _apis automatically)
        await self.bt.subtensor._init_runtime()

    def _find_metadata_encoded(self) -> pathlib.Path | None:
        # Walk up parents to locate the test data file under consensus/tests/data
        for base in pathlib.Path(__file__).resolve().parents:
            candidate = base / "infinite_hashes" / "consensus" / "tests" / "data" / "metadata_at_version.json"
            if candidate.exists():
                return candidate
        return None

    def set_block_context(self, *, number: int, block_hash: str | None = None) -> str:
        """Set the current block context and return the block hash.

        If block_hash is not provided, compute a deterministic 32-byte
        big-endian hash from the number.
        """
        if block_hash is None:
            block_hash = f"0x{int(number).to_bytes(32, byteorder='big').hex()}"
        # RPC expects hex-encoded block number
        self.transport.responses["chain_getHeader"] = {"result": {"number": f"0x{int(number):x}"}}
        self.transport.responses["chain_getBlockHash"] = {"result": block_hash}
        return block_hash

    def reset(self):
        """Clear dynamic responses and restore minimal baseline state.

        Keeps the encoded metadata used for registry bootstrap.
        """
        # Preserve metadata response if present
        md = None
        if "state_call" in self.transport.responses:
            md = self.transport.responses["state_call"].get("Metadata_metadata_at_version")

        self.transport.responses.clear()

        if md is not None:
            self.transport.responses["state_call"] = {
                "Metadata_metadata_at_version": md,
            }

        # Default block context
        self.set_block_context(number=1)

    def set_state_call(self, api_name: str, scale_type: str, data: dict):
        """
        Encode a dict into SCALE using the client's registry and set as state_call response.
        """
        scale_obj = self.bt.subtensor._registry.create_scale_object(scale_type)
        scale_obj.encode(data)

        self.transport.responses.setdefault("state_call", {})[api_name] = {"result": f"0x{scale_obj.data.data.hex()}"}

    def set_runtime_api(self, api: str, method: str, data):
        """
        Encode output of a Runtime API method using the metadata-derived type id.
        """
        api_info = self.bt.subtensor._apis[api]
        out_ty = api_info["methods"][method]["output"]
        scale_obj = self.bt.subtensor._registry.create_scale_object(f"scale_info::{out_ty}")
        scale_obj.encode(data)
        self.transport.responses.setdefault("state_call", {})[f"{api}_{method}"] = {
            "result": f"0x{scale_obj.data.data.hex()}"
        }

    def set_neurons_lite(self, netuid: int, neurons: list[dict]):
        """
        Encode and inject output for NeuronInfoRuntimeApi_get_neurons_lite.

        neurons: list of dicts with keys:
          - hotkey: ss58 string
          - coldkey: ss58 string (optional, defaults to hotkey)
          - uid: int (optional, auto-assigned)
          - stake: int | float (human units, will be scaled by 1e9)
        Other fields are populated with minimal zero/defaults.
        """
        # Discover output type id from runtime APIs
        api = self.bt.subtensor._apis["NeuronInfoRuntimeApi"]
        out_ty = api["methods"]["get_neurons_lite"]["output"]
        scale_obj = self.bt.subtensor._registry.create_scale_object(f"scale_info::{out_ty}")

        encoded_neurons = []
        for i, n in enumerate(neurons):
            hk_ss58 = n["hotkey"]
            ck_ss58 = n.get("coldkey", hk_ss58)
            stake_val = n.get("stake", 1.0)
            balance = int(stake_val * 1_000_000_000) if isinstance(stake_val, float) else int(stake_val)

            encoded_neurons.append(
                {
                    "hotkey": hk_ss58,
                    "coldkey": ck_ss58,
                    "uid": n.get("uid", i),
                    "netuid": netuid,
                    "active": True,
                    "axon_info": {
                        "block": 0,
                        "version": 0,
                        "ip": 0,
                        "port": 0,
                        "ip_type": 0,
                        "protocol": 0,
                        "placeholder1": 0,
                        "placeholder2": 0,
                    },
                    "prometheus_info": {
                        "block": 0,
                        "version": 0,
                        "ip": 0,
                        "port": 0,
                        "ip_type": 0,
                    },
                    "stake": [
                        [hk_ss58, balance],
                    ],
                    "rank": 0,
                    "emission": 0,
                    "incentive": 0,
                    "consensus": 0,
                    "trust": 0,
                    "validator_trust": 0,
                    "dividends": 0,
                    "last_update": 0,
                    "validator_permit": True,
                    "weights": [],
                    "bonds": [],
                    "pruning_score": 0,
                }
            )

        scale_obj.encode(encoded_neurons)
        self.transport.responses.setdefault("state_call", {})["NeuronInfoRuntimeApi_get_neurons_lite"] = {
            "result": f"0x{scale_obj.data.data.hex()}"
        }

    def set_commitments(self, netuid: int, entries: dict[str, bytes | str | dict]):
        """
        Inject Commitments.CommitmentOf entries for the given netuid.

        entries: mapping of hotkey -> commitment payload (bytes|hex str|dict)
        If dict is provided, it is JSON-encoded first.
        """

        # Normalize payload to bytes
        def _to_bytes(v) -> bytes:
            if isinstance(v, bytes):
                return v
            if isinstance(v, str):
                if v.startswith("0x"):
                    return bytes.fromhex(v[2:])
                return v.encode("utf-8")
            return json.dumps(v).encode("utf-8")

        pallet = self.bt.subtensor._metadata.get_metadata_pallet("Commitments")
        storage_fn = pallet.get_storage_function("CommitmentOf")
        value_type = storage_fn.get_value_type_string()

        keys = []
        changes = []

        for hotkey, payload in entries.items():
            raw = _to_bytes(payload)

            # Build Registration value directly
            registration = {
                "block": 0,
                "deposit": 0,
                "info": {
                    "fields": [[{f"Raw{len(raw)}": f"0x{raw.hex()}"}]],
                },
            }

            encoded = self.bt.subtensor._registry.create_scale_object(value_type)
            encoded.encode(registration)
            encoded_hex = f"0x{encoded.data.data.hex()}"

            key = self.bt.subtensor.state._storage_key(
                pallet,
                storage_fn,
                (netuid, hotkey),
            )

            keys.append(key)
            changes.append([key, encoded_hex])

        # Provide getKeys and queryStorageAt responses
        self.transport.responses["state_getKeys"] = {"result": keys}
        self.transport.responses["state_getKeysPaged"] = {"result": keys}
        self.transport.responses["state_queryStorageAt"] = {
            "result": [
                {
                    "block": self.bt.subtensor.chain.getBlockHash if False else "0xabc",
                    "changes": changes,
                }
            ]
        }
