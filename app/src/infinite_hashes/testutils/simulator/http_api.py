from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, cast

import structlog
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .state import MECHANISM_SPLIT_VALUE_MAX, SimulatorState, _default_block_hash


class HTTPError(Exception):
    """Error raised when an HTTP handler wants to short-circuit the response."""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.payload = {"error": message}


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Route:
    method: str
    pattern: tuple[str, ...]
    handler: Callable[[Request], Any]
    params_model: type[BaseModel] | None = None
    query_model: type[BaseModel] | None = None
    body_model: type[BaseModel] | None = None

    def match(self, segments: Iterable[str]) -> dict[str, str] | None:
        parts = list(segments)
        if len(parts) != len(self.pattern):
            return None

        params: dict[str, str] = {}
        for token, value in zip(self.pattern, parts):
            if token.startswith("<") and token.endswith(">"):
                params[token[1:-1]] = value
            elif token == value:
                continue
            else:
                return None
        return params


@dataclass
class Request:
    state: SimulatorState
    params: Any
    query: Any
    body: Any


class HTTPRouter:
    def __init__(self, routes: Iterable[Route]):
        self._routes = list(routes)

    def dispatch(
        self,
        method: str,
        segments: list[str],
        query: Mapping[str, list[str]],
        body: Any,
        *,
        state: SimulatorState,
    ) -> Any:
        for route in self._routes:
            if route.method != method:
                continue
            params_raw = route.match(segments)
            if params_raw is None:
                continue

            try:
                params = route.params_model(**params_raw) if route.params_model else params_raw
                query_data = _flatten_query(query)
                query_obj = route.query_model(**query_data) if route.query_model else query
                body_data = body or {}
                body_obj = route.body_model(**body_data) if route.body_model else body
            except ValidationError as exc:
                raise HTTPError(HTTPStatus.BAD_REQUEST, _format_validation_error(exc)) from exc

            request = Request(state=state, params=params, query=query_obj, body=body_obj)
            return route.handler(request)

        raise HTTPError(HTTPStatus.NOT_FOUND, "not found")


class SimulatorHTTPAPI:
    """High level HTTP API surface for the simulator."""

    def __init__(self, state: SimulatorState):
        self.state = state
        self._router = HTTPRouter(
            [
                Route("GET", tuple(), self._health, params_model=EmptyParams),
                Route("GET", ("health",), self._health, params_model=EmptyParams),
                Route("GET", ("state",), self._state),
                Route("GET", ("head",), self._head),
                Route("GET", ("blocks", "<number>"), self._get_block, params_model=BlockNumberPath),
                Route("GET", ("subnets", "<netuid>"), self._get_subnet, params_model=NetuidPath),
                Route(
                    "GET",
                    ("subnets", "<netuid>", "epoch"),
                    self._get_subnet_epoch,
                    params_model=NetuidPath,
                    query_model=BlockQuery,
                ),
                Route(
                    "GET",
                    ("subnets", "<netuid>", "neurons"),
                    self._get_subnet_neurons,
                    params_model=NetuidPath,
                    query_model=NeuronQuery,
                ),
                Route("GET", ("subnets", "<netuid>", "weights"), self._get_subnet_weights, params_model=NetuidPath),
                Route(
                    "GET",
                    ("subnets", "<netuid>", "commitments"),
                    self._get_subnet_commitments,
                    params_model=NetuidPath,
                    query_model=CommitmentQuery,
                ),
                Route(
                    "GET",
                    ("subnets", "<netuid>", "mechanisms", "<mecid>", "weights"),
                    self._get_mechanism_weights,
                    params_model=MechanismPath,
                    query_model=MechanismWeightsQuery,
                ),
                Route(
                    "GET",
                    ("subnets", "<netuid>", "mechanism-split"),
                    self._get_mechanism_split,
                    params_model=NetuidPath,
                ),
                Route("POST", ("head",), self._post_head, body_model=HeadUpdateBody),
                Route("POST", ("blocks", "advance"), self._post_advance_head, body_model=AdvanceHeadBody),
                Route(
                    "POST",
                    ("blocks", "<number>"),
                    self._post_block,
                    params_model=BlockNumberPath,
                    body_model=BlockBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>"),
                    self._post_subnet_update,
                    params_model=NetuidPath,
                    body_model=SubnetUpdateBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "neurons"),
                    self._post_subnet_neurons,
                    params_model=NetuidPath,
                    body_model=SubnetNeuronsBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "weights"),
                    self._post_subnet_weights,
                    params_model=NetuidPath,
                    body_model=SubnetWeightsBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "commitments"),
                    self._post_subnet_commitments,
                    params_model=NetuidPath,
                    body_model=SubnetCommitmentsBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "mechanisms", "<mecid>", "weights"),
                    self._post_mechanism_weights,
                    params_model=MechanismPath,
                    body_model=MechanismWeightsBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "mechanisms", "<mecid>", "commits", "reveal"),
                    self._post_mechanism_reveal,
                    params_model=MechanismPath,
                    body_model=MechanismRevealBody,
                ),
                Route(
                    "POST",
                    ("subnets", "<netuid>", "mechanism-split"),
                    self._post_mechanism_split,
                    params_model=NetuidPath,
                    body_model=MechanismSplitBody,
                ),
            ]
        )

    # -- Public interface -------------------------------------------------

    def handle_get(self, segments: list[str], query: Mapping[str, list[str]]) -> Any:
        return self._router.dispatch("GET", segments, query, body=None, state=self.state)

    def handle_post(self, segments: list[str], query: Mapping[str, list[str]], body: Any) -> Any:
        return self._router.dispatch("POST", segments, query, body=body or {}, state=self.state)

    # -- GET Handlers -----------------------------------------------------

    def _health(self, request: Request) -> Any:
        return {"status": "ok"}

    def _state(self, request: Request) -> Any:
        return request.state.to_dict()

    def _head(self, request: Request) -> Any:
        return request.state.head().to_dict()

    def _get_block(self, request: Request) -> Any:
        params = cast(BlockNumberPath, request.params)
        try:
            return request.state.get_block(params.number).to_dict()
        except KeyError as exc:
            raise HTTPError(HTTPStatus.NOT_FOUND, "block not found") from exc

    def _get_subnet(self, request: Request) -> Any:
        netuid = cast(NetuidPath, request.params).netuid
        subnet = request.state.ensure_subnet(netuid)
        payload = subnet.to_dict()
        payload["epoch"] = request.state.subnet_epoch(netuid)
        return payload

    def _get_subnet_epoch(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        query = cast(BlockQuery, request.query)
        return request.state.subnet_epoch(params.netuid, block_number=query.block)

    def _get_subnet_neurons(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        query = cast(NeuronQuery, request.query)
        block_number = None
        if query.block_hash:
            block_number = request.state.block_number_for_hash(query.block_hash)
        elif query.block is not None:
            block_number = query.block
        neurons = request.state.neurons(params.netuid, block_number=block_number)
        return {"neurons": neurons}

    def _get_subnet_weights(self, request: Request) -> Any:
        netuid = cast(NetuidPath, request.params).netuid
        return {"weights": request.state.weights(netuid)}

    def _get_subnet_commitments(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        query = cast(CommitmentQuery, request.query)
        block_hash = query.block_hash
        if block_hash is None and query.block is not None:
            try:
                block_hash = request.state.get_block(query.block).hash
            except KeyError as exc:
                raise HTTPError(HTTPStatus.BAD_REQUEST, "invalid block") from exc
        if not block_hash:
            raise HTTPError(HTTPStatus.BAD_REQUEST, "missing block_hash")
        entries = request.state.fetch_commitments(params.netuid, block_hash=block_hash)
        return {"block_hash": block_hash, "entries": entries}

    def _get_mechanism_weights(self, request: Request) -> Any:
        params = cast(MechanismPath, request.params)
        query = cast(MechanismWeightsQuery, request.query)
        weights = request.state.get_mechanism_weights(
            params.netuid, params.mecid, at_block=query.block, hotkey=query.hotkey
        )
        payload: dict[str, Any] = {"mecid": params.mecid, "weights": weights}
        if query.block is not None:
            payload["at_block"] = query.block
        if query.hotkey is not None:
            payload["hotkey"] = query.hotkey
        return payload

    def _get_mechanism_split(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        netuid = params.netuid
        return {
            "netuid": netuid,
            "mechanism_count": request.state.mechanism_count(netuid),
            "split": request.state.mechanism_emission_split(netuid),
        }

    # -- POST Handlers ----------------------------------------------------

    def _post_head(self, request: Request) -> Any:
        body = cast(HeadUpdateBody, request.body)
        number = body.number if body.number is not None else request.state.head().number
        record = request.state.set_head(number, timestamp=body.timestamp, block_hash=body.hash)
        logger.info(
            "http.set_head",
            number=record.number,
            hash=record.hash,
            timestamp=record.timestamp.isoformat(),
        )
        return record.to_dict()

    def _post_advance_head(self, request: Request) -> Any:
        body = cast(AdvanceHeadBody, request.body)
        last = request.state.advance_head(body.steps, timestamps=body.timestamps, step_seconds=body.step_seconds)
        logger.info(
            "http.advance_head",
            steps=body.steps,
            head=last.number,
            step_seconds=body.step_seconds,
            timestamps_count=len(body.timestamps or []),
        )
        return {"head": last.to_dict()}

    def _post_block(self, request: Request) -> Any:
        params = cast(BlockNumberPath, request.params)
        body = cast(BlockBody, request.body)
        block_hash = body.hash
        if block_hash == "auto":
            block_hash = _default_block_hash(params.number)
        record = request.state.set_block(params.number, timestamp=body.timestamp, block_hash=block_hash)
        logger.info(
            "http.set_block",
            number=record.number,
            hash=record.hash,
            timestamp=record.timestamp.isoformat(),
            auto_hash=body.hash == "auto",
        )
        return record.to_dict()

    def _post_subnet_update(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        body = cast(SubnetUpdateBody, request.body)
        subnet = request.state.update_subnet(params.netuid, tempo=body.tempo)
        if body.owner_hotkey and body.owner_coldkey:
            request.state.set_subnet_owner(params.netuid, body.owner_hotkey, body.owner_coldkey)
        logger.info(
            "http.update_subnet",
            netuid=params.netuid,
            tempo=subnet.tempo,
            owner_hotkey=body.owner_hotkey,
            owner_coldkey=body.owner_coldkey,
        )
        payload = subnet.to_dict()
        payload["epoch"] = request.state.subnet_epoch(params.netuid)
        return payload

    def _post_subnet_neurons(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        body = cast(SubnetNeuronsBody, request.body)
        request.state.set_neurons(params.netuid, body.neurons)
        logger.info(
            "http.set_neurons",
            netuid=params.netuid,
            count=len(body.neurons),
        )
        return {"neurons": request.state.neurons(params.netuid)}

    def _post_subnet_weights(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        body = cast(SubnetWeightsBody, request.body)
        request.state.set_weights(params.netuid, body.weights)
        logger.info(
            "http.set_weights",
            netuid=params.netuid,
            entries=len(body.weights),
        )
        return {"weights": request.state.weights(params.netuid)}

    def _post_subnet_commitments(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        body = cast(SubnetCommitmentsBody, request.body)

        block_hash = body.block_hash
        if block_hash is None and body.block is not None:
            try:
                block_hash = request.state.get_block(body.block).hash
            except KeyError as exc:
                raise HTTPError(HTTPStatus.BAD_REQUEST, "invalid block") from exc
        if not block_hash:
            raise HTTPError(HTTPStatus.BAD_REQUEST, "block_hash required")

        if body.entries is not None:
            request.state.set_commitments(params.netuid, block_hash=block_hash, entries=body.entries)
            logger.info(
                "http.set_commitments",
                netuid=params.netuid,
                block_hash=block_hash,
                entries=len(body.entries),
            )
        else:
            request.state.publish_commitment(
                params.netuid,
                block_hash=block_hash,
                hotkey=str(body.hotkey),
                payload=str(body.payload),
            )
            logger.info(
                "http.publish_commitment",
                netuid=params.netuid,
                block_hash=block_hash,
                hotkey=str(body.hotkey),
                payload_type=type(body.payload).__name__,
            )

        entries = request.state.fetch_commitments(params.netuid, block_hash=block_hash)
        return {"block_hash": block_hash, "entries": entries}

    def _post_mechanism_weights(self, request: Request) -> Any:
        params = cast(MechanismPath, request.params)
        body = cast(MechanismWeightsBody, request.body)
        request.state.set_mechanism_weights(params.netuid, params.mecid, body.weights)
        logger.info(
            "http.set_mechanism_weights",
            netuid=params.netuid,
            mecid=params.mecid,
            entries=len(body.weights),
        )
        return {"mecid": params.mecid, "weights": request.state.get_mechanism_weights(params.netuid, params.mecid)}

    def _post_mechanism_reveal(self, request: Request) -> Any:
        params = cast(MechanismPath, request.params)
        body = cast(MechanismRevealBody, request.body)
        success = request.state.reveal_weights(params.netuid, params.mecid, body.hotkey, body.weights, body.salt_bytes)
        logger.info(
            "http.reveal_mechanism_weights",
            netuid=params.netuid,
            mecid=params.mecid,
            hotkey=body.hotkey,
            success=success,
            entries=len(body.weights),
        )
        response: dict[str, Any] = {"success": success, "mecid": params.mecid, "hotkey": body.hotkey}
        response["weights"] = request.state.get_mechanism_weights(params.netuid, params.mecid) if success else {}
        return response

    def _post_mechanism_split(self, request: Request) -> Any:
        params = cast(NetuidPath, request.params)
        body = cast(MechanismSplitBody, request.body)

        if body.count is not None:
            request.state.set_mechanism_count(params.netuid, body.count)
        if body.clear:
            request.state.set_mechanism_emission_split(params.netuid, None)
        elif body.split is not None:
            request.state.set_mechanism_emission_split(params.netuid, body.split)

        logger.info(
            "http.set_mechanism_split",
            netuid=params.netuid,
            count=body.count,
            updated_split=body.split is not None or body.clear,
        )
        return {
            "netuid": params.netuid,
            "mechanism_count": request.state.mechanism_count(params.netuid),
            "split": request.state.mechanism_emission_split(params.netuid),
        }


class EmptyParams(BaseModel):
    pass


class BlockNumberPath(BaseModel):
    number: int


class NetuidPath(BaseModel):
    netuid: int


class MechanismPath(NetuidPath):
    mecid: int


class BlockQuery(BaseModel):
    block: int | None = None


class NeuronQuery(BlockQuery):
    block_hash: str | None = None


class CommitmentQuery(BlockQuery):
    block_hash: str | None = None


class MechanismWeightsQuery(BlockQuery):
    hotkey: str | None = None


class HeadUpdateBody(BaseModel):
    number: int | None = None
    timestamp: Any | None = None
    hash: str | None = None


class AdvanceHeadBody(BaseModel):
    steps: int = 1
    timestamps: list[Any] | None = None
    step_seconds: float | None = None

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("steps must be non-negative")
        return value


class BlockBody(BaseModel):
    timestamp: Any | None = None
    hash: str | None = None


class SubnetUpdateBody(BaseModel):
    tempo: int | None = None
    owner_hotkey: str | None = None
    owner_coldkey: str | None = None


class SubnetNeuronsBody(BaseModel):
    neurons: list[Mapping[str, Any]] = Field(default_factory=list)


class SubnetWeightsBody(BaseModel):
    weights: dict[int, int] = Field(default_factory=dict)

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, value: Any) -> dict[int, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("weights must be a mapping")
        try:
            return {int(k): int(v) for k, v in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must map integers to integers") from exc


class SubnetCommitmentsBody(BaseModel):
    block_hash: str | None = None
    block: int | None = None
    entries: Mapping[str, Any] | None = None
    hotkey: str | None = None
    payload: Any | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> SubnetCommitmentsBody:
        if self.entries is None and (self.hotkey is None or self.payload is None):
            raise ValueError("hotkey and payload required when entries are not provided")
        return self


class MechanismWeightsBody(BaseModel):
    weights: dict[int, int] = Field(default_factory=dict)

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, value: Any) -> dict[int, int]:
        return SubnetWeightsBody._coerce_weights(value)


class MechanismRevealBody(BaseModel):
    hotkey: str
    weights: dict[int, int] = Field(default_factory=dict)
    salt: str

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, value: Any) -> dict[int, int]:
        return SubnetWeightsBody._coerce_weights(value)

    @field_validator("salt")
    @classmethod
    def _validate_salt(cls, value: str) -> str:
        try:
            bytes.fromhex(value.removeprefix("0x"))
        except ValueError as exc:
            raise ValueError("salt must be a hex string") from exc
        return value

    @property
    def salt_bytes(self) -> bytes:
        return bytes.fromhex(self.salt.removeprefix("0x"))


class MechanismSplitBody(BaseModel):
    count: int | None = None
    split: list[int] | None = None
    clear: bool = False

    @field_validator("count")
    @classmethod
    def _validate_count(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 1 or value > 255:
            raise ValueError("count must be between 1 and 255")
        return value

    @field_validator("split", mode="before")
    @classmethod
    def _coerce_split(cls, value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = list(value.values())
        if isinstance(value, str | bytes):
            raise ValueError("split must be a sequence of integers")
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("split must be a sequence of integers") from exc

    @field_validator("split")
    @classmethod
    def _validate_split_values(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("split must have at least one entry")
        for entry in value:
            if entry < 0:
                raise ValueError("split values must be non-negative")
            if MECHANISM_SPLIT_VALUE_MAX is not None and entry > MECHANISM_SPLIT_VALUE_MAX:
                raise ValueError(f"split values must be <= {MECHANISM_SPLIT_VALUE_MAX}")
        return value


def _flatten_query(query: Mapping[str, list[str]]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, values in query.items():
        if not values:
            continue
        flattened[key] = values if len(values) > 1 else values[0]
    return flattened


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) if parts else "invalid request"
