from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel

# Binary suffix separator for commitments (0xFF - not valid UTF-8)
BINARY_SEPARATOR = b"\xff"


def split_commitment_binary(commitment: bytes | str) -> tuple[str, bytes]:
    """Split commitment into text part and binary suffix.

    Returns:
        (text_part, binary_suffix) where text_part is UTF-8 decoded string
        and binary_suffix is raw bytes (empty if no binary data).
    """
    if isinstance(commitment, str):
        commitment_bytes = commitment.encode("utf-8")
    else:
        commitment_bytes = bytes(commitment)

    separator_idx = commitment_bytes.find(BINARY_SEPARATOR)
    if separator_idx >= 0:
        text_bytes = commitment_bytes[:separator_idx]
        binary_data = commitment_bytes[separator_idx + 1 :]
        return text_bytes.decode("utf-8"), binary_data
    else:
        return commitment_bytes.decode("utf-8"), b""


def join_commitment_binary(text: str, binary_suffix: bytes) -> bytes:
    """Join text part and binary suffix into commitment bytes."""
    text_bytes = text.encode("utf-8")
    if binary_suffix:
        return text_bytes + BINARY_SEPARATOR + binary_suffix
    return text_bytes


class CompactCommitment(BaseModel, ABC):
    """Abstract compact-serialization interface for chain commitments.

    Contract:
    - Instances must carry a version `v` (int >= 1) and type token `t` (short string).
    - The payload `d` is represented as a compact string without JSON overhead.
    - Subclasses implement how to turn their internal structure into `d` and back.

    The wire format is a semicolon-delimited sequence:
        v;t;d
    where `d` is subclass-specific and should be as short as possible.
    """

    # Subclasses should define the accepted type tokens (e.g., ("p", "price")).
    # Common version field for all commitments.
    v: int = 1

    @abstractmethod
    def _d_compact(self) -> str:  # pragma: no cover - implemented by subclass
        """Return the compact `d` string for this commitment."""
        ...

    @classmethod
    @abstractmethod
    def _from_d_compact(cls, v: int, d: str):  # pragma: no cover - subclass returns instance
        """Construct an instance from version and compact `d` string."""
        ...

    def to_compact(self) -> str:
        """Serialize the commitment into the compact wire format `v;t;d`.

        Subclasses should ensure `t` is the compact, short token (e.g. 'p').
        """
        v = self.v
        t = self._compact_t()
        s = f"{v};{t};{self._d_compact()}"
        if len(s.encode("utf-8")) > 128:
            raise ValueError("compact commitment exceeds 128 bytes")
        return s

    @classmethod
    def from_compact(cls, raw: str):
        """Parse a compact-serialized commitment into an instance of `cls`.

        The caller should ensure `cls` corresponds to the `t` token contained in `raw`.
        """
        s = (raw or "").strip()
        parts = s.split(";", 2)
        if len(parts) < 3:
            raise ValueError("compact format requires v;t;d")
        try:
            v = int(parts[0])
        except ValueError as e:
            raise ValueError("invalid version in compact format") from e
        t = parts[1]
        d = parts[2]
        allowed = cls._allowed_tokens()
        if allowed and t not in allowed:
            raise ValueError("type token mismatch for commitment class")
        return cls._from_d_compact(v=v, d=d)

    def _compact_t(self) -> str:
        """Return the compact token to use when serializing.

        Default uses the instance `t` if present, otherwise the first token in TYPE_TOKENS.
        Subclasses may override if needed.
        """
        t = getattr(self, "t", None)
        if isinstance(t, str) and t:
            return t
        tokens = self._allowed_tokens()
        if tokens:
            # prefer shortest token for compactness
            return min(tokens, key=len)
        return ""

    @classmethod
    def _allowed_tokens(cls) -> tuple[str, ...]:
        """Return allowed type tokens inferred from subclass `t` Literal annotation.

        If subclass doesn't annotate `t` with `Literal[...]`, returns empty tuple.
        """
        # Resolve forward references and future annotations
        try:
            anns = get_type_hints(cls, include_extras=True)
        except Exception:  # fallback to raw annotations
            anns = getattr(cls, "__annotations__", {})
        ann = anns.get("t")
        if ann is None:
            return tuple()
        origin = get_origin(ann)
        if origin is Literal or getattr(origin, "__name__", "") == "Literal":
            args = get_args(ann)
            return tuple(str(a) for a in args)
        return tuple()
