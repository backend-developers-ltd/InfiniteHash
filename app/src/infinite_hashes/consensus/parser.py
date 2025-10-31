"""Generic commitment parser with auto-discovery and binary suffix handling."""

from __future__ import annotations

from typing import TypeVar

from infinite_hashes.consensus.bidding import BiddingCommitment
from infinite_hashes.consensus.commitment import split_commitment_binary
from infinite_hashes.consensus.price import PriceCommitment

T = TypeVar("T")

# All known commitment types
_COMMITMENT_CLASSES = [
    PriceCommitment,
    BiddingCommitment,
]


def _build_token_map(classes: list[type]) -> dict[str, type]:
    """Auto-discover tokens from commitment classes."""
    token_map = {}
    for cls in classes:
        tokens = cls._allowed_tokens()
        for token in tokens:
            token_map[token] = cls
    return token_map


# Global registry: auto-discovered token -> class mapping
_COMMITMENT_REGISTRY = _build_token_map(_COMMITMENT_CLASSES)


def parse_commitment(
    raw: bytes | str,
    expected_types: list[type[T]] | None = None,
) -> T | None:
    """Generic commitment parser that handles binary suffixes and type detection.

    Args:
        raw: Raw commitment data (bytes or string)
        expected_types: List of commitment classes to accept.
                       If None, accepts all registered types.

    Returns:
        Parsed commitment instance or None if:
        - Invalid format
        - Type doesn't match expected_types filter
        - Parse error

    Example:
        # Parse only price commitments
        from infinite_hashes.consensus.price import PriceCommitment
        commit = parse_commitment(raw, [PriceCommitment])

        # Parse any registered type (auto-discovery)
        commit = parse_commitment(raw)
    """
    try:
        # Split binary suffix (if any)
        text, binary_suffix = split_commitment_binary(raw)

        # Quick parse to extract type token (v;t;d format)
        parts = text.strip().split(";", 2)
        if len(parts) < 2:
            return None

        type_token = parts[1]

        # Build token map from expected types or use registry
        if expected_types is not None:
            type_map = _build_token_map(expected_types)
        else:
            type_map = _COMMITMENT_REGISTRY

        # Filter by expected types
        if type_token not in type_map:
            return None

        commitment_class = type_map[type_token]

        # Parse using the appropriate class
        # Check if the class has special binary handling
        if hasattr(commitment_class, "from_compact_bytes"):
            # Reconstruct original raw for binary-aware parsing
            if binary_suffix:
                if isinstance(raw, str):
                    raw_bytes = raw.encode("utf-8")
                else:
                    raw_bytes = bytes(raw)
                return commitment_class.from_compact_bytes(raw_bytes)
            else:
                return commitment_class.from_compact_bytes(text)
        else:
            # Fallback to text-only parsing
            return commitment_class.from_compact(text)

    except (ValueError, UnicodeDecodeError, AttributeError):
        return None
