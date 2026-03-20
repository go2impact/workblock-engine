"""Utility helpers for deterministic, compact identifier generation.

Example usage (handles both naive and aware datetimes)::

    >>> from datetime import datetime
    >>> generate_workblock_id("alice", datetime(2024, 1, 1, 9, 0, 0), sequence=0)
    'wb:31my5iJtNx:U06GFvM:00'

The helper automatically normalizes naive datetimes to UTC before encoding so
ID generation remains stable across environments.
"""
from __future__ import annotations

from hashlib import md5
from typing import Union
from datetime import datetime, timezone

_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BASE = len(_BASE62_ALPHABET)


def _ensure_non_negative(value: int) -> int:
    if value < 0:
        raise ValueError("Base62 encoding requires non-negative integers")
    return value


def to_base62(value: int) -> str:
    """Convert a non-negative integer to a base62 string."""
    value = _ensure_non_negative(value)
    if value == 0:
        return "0"

    digits = []
    while value > 0:
        value, remainder = divmod(value, _BASE)
        digits.append(_BASE62_ALPHABET[remainder])

    return "".join(reversed(digits))


def from_base62(value: str) -> int:
    """Convert a base62 string back to an integer."""
    if not value:
        raise ValueError("Base62 string must be non-empty")

    num = 0
    for char in value:
        try:
            digit = _BASE62_ALPHABET.index(char)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid base62 character: {char}") from exc
        num = num * _BASE + digit
    return num


def normalize_timestamp(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in UTC without altering the instant."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def timestamp_token(dt: datetime) -> str:
    """Return base62 token for a datetime after normalizing to UTC milliseconds.

    Accepts both naive (assumed UTC) and timezone-aware datetimes, always
    producing the same token for the same instant irrespective of local
    environment.
    """
    normalized = normalize_timestamp(dt)
    millis = int(normalized.timestamp() * 1000)
    if millis < 0:
        raise ValueError("Timestamps before Unix epoch are not supported")
    return to_base62(millis)


def _hash_user_id(user_id: Union[str, int]) -> int:
    """Create a stable 128-bit hash from user identifiers."""
    user_bytes = str(user_id).encode("utf-8")
    return int.from_bytes(md5(user_bytes).digest(), byteorder="big", signed=False)


def user_token(user_id: Union[str, int], length: int = 10) -> str:
    """Derive a fixed-length base62 token for a user identifier.

    Defaults to 10 base62 characters (~60 bits) to minimize cross-user collisions.
    """
    if length <= 0:
        raise ValueError("Token length must be positive")

    hashed_value = _hash_user_id(user_id)
    base62_value = to_base62(hashed_value)

    if len(base62_value) >= length:
        return base62_value[:length]

    return base62_value.rjust(length, "0")


def generate_workblock_id(user_id: Union[str, int], start_time: datetime, sequence: int) -> str:
    """Generate deterministic workblock ID (``wb:<user_token>:<time_token>:<block_token>``).

    Args:
        user_id: Raw identifier consistently used for the user across runs.
        start_time: Workblock start timestamp (naive or aware datetime).
        sequence: Zero-based counter scoped to the user/workflow ordering.

    Returns:
        Stable identifier suitable for referencing workblocks across users.
    """
    if sequence < 0:
        raise ValueError("sequence must be non-negative")

    user_part = user_token(user_id)
    time_part = timestamp_token(start_time)
    block_part = slice_token(sequence)
    return f"wb:{user_part}:{time_part}:{block_part}"


def slice_token(slice_index: int, min_length: int = 2) -> str:
    """Return base62 token for a slice counter with zero-padding.

    Ensures lexicographic ordering lines up with numeric ordering by left
    padding with ``0`` when the representation is shorter than ``min_length``.
    """
    slice_index = _ensure_non_negative(slice_index)
    token = to_base62(slice_index)
    if len(token) >= min_length:
        return token
    return token.rjust(min_length, "0")
