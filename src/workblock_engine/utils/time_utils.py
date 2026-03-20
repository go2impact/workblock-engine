"""Time-related utility functions."""
from datetime import datetime, timezone


def make_tz_aware(dt: datetime) -> datetime:
    """Convert naive datetime to UTC or return as-is if already aware."""
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt