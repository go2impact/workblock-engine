import pytest
from datetime import datetime, timedelta, timezone
from workblock_engine.utils.id_utils import (
    to_base62, from_base62, generate_workblock_id, user_token, timestamp_token
)


class TestBase62:
    def test_zero(self):
        assert to_base62(0) == "0"

    def test_roundtrip(self):
        for n in [0, 1, 61, 62, 1000, 999999]:
            assert from_base62(to_base62(n)) == n

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="Base62 encoding requires non-negative integers"):
            to_base62(-1)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Base62 string must be non-empty"):
            from_base62("")

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid base62 character: !"):
            from_base62("!")


class TestGenerateWorkblockId:
    def test_deterministic(self):
        dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        id1 = generate_workblock_id("alice", dt, 0)
        id2 = generate_workblock_id("alice", dt, 0)
        assert id1 == id2

    def test_format(self):
        dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        result = generate_workblock_id("alice", dt, 0)
        assert result.startswith("wb:")
        parts = result.split(":")
        assert len(parts) == 4

    def test_negative_sequence_raises(self):
        dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="sequence must be non-negative"):
            generate_workblock_id("alice", dt, -1)

    def test_timestamp_token_normalizes_equivalent_instants(self):
        utc_dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        offset_dt = datetime(2026, 1, 1, 4, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

        assert timestamp_token(utc_dt) == timestamp_token(offset_dt)
        assert generate_workblock_id("alice", utc_dt, 0) == generate_workblock_id("alice", offset_dt, 0)

    def test_naive_datetime_treated_as_utc(self):
        naive_dt = datetime(2026, 1, 1, 9, 0, 0)
        aware_utc_dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        assert timestamp_token(naive_dt) == timestamp_token(aware_utc_dt)


class TestUserToken:
    def test_fixed_length(self):
        token = user_token("alice", length=10)
        assert len(token) == 10

    def test_deterministic(self):
        assert user_token("alice") == user_token("alice")

    def test_zero_length_raises(self):
        with pytest.raises(ValueError, match="Token length must be positive"):
            user_token("alice", length=0)


class TestTimestampToken:
    def test_pre_epoch_raises(self):
        dt = datetime(1969, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Timestamps before Unix epoch are not supported"):
            timestamp_token(dt)
