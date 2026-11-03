import json
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import workblock_engine.engine as engine_module
from workblock_engine import WorkblockResult, process


def _make_activities(user_id="u1", base_hour=9, count=3):
    """Create a realistic set of activities for one user."""
    base = datetime(2026, 1, 1, base_hour, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(count):
        rows.append(
            {
                "user_id": user_id,
                "timestamp": base + timedelta(minutes=i * 10),
                "name": "chrome.exe",
                "title": f"Tab {i} - Chrome",
                "url": f"https://example.com/page{i}",
                "task_name": "",
            }
        )
    return pl.DataFrame(rows)


def _make_keystrokes(user_id="u1", base_hour=9, count=3):
    """Create matching keystrokes."""
    base = datetime(2026, 1, 1, base_hour, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(count):
        rows.append(
            {
                "user_id": user_id,
                "timestamp": base + timedelta(minutes=i * 10 + 2),
                "keystrokes": f"typing{i}",
                "app": "chrome.exe",
                "exec": "chrome.exe",
                "window_title": f"Tab {i} - Chrome",
            }
        )
    return pl.DataFrame(rows)


class TestProcess:
    def test_returns_workblock_result(self):
        result = process(_make_activities(), _make_keystrokes())
        assert isinstance(result, WorkblockResult)

    def test_workblocks_have_required_columns(self):
        result = process(_make_activities(), _make_keystrokes())
        for col in [
            "workblock_id",
            "user_id",
            "date",
            "start_time",
            "end_time",
            "duration_minutes",
            "active_duration_minutes",
            "total_activities",
        ]:
            assert col in result.workblocks.columns, f"Missing column: {col}"

    def test_slices_have_required_columns(self):
        result = process(_make_activities(), _make_keystrokes())
        for col in [
            "workblock_id",
            "sequence",
            "user_id",
            "date",
            "start_time",
            "end_time",
            "duration_minutes",
            "application",
            "exec",
            "url",
            "window_title",
            "keystroke_count",
            "raw_keystrokes",
            "activity_count",
            "is_multitasking",
            "concurrent_activities",
            "concurrent_apps",
            "is_synthetic",
            "is_orphan",
            "synthetic_reason",
            "cross_app_keystrokes",
        ]:
            assert col in result.atomic_slices.columns, f"Missing column: {col}"

    def test_user_id_in_output(self):
        result = process(_make_activities(), _make_keystrokes())
        assert result.workblocks["user_id"][0] == "u1"
        assert result.atomic_slices["user_id"][0] == "u1"

    def test_date_derived_from_start_time(self):
        result = process(_make_activities(), _make_keystrokes())
        assert result.workblocks["date"][0] == "2026-01-01"

    def test_raw_keystrokes_is_json_string(self):
        result = process(_make_activities(), _make_keystrokes())
        raw = result.atomic_slices["raw_keystrokes"][0]
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert isinstance(parsed, list)

    def test_multi_user(self):
        a1 = _make_activities("u1")
        a2 = _make_activities("u2", base_hour=10)
        activities = pl.concat([a1, a2])
        k1 = _make_keystrokes("u1")
        k2 = _make_keystrokes("u2", base_hour=10)
        keystrokes = pl.concat([k1, k2])

        result = process(activities, keystrokes)
        user_ids = result.workblocks["user_id"].unique().sort().to_list()
        assert "u1" in user_ids
        assert "u2" in user_ids

    def test_empty_activities_empty_keystrokes(self):
        activities = pl.DataFrame(
            {
                "user_id": [],
                "timestamp": [],
                "name": [],
                "title": [],
                "url": [],
                "task_name": [],
            }
        ).cast({"timestamp": pl.Datetime(time_zone="UTC")})
        keystrokes = pl.DataFrame(
            {
                "user_id": [],
                "timestamp": [],
                "keystrokes": [],
                "app": [],
                "exec": [],
                "window_title": [],
            }
        ).cast({"timestamp": pl.Datetime(time_zone="UTC")})
        result = process(activities, keystrokes)
        assert result.workblocks.is_empty()
        assert result.atomic_slices.is_empty()
        assert result.workblocks.columns == [
            "workblock_id",
            "user_id",
            "date",
            "start_time",
            "end_time",
            "duration_minutes",
            "active_duration_minutes",
            "total_activities",
        ]
        assert result.workblocks.schema == {
            "workblock_id": pl.String,
            "user_id": pl.String,
            "date": pl.String,
            "start_time": pl.String,
            "end_time": pl.String,
            "duration_minutes": pl.Float64,
            "active_duration_minutes": pl.Float64,
            "total_activities": pl.Int64,
        }
        assert result.atomic_slices.columns == [
            "workblock_id",
            "sequence",
            "user_id",
            "date",
            "start_time",
            "end_time",
            "duration_minutes",
            "application",
            "exec",
            "url",
            "window_title",
            "keystroke_count",
            "raw_keystrokes",
            "activity_count",
            "is_multitasking",
            "concurrent_activities",
            "concurrent_apps",
            "is_synthetic",
            "is_orphan",
            "synthetic_reason",
            "cross_app_keystrokes",
        ]
        assert result.atomic_slices.schema == {
            "workblock_id": pl.String,
            "sequence": pl.Int64,
            "user_id": pl.String,
            "date": pl.String,
            "start_time": pl.String,
            "end_time": pl.String,
            "duration_minutes": pl.Float64,
            "application": pl.String,
            "exec": pl.String,
            "url": pl.String,
            "window_title": pl.String,
            "keystroke_count": pl.Int64,
            "raw_keystrokes": pl.String,
            "activity_count": pl.Int64,
            "is_multitasking": pl.Boolean,
            "concurrent_activities": pl.Int64,
            "concurrent_apps": pl.List(pl.String),
            "is_synthetic": pl.Boolean,
            "is_orphan": pl.Boolean,
            "synthetic_reason": pl.String,
            "cross_app_keystrokes": pl.Int64,
        }
        assert result.skipped_users == []

    def test_empty_keystrokes_still_produces_workblocks(self):
        keystrokes = pl.DataFrame(
            {
                "user_id": [],
                "timestamp": [],
                "keystrokes": [],
                "app": [],
                "exec": [],
                "window_title": [],
            }
        ).cast({"timestamp": pl.Datetime(time_zone="UTC")})
        result = process(_make_activities(), keystrokes)
        assert not result.workblocks.is_empty()

    def test_keystroke_only_user(self):
        """User with keystrokes but no activities should still produce output."""
        activities = pl.DataFrame(
            {
                "user_id": [],
                "timestamp": [],
                "name": [],
                "title": [],
                "url": [],
                "task_name": [],
            }
        ).cast({"timestamp": pl.Datetime(time_zone="UTC")})
        keystrokes = _make_keystrokes("u1")
        result = process(activities, keystrokes)
        u1_workblocks = (
            result.workblocks.filter(pl.col("user_id") == "u1").sort("start_time")
        )
        u1_slices = result.atomic_slices.filter(pl.col("user_id") == "u1").sort("start_time")

        assert result.skipped_users == []
        assert u1_workblocks.height == 1
        assert u1_workblocks["start_time"][0] == "2026-01-01T09:02:00+00:00"
        assert u1_workblocks["end_time"][0] == "2026-01-01T09:22:00+00:00"
        assert u1_workblocks["total_activities"][0] == 3
        assert u1_slices.height == 3
        assert u1_slices["is_synthetic"].to_list() == [True, True, True]
        assert u1_slices["is_orphan"].to_list() == [True, True, True]
        assert u1_slices["synthetic_reason"].to_list() == [
            "orphaned_keystrokes",
            "orphaned_keystrokes",
            "orphaned_keystrokes",
        ]
        assert u1_slices["keystroke_count"].to_list() == [1, 1, 1]
        parsed_keystrokes = [json.loads(value) for value in u1_slices["raw_keystrokes"].to_list()]
        assert [len(value) for value in parsed_keystrokes] == [1, 1, 1]
        assert [value[0]["time"] for value in parsed_keystrokes] == [
            "09:02:00",
            "09:12:00",
            "09:22:00",
        ]
        assert [value[0]["text"] for value in parsed_keystrokes] == [
            "typing0",
            "typing1",
            "typing2",
        ]
        assert [value[0]["app"] for value in parsed_keystrokes] == [
            "chrome",
            "chrome",
            "chrome",
        ]
        assert [value[0]["url"] for value in parsed_keystrokes] == ["", "", ""]

    def test_skipped_users_reported(self, monkeypatch):
        original_match_keystrokes = engine_module._match_keystrokes

        def fail_for_u2(activities_df, keystrokes_df, user_id):
            if user_id == "u2":
                raise RuntimeError("boom")
            return original_match_keystrokes(activities_df, keystrokes_df, user_id)

        monkeypatch.setattr(engine_module, "_match_keystrokes", fail_for_u2)

        activities = pl.concat([_make_activities("u1"), _make_activities("u2", 10)])
        keystrokes = pl.concat([_make_keystrokes("u1"), _make_keystrokes("u2", 10)])

        result = process(activities, keystrokes)

        assert result.skipped_users == ["u2"]
        assert result.workblocks["user_id"].unique().sort().to_list() == ["u1"]
        assert result.atomic_slices["user_id"].unique().sort().to_list() == ["u1"]

    @pytest.mark.parametrize("active_minutes", [20, 120])
    def test_orphan_after_many_regular_slices_preserves_user(self, active_minutes):
        base = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
        activities = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": [base],
                "end": [base + timedelta(minutes=active_minutes)],
                "name": ["chrome.exe"],
                "title": ["Editor"],
                "url": ["https://example.com"],
                "task_name": [""],
            }
        )
        keystrokes = pl.DataFrame(
            [
                {
                    "user_id": "u1",
                    "timestamp": base + timedelta(minutes=i, seconds=10),
                    "keystrokes": f"typing{i}",
                    "app": "chrome.exe",
                    "exec": "chrome.exe",
                    "window_title": "Editor",
                }
                for i in range(active_minutes)
            ]
            + [
                {
                    "user_id": "u1",
                    "timestamp": base + timedelta(minutes=active_minutes + 61 + i),
                    "keystrokes": f"orphan{i}",
                    "app": "code.exe",
                    "exec": "code.exe",
                    "window_title": "Editor",
                }
                for i in range(6)
            ]
        )

        result = process(activities, keystrokes)

        assert result.skipped_users == []
        assert result.workblocks.height == 2
        regular = result.atomic_slices.filter(~pl.col("is_synthetic"))
        orphans = result.atomic_slices.filter(pl.col("is_orphan"))
        assert regular.height == active_minutes
        assert regular["keystroke_count"].sum() == active_minutes
        assert orphans.height > 0
        assert orphans["is_synthetic"].all()
        assert orphans["synthetic_reason"].unique().to_list() == ["orphaned_keystrokes"]

    def test_process_string_timestamps(self):
        """Utf8 timestamps should be coerced to timezone-aware datetimes."""
        activities = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:00:00"],
                "name": ["chrome.exe"],
                "title": ["Google"],
                "url": [""],
                "task_name": [""],
            }
        )
        keystrokes = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:05:00"],
                "keystrokes": ["hello"],
                "app": ["chrome.exe"],
                "exec": ["chrome.exe"],
                "window_title": ["Google"],
            }
        )
        result = process(activities, keystrokes)
        assert not result.workblocks.is_empty()

    def test_process_rfc3339_z_timestamps(self):
        activities = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:00:00Z"],
                "name": ["chrome.exe"],
                "title": ["Google"],
                "url": [""],
                "task_name": [""],
            }
        )
        keystrokes = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:05:00Z"],
                "keystrokes": ["hello"],
                "app": ["chrome.exe"],
                "exec": ["chrome.exe"],
                "window_title": ["Google"],
            }
        )

        result = process(activities, keystrokes)

        assert result.skipped_users == []
        assert result.workblocks["start_time"][0] == "2026-01-01T09:00:00+00:00"

    def test_process_offset_timestamps_normalized_to_utc(self):
        activities = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:00:00+02:00"],
                "name": ["chrome.exe"],
                "title": ["Google"],
                "url": [""],
                "task_name": [""],
            }
        )
        keystrokes = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": ["2026-01-01T09:05:00+02:00"],
                "keystrokes": ["hello"],
                "app": ["chrome.exe"],
                "exec": ["chrome.exe"],
                "window_title": ["Google"],
            }
        )

        result = process(activities, keystrokes)

        assert result.skipped_users == []
        assert result.workblocks["start_time"][0] == "2026-01-01T07:00:00+00:00"

    def test_cross_midnight_multi_day_output(self):
        """Activities crossing midnight should produce rows for multiple dates."""
        activities = pl.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "timestamp": [
                    datetime(2026, 1, 1, 23, 50, tzinfo=timezone.utc),
                    datetime(2026, 1, 2, 0, 10, tzinfo=timezone.utc),
                ],
                "name": ["chrome.exe", "chrome.exe"],
                "title": ["Tab1", "Tab2"],
                "url": ["", ""],
                "task_name": ["", ""],
            }
        )
        keystrokes = pl.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": [datetime(2026, 1, 1, 23, 55, tzinfo=timezone.utc)],
                "keystrokes": ["typing"],
                "app": ["chrome.exe"],
                "exec": ["chrome.exe"],
                "window_title": ["Tab1"],
            }
        )
        result = process(activities, keystrokes)
        assert result.workblocks["date"].unique().sort().to_list() == [
            "2026-01-01",
            "2026-01-02",
        ]
        assert result.atomic_slices["date"].unique().sort().to_list() == [
            "2026-01-01",
            "2026-01-02",
        ]
        assert result.workblocks.select(
            (pl.col("date") == pl.col("start_time").str.slice(0, 10)).all().alias("ok")
        )["ok"][0]
        assert result.atomic_slices.select(
            (pl.col("date") == pl.col("start_time").str.slice(0, 10)).all().alias("ok")
        )["ok"][0]

    def test_frozen_result(self):
        result = process(_make_activities(), _make_keystrokes())
        with pytest.raises(AttributeError):
            result.workblocks = None  # type: ignore[misc]
