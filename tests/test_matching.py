from datetime import datetime, timezone
import json
import polars as pl
from workblock_engine.matching.synthetic import (
    phase3_create_gap_activities,
    phase4_create_orphan_activities,
)


class TestPhase3GapActivities:
    def test_no_gap_returns_empty(self):
        activities = [
            {"start": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
             "end": datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)},
            {"start": datetime(2026, 1, 1, 9, 34, tzinfo=timezone.utc),
             "end": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)},
        ]
        keystrokes = [
            {"id": 1, "timestamp": datetime(2026, 1, 1, 9, 32, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "hello", "app": "chrome",
             "text": "hello", "window_title": "Search"},
        ]
        result = phase3_create_gap_activities(activities, keystrokes)
        assert result == []

    def test_gap_with_keystrokes_creates_synthetic(self):
        activities = [
            {"start": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
             "end": datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)},
            {"start": datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
             "end": datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc)},
        ]
        keystrokes = [
            {"id": 1, "timestamp": datetime(2026, 1, 1, 9, 40, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "hello", "app": "chrome",
             "text": "hello", "url": "https://example.com",
             "window_title": "Example"},
        ]
        result = phase3_create_gap_activities(activities, keystrokes)
        assert len(result) == 1
        activity = result[0]
        assert activity["is_synthetic"] is True
        assert activity["synthetic_reason"] == "activity_gap"
        assert activity["app_normalized"] == "chrome"
        assert activity["start"] == keystrokes[0]["timestamp"]
        assert activity["end"] == keystrokes[0]["timestamp"]
        assert activity["title"] == "https://example.com"
        assert activity["url"] == "https://example.com"
        assert activity["_keystrokes"] == keystrokes

    def test_large_dataset_uses_polars_path(self):
        activities = []
        base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
        for index in range(50):
            if index < 25:
                start = base.replace(hour=8, minute=index)
            else:
                start = base.replace(hour=8, minute=index + 9)
            activities.append({
                "start": start,
                "end": start.replace(minute=start.minute + 1),
            })

        keystrokes = [
            {"id": 99, "timestamp": datetime(2026, 1, 1, 8, 30, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "hello", "app": "chrome",
             "text": "hello", "url": "https://example.com",
             "window_title": "Example"},
        ]

        result = phase3_create_gap_activities(activities, keystrokes)
        assert len(result) == 1
        activity = result[0]
        assert activity["is_synthetic"] is True
        assert activity["synthetic_reason"] == "activity_gap"
        assert activity["app_normalized"] == "chrome"
        assert activity["start"] == keystrokes[0]["timestamp"]
        assert activity["end"] == keystrokes[0]["timestamp"]
        assert activity["title"] == "https://example.com"
        assert activity["url"] == "https://example.com"
        assert activity["_keystrokes"] == keystrokes


class TestPhase4OrphanActivities:
    def test_empty_returns_empty(self):
        assert phase4_create_orphan_activities([]) == []

    def test_single_keystroke_creates_orphan(self):
        keystrokes = [
            {"id": 1, "timestamp": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "test", "app": "chrome",
             "text": "test"},
        ]
        result = phase4_create_orphan_activities(keystrokes)
        assert len(result) == 1
        assert result[0]["is_synthetic"] is True
        assert result[0]["is_orphan"] is True
        assert result[0]["synthetic_reason"] == "orphaned_keystrokes"

    def test_multiple_apps_and_gap_grouping(self):
        keystrokes = [
            {"id": 1, "timestamp": datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "a", "app": "chrome",
             "text": "a", "url": "https://chrome.example"},
            {"id": 2, "timestamp": datetime(2026, 1, 1, 9, 4, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "b", "app": "chrome",
             "text": "b", "window_title": "Chrome search"},
            {"id": 3, "timestamp": datetime(2026, 1, 1, 9, 10, tzinfo=timezone.utc),
             "exec": "chrome.exe", "keystrokes": "c", "app": "chrome",
             "text": "c", "window_title": "Chrome later"},
            {"id": 4, "timestamp": datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc),
             "exec": "slack.exe", "keystrokes": "d", "app": "slack",
             "text": "d", "window_title": "Slack channel"},
            {"id": 5, "timestamp": datetime(2026, 1, 1, 9, 5, tzinfo=timezone.utc),
             "exec": "slack.exe", "keystrokes": "e", "app": "slack",
             "text": "e", "window_title": "Slack channel follow-up"},
            {"id": 6, "timestamp": datetime(2026, 1, 1, 9, 11, tzinfo=timezone.utc),
             "exec": "slack.exe", "keystrokes": "f", "app": "slack",
             "text": "f", "window_title": "Slack later"},
        ]

        result = phase4_create_orphan_activities(keystrokes)
        assert len(result) == 4

        grouped = {(activity["app_normalized"], activity["start"]): activity for activity in result}

        chrome_first = grouped[("chrome", datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc))]
        assert chrome_first["end"] == datetime(2026, 1, 1, 9, 4, tzinfo=timezone.utc)
        assert [ks["id"] for ks in chrome_first["_keystrokes"]] == [1, 2]
        assert chrome_first["title"] == "https://chrome.example"
        assert chrome_first["url"] == "https://chrome.example"

        chrome_second = grouped[("chrome", datetime(2026, 1, 1, 9, 10, tzinfo=timezone.utc))]
        assert chrome_second["end"] == datetime(2026, 1, 1, 9, 10, tzinfo=timezone.utc)
        assert [ks["id"] for ks in chrome_second["_keystrokes"]] == [3]

        slack_first = grouped[("slack", datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc))]
        assert slack_first["end"] == datetime(2026, 1, 1, 9, 5, tzinfo=timezone.utc)
        assert [ks["id"] for ks in slack_first["_keystrokes"]] == [4, 5]

        slack_second = grouped[("slack", datetime(2026, 1, 1, 9, 11, tzinfo=timezone.utc))]
        assert slack_second["end"] == datetime(2026, 1, 1, 9, 11, tzinfo=timezone.utc)
        assert [ks["id"] for ks in slack_second["_keystrokes"]] == [6]


from workblock_engine.matching.keystroke_matcher import join_keystrokes_to_activities


class TestKeystrokeMatcher:
    def test_empty_keystrokes_adds_columns(self):
        activities = pl.DataFrame({
            "user_id": ["u1"],
            "start": [datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)],
            "end": [datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)],
            "name": ["chrome.exe"],
            "title": ["Google"],
            "url": [""],
            "app_normalized": ["chrome"],
            "duration_minutes": [30.0],
        })
        keystrokes = pl.DataFrame({
            "user_id": [], "timestamp": [], "keystrokes": [],
            "app": [], "exec": [], "window_title": [],
        }).cast({
            "timestamp": pl.Datetime(time_zone="UTC"),
        })
        result = join_keystrokes_to_activities(activities, keystrokes, "u1")
        assert "keystrokes" in result.columns
        assert "keystroke_count" in result.columns
        assert "cross_app_keystrokes" in result.columns
        assert "is_synthetic" in result.columns
        assert "synthetic_reason" in result.columns
        assert result["keystroke_count"][0] == 0
        assert result["cross_app_keystrokes"][0] == 0
        assert result["is_synthetic"][0] is False
        assert result["synthetic_reason"][0] == ""
        assert result["keystrokes"][0] == "[]"
        assert result.height == 1

    def test_phase1_temporal_app_match(self):
        activities = pl.DataFrame({
            "user_id": ["u1"],
            "start": [datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)],
            "end": [datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)],
            "name": ["chrome.exe"],
            "title": ["Google"],
            "url": [""],
            "duration_minutes": [30.0],
        })
        keystrokes = pl.DataFrame({
            "user_id": ["u1"],
            "timestamp": [datetime(2026, 1, 1, 9, 10, tzinfo=timezone.utc)],
            "keystrokes": ["hello"],
            "app": ["chrome"],
            "exec": ["chrome.exe"],
            "window_title": ["Google"],
            "text": ["hello"],
        })
        result = join_keystrokes_to_activities(activities, keystrokes, "u1")
        assert result.height == 1
        assert result["keystroke_count"][0] == 1
        assert result["cross_app_keystrokes"][0] == 0
        assert result["is_synthetic"][0] is False
        assert result["synthetic_reason"][0] == ""

        serialized_keystrokes = json.loads(result["keystrokes"][0])
        assert serialized_keystrokes == [{
            "id": serialized_keystrokes[0]["id"],
            "time": "09:10:00",
            "text": "hello",
            "app": "chrome",
            "url": "",
        }]
