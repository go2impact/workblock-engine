import json
from datetime import datetime, timedelta, timezone

import polars as pl

from workblock_engine.building.workblock_builder import WorkblockBuilder


class TestWorkblockBuilder:
    def _make_activities(self, count=3):
        """Create `count` 15-minute activities starting every 20 minutes."""
        base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(count):
            start = base + timedelta(minutes=i * 20)
            end = start + timedelta(minutes=15)
            rows.append(
                {
                    "user_id": "u1",
                    "start": start,
                    "end": end,
                    "name": "chrome.exe",
                    "title": "Google",
                    "url": "",
                    "exec": "chrome.exe",
                    "app_filename": "chrome.exe",
                    "app_normalized": "chrome",
                    "duration_minutes": 15.0,
                    "keystrokes": json.dumps([]),
                    "keystroke_count": 0,
                    "cross_app_keystrokes": 0,
                    "is_synthetic": False,
                    "synthetic_reason": "",
                }
            )
        return pl.DataFrame(rows)

    def test_build_returns_workblocks(self):
        df = self._make_activities()
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        assert "workblocks" in result
        assert len(result["workblocks"]) > 0

    def test_build_returns_atomic_slices(self):
        df = self._make_activities()
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        assert "atomic_slices" in result
        assert len(result["atomic_slices"]) > 0

    def test_workblock_has_required_fields(self):
        df = self._make_activities()
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        wb = result["workblocks"][0]
        assert "workblock_id" in wb
        assert "start_time" in wb
        assert "end_time" in wb
        assert "duration_minutes" in wb

    def test_60_min_gap_creates_new_workblock(self):
        """Activities >60 min apart should be in separate workblocks."""
        base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        df = pl.DataFrame(
            [
                {
                    "user_id": "u1",
                    "start": base,
                    "end": base + timedelta(minutes=15),
                    "name": "chrome.exe",
                    "title": "",
                    "url": "",
                    "exec": "chrome.exe",
                    "app_filename": "chrome.exe",
                    "app_normalized": "chrome",
                    "duration_minutes": 15.0,
                    "keystrokes": "[]",
                    "keystroke_count": 0,
                    "cross_app_keystrokes": 0,
                    "is_synthetic": False,
                    "synthetic_reason": "",
                },
                {
                    "user_id": "u1",
                    "start": base + timedelta(minutes=90),
                    "end": base + timedelta(minutes=105),
                    "name": "chrome.exe",
                    "title": "",
                    "url": "",
                    "exec": "chrome.exe",
                    "app_filename": "chrome.exe",
                    "app_normalized": "chrome",
                    "duration_minutes": 15.0,
                    "keystrokes": "[]",
                    "keystroke_count": 0,
                    "cross_app_keystrokes": 0,
                    "is_synthetic": False,
                    "synthetic_reason": "",
                },
            ]
        )
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        assert len(result["workblocks"]) == 2

    def test_short_workblock_filtered(self):
        """Workblocks < 5 minutes should be filtered out."""
        base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        df = pl.DataFrame(
            [
                {
                    "user_id": "u1",
                    "start": base,
                    "end": base + timedelta(minutes=3),
                    "name": "chrome.exe",
                    "title": "",
                    "url": "",
                    "exec": "chrome.exe",
                    "app_filename": "chrome.exe",
                    "app_normalized": "chrome",
                    "duration_minutes": 3.0,
                    "keystrokes": "[]",
                    "keystroke_count": 0,
                    "cross_app_keystrokes": 0,
                    "is_synthetic": False,
                    "synthetic_reason": "",
                },
            ]
        )
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        assert len(result["workblocks"]) == 0

    def test_no_enrichment_methods(self):
        """Verify enrichment methods were removed."""
        builder = WorkblockBuilder()
        assert not hasattr(builder, "enrich_with_jobs")
        assert not hasattr(builder, "enrich_with_manifests")
        assert not hasattr(builder, "format_workblocks_text")
        assert not hasattr(builder, "calculate_usage_statistics")
        assert not hasattr(builder, "build_and_enrich_workblocks")

    def test_no_usage_statistics_in_return(self):
        df = self._make_activities()
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        assert "usage_statistics" not in result

    def test_provenance_fields_in_slices(self):
        """Verify is_synthetic, is_orphan, synthetic_reason, cross_app_keystrokes
        survive from source activity into atomic slices."""
        base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        df = pl.DataFrame(
            [
                {
                    "user_id": "u1",
                    "start": base,
                    "end": base + timedelta(minutes=15),
                    "name": "chrome.exe",
                    "title": "",
                    "url": "",
                    "exec": "chrome.exe",
                    "app_filename": "chrome.exe",
                    "app_normalized": "chrome",
                    "duration_minutes": 15.0,
                    "keystrokes": "[]",
                    "keystroke_count": 0,
                    "cross_app_keystrokes": 3,
                    "is_synthetic": True,
                    "is_orphan": True,
                    "synthetic_reason": "orphaned_keystrokes",
                },
            ]
        )
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        slices = result["atomic_slices"]
        assert len(slices) > 0
        s = slices[0]
        assert s.get("is_synthetic") is True
        assert s.get("is_orphan") is True
        assert s.get("synthetic_reason") == "orphaned_keystrokes"
        assert s.get("cross_app_keystrokes") == 3

    def test_provenance_defaults_for_non_synthetic(self):
        """Non-synthetic activities should get default provenance values."""
        df = self._make_activities(count=3)
        builder = WorkblockBuilder()
        result = builder.build_workblocks(df, "2026-01-01")
        for s in result["atomic_slices"]:
            assert s.get("is_synthetic") is False
            assert s.get("is_orphan") is False
            assert s.get("synthetic_reason") == ""
            assert s.get("cross_app_keystrokes") == 0

    def test_consolidation_does_not_merge_differing_provenance(self):
        """Adjacent slices with different provenance must NOT be consolidated."""
        base = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        builder = WorkblockBuilder()

        scenarios = [
            (
                "is_synthetic",
                {"is_synthetic": False, "is_orphan": False, "synthetic_reason": "", "cross_app_keystrokes": 0},
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "", "cross_app_keystrokes": 0},
            ),
            (
                "is_orphan",
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "orphaned_keystrokes", "cross_app_keystrokes": 0},
                {"is_synthetic": True, "is_orphan": True, "synthetic_reason": "orphaned_keystrokes", "cross_app_keystrokes": 0},
            ),
            (
                "synthetic_reason",
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "activity_gap", "cross_app_keystrokes": 0},
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "orphaned_keystrokes", "cross_app_keystrokes": 0},
            ),
            (
                "cross_app_keystrokes",
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "activity_gap", "cross_app_keystrokes": 0},
                {"is_synthetic": True, "is_orphan": False, "synthetic_reason": "activity_gap", "cross_app_keystrokes": 2},
            ),
        ]

        for field_name, first_provenance, second_provenance in scenarios:
            df = pl.DataFrame(
                [
                    {
                        "user_id": "u1",
                        "start": base,
                        "end": base + timedelta(minutes=5),
                        "name": "chrome.exe",
                        "title": "",
                        "url": "",
                        "exec": "chrome.exe",
                        "app_filename": "chrome.exe",
                        "app_normalized": "chrome",
                        "duration_minutes": 5.0,
                        "keystrokes": "[]",
                        "keystroke_count": 0,
                        **first_provenance,
                    },
                    {
                        "user_id": "u1",
                        "start": base + timedelta(minutes=5),
                        "end": base + timedelta(minutes=10),
                        "name": "chrome.exe",
                        "title": "",
                        "url": "",
                        "exec": "chrome.exe",
                        "app_filename": "chrome.exe",
                        "app_normalized": "chrome",
                        "duration_minutes": 5.0,
                        "keystrokes": "[]",
                        "keystroke_count": 0,
                        **second_provenance,
                    },
                ]
            )

            result = builder.build_workblocks(df, "2026-01-01")
            slices = result["atomic_slices"]

            assert len(result["workblocks"]) == 1, field_name
            assert len(slices) == 2, field_name

            first_slice, second_slice = slices
            assert first_slice["start_time"] == base.isoformat(), field_name
            assert first_slice["end_time"] == (base + timedelta(minutes=5)).isoformat(), field_name
            assert second_slice["start_time"] == (base + timedelta(minutes=5)).isoformat(), field_name
            assert second_slice["end_time"] == (base + timedelta(minutes=10)).isoformat(), field_name

            for key, value in first_provenance.items():
                assert first_slice.get(key) == value, f"{field_name}: first {key}"
            for key, value in second_provenance.items():
                assert second_slice.get(key) == value, f"{field_name}: second {key}"


class TestSubMinuteKeystrokeAssignment:
    """P1: Keystrokes should be assigned to segments by timestamp, not minute bucket."""

    def _make_row(self, start, end, name, keystrokes_json, **overrides):
        row = {
            "user_id": "u1",
            "start": start,
            "end": end,
            "name": name,
            "title": f"{name} window",
            "url": "",
            "exec": f"/usr/bin/{name.lower()}",
            "app_filename": f"{name.lower()}.exe",
            "app_normalized": name.lower(),
            "duration_minutes": (end - start).total_seconds() / 60,
            "keystrokes": keystrokes_json,
            "keystroke_count": 0,
            "cross_app_keystrokes": 0,
            "is_synthetic": False,
            "synthetic_reason": "",
        }
        row.update(overrides)
        return row

    def test_keystrokes_assigned_to_correct_sub_minute_segment(self):
        """A keystroke at 09:03:45 should only go to the segment containing that timestamp."""
        builder = WorkblockBuilder()
        base = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        df = pl.DataFrame([
            self._make_row(
                base, base + timedelta(minutes=3, seconds=30),
                "Chrome",
                json.dumps([{"id": "ks1", "time": "09:03:45", "text": "hello"}]),
            ),
            self._make_row(
                base + timedelta(minutes=3, seconds=30), base + timedelta(minutes=10),
                "Slack",
                json.dumps([]),
            ),
        ])

        result = builder.build_workblocks(df, "2024-01-01")
        slices = result["atomic_slices"]

        for s in slices:
            if s["start_time"] < "2024-01-01T09:03:30":
                assert s["keystroke_count"] == 0, (
                    f"Slice {s['start_time']}-{s['end_time']} should have 0 keystrokes, "
                    f"got {s['keystroke_count']} (keystroke leaked to wrong segment)"
                )

        slices_after = [s for s in slices if s["start_time"] >= "2024-01-01T09:03:30"
                        and s["keystroke_count"] > 0]
        assert len(slices_after) >= 1, "Keystroke at 09:03:45 not found in any post-boundary slice"

    def test_keystrokes_without_seconds_default_to_zero(self):
        """Keystrokes with HH:MM format (no seconds) should default seconds to 0."""
        builder = WorkblockBuilder()
        base = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        df = pl.DataFrame([
            self._make_row(
                base, base + timedelta(minutes=3, seconds=30),
                "Chrome",
                json.dumps([{"id": "ks1", "time": "09:03", "text": "no seconds"}]),
            ),
            self._make_row(
                base + timedelta(minutes=3, seconds=30), base + timedelta(minutes=10),
                "Slack",
                json.dumps([]),
            ),
        ])

        result = builder.build_workblocks(df, "2024-01-01")
        slices = result["atomic_slices"]

        slices_with_ks = [s for s in slices if s["keystroke_count"] > 0]
        assert len(slices_with_ks) == 1
        # start_time may have tz suffix (+00:00), so compare the prefix
        assert slices_with_ks[0]["start_time"][:19] <= "2024-01-01T09:03:00"
        assert slices_with_ks[0]["end_time"][:19] > "2024-01-01T09:03:00"


class TestAppAwarePrimaryActivity:
    """P2: Primary activity should match keystroke source app, not blindly use activities[0]."""

    def _make_row(self, start, end, name, keystrokes_json, **overrides):
        row = {
            "user_id": "u1",
            "start": start,
            "end": end,
            "name": name,
            "title": f"{name} window",
            "url": "",
            "exec": f"/usr/bin/{name.lower()}",
            "app_filename": f"{name.lower()}.exe",
            "app_normalized": name.lower(),
            "duration_minutes": (end - start).total_seconds() / 60,
            "keystrokes": keystrokes_json,
            "keystroke_count": 0,
            "cross_app_keystrokes": 0,
            "is_synthetic": False,
            "synthetic_reason": "",
        }
        row.update(overrides)
        return row

    def test_primary_activity_matches_keystroke_source_app(self):
        """If a Slack keystroke is in a segment with both Chrome and Slack,
        the slice should be labeled as slack.exe, not chrome.exe."""
        builder = WorkblockBuilder()
        base = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        df = pl.DataFrame([
            self._make_row(base, base + timedelta(minutes=10), "Chrome", json.dumps([])),
            self._make_row(
                base, base + timedelta(minutes=10), "Slack",
                json.dumps([{"id": "ks1", "time": "09:05:00", "text": "hey team"}]),
            ),
        ])

        result = builder.build_workblocks(df, "2024-01-01")

        slices_with_ks = [s for s in result["atomic_slices"] if s["keystroke_count"] > 0]
        assert len(slices_with_ks) >= 1

        ks_slice = slices_with_ks[0]
        assert ks_slice["application"] == "slack.exe", (
            f"Expected 'slack.exe'-labeled slice, got '{ks_slice['application']}'"
        )

    def test_no_keystrokes_falls_back_to_first_activity(self):
        """When a segment has no keystrokes, fall back to activities[0]."""
        builder = WorkblockBuilder()
        base = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)

        df = pl.DataFrame([
            self._make_row(base, base + timedelta(minutes=5), "Chrome", json.dumps([])),
            self._make_row(base, base + timedelta(minutes=5), "Slack", json.dumps([])),
        ])

        result = builder.build_workblocks(df, "2024-01-01")
        assert len(result["atomic_slices"]) > 0
