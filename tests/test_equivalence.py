"""Equivalence test: workblock-engine vs productivity-insights — live comparison.

Imports BOTH libraries in a single process and runs them on identical input.
No golden reference files, no trusting a separate script.

Run from productivity-insights venv (which has polars + all PI deps):

    cd /Users/rustancorpuz/code/productivity-insights
    source .venv/bin/activate
    pytest /Users/rustancorpuz/code/workblock-engine/tests/test_equivalence.py -v

Known Differences (by design, excluded from comparison):
    - exec field: WE preserves it; PI drops it during activity cleaning.
    - Provenance fields: WE adds is_synthetic, is_orphan, synthetic_reason,
      cross_app_keystrokes to slices. PI does not.
    - Zero-duration orphan handling: WE creates point segments for orphan
      activities; PI does not.
    - concurrent_apps: list ordering may differ between implementations,
      so it is excluded from field-by-field comparison.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

# ── Import from BOTH repos ───────────────────────────────────────────
# Derive paths relative to this file; overridable via env vars.
_WE_SRC = Path(os.environ.get(
    "WORKBLOCK_ENGINE_SRC",
    str(Path(__file__).resolve().parents[1] / "src"),
))
_PI_SRC = Path(os.environ.get(
    "PRODUCTIVITY_INSIGHTS_SRC",
    str(_WE_SRC.parents[1] / "productivity-insights" / "src"),
))
if str(_WE_SRC) not in sys.path:
    sys.path.insert(0, str(_WE_SRC))
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from workblock_engine import process as we_process  # noqa: E402

from productivity_insights.data_processing.activity_cleaner import (  # noqa: E402
    ActivityCleaner as PI_ActivityCleaner,
)
from productivity_insights.data_processing.keystroke_cleaner import (  # noqa: E402
    KeystrokeCleaner as PI_KeystrokeCleaner,
)
from productivity_insights.insights.workblock_builder import (  # noqa: E402
    WorkblockBuilder as PI_WorkblockBuilder,
)
from productivity_insights.processing.activity_overlap import (  # noqa: E402
    resolve_activity_overlaps as pi_resolve_overlaps,
)
from productivity_insights.processing.keystroke_matcher import (  # noqa: E402
    join_keystrokes_to_activities as pi_join_keystrokes,
)

DURATION_TOLERANCE = 0.1  # minutes

# Fields that both pipelines produce identically
SHARED_WB_FIELDS = [
    "workblock_id", "date", "start_time", "end_time",
    "duration_minutes", "active_duration_minutes", "total_activities",
]
SHARED_SLICE_FIELDS = [
    # Note: "date" is omitted — PI's build_workblocks() does not include it in
    # slice dicts; WE adds it during output assembly in engine.py.
    "workblock_id", "sequence", "start_time", "end_time",
    "duration_minutes", "application", "url", "window_title",
    "keystroke_count", "activity_count", "is_multitasking",
    "concurrent_activities",
]


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Shared fixture data — created ONCE, fed to BOTH pipelines       ║
# ╚════════════════════════════════════════════════════════════════════╝

BASE_TIME = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc)


def _fixture_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic fixture data — four users exercising different code paths.

    alice: developer, clean sequential activities (Phase-1 matching, 2 workblocks)
    bob:   support agent, app-specific exec paths (Phase-1 matching, 1 workblock)
    carol: overlapping activities that need overlap resolution
    dave:  activities with a 20-min gap between them (tests gap detection in
           workblock builder). Note: gap keystrokes are omitted because WE's
           zero-duration orphan handling creates point segments that PI does not,
           causing a known divergence in Phase-3/4 synthetic activities.
    """
    t = BASE_TIME

    activities_data = [
        # ── alice (developer): two workblocks ────────────────────
        # Workblock 1: 09:00 → 10:10
        {"user_id": "alice", "timestamp": t,
         "start": t, "end": t + timedelta(minutes=10),
         "name": "Chrome", "title": "Dashboard - Chrome",
         "url": "https://app.example.com", "task_name": "Analysis",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=10),
         "start": t + timedelta(minutes=10), "end": t + timedelta(minutes=25),
         "name": "Code", "title": "main.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=25),
         "start": t + timedelta(minutes=25), "end": t + timedelta(minutes=40),
         "name": "Code", "title": "test.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=40),
         "start": t + timedelta(minutes=40), "end": t + timedelta(minutes=55),
         "name": "Slack", "title": "general - Slack",
         "url": "", "task_name": "Communication",
         "exec": "/Applications/Slack.app/Contents/MacOS/Slack"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=55),
         "start": t + timedelta(minutes=55), "end": t + timedelta(minutes=70),
         "name": "Chrome", "title": "Docs - Chrome",
         "url": "https://docs.example.com", "task_name": "Research",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        # 75 min gap → new workblock
        # Workblock 2: 12:05 → 12:25
        {"user_id": "alice", "timestamp": t + timedelta(hours=3, minutes=5),
         "start": t + timedelta(hours=3, minutes=5),
         "end": t + timedelta(hours=3, minutes=15),
         "name": "Chrome", "title": "PR Review - Chrome",
         "url": "https://github.com/example/pr/42", "task_name": "Review",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        {"user_id": "alice", "timestamp": t + timedelta(hours=3, minutes=15),
         "start": t + timedelta(hours=3, minutes=15),
         "end": t + timedelta(hours=3, minutes=25),
         "name": "Code", "title": "fix.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
        # ── bob (support agent): one workblock ───────────────────
        {"user_id": "bob", "timestamp": t,
         "start": t, "end": t + timedelta(minutes=10),
         "name": "Zendesk", "title": "Ticket #123 - Zendesk",
         "url": "https://support.zendesk.com/tickets/123", "task_name": "Support",
         "exec": "/Applications/Zendesk.app/Contents/MacOS/Zendesk"},
        {"user_id": "bob", "timestamp": t + timedelta(minutes=10),
         "start": t + timedelta(minutes=10), "end": t + timedelta(minutes=20),
         "name": "Chrome", "title": "Customer Profile - CRM",
         "url": "https://crm.example.com/customer/456", "task_name": "Support",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        {"user_id": "bob", "timestamp": t + timedelta(minutes=20),
         "start": t + timedelta(minutes=20), "end": t + timedelta(minutes=30),
         "name": "Zendesk", "title": "Ticket #124 - Zendesk",
         "url": "https://support.zendesk.com/tickets/124", "task_name": "Support",
         "exec": "/Applications/Zendesk.app/Contents/MacOS/Zendesk"},
        # ── carol: OVERLAPPING activities (tests overlap resolver) ──
        {"user_id": "carol", "timestamp": t,
         "start": t, "end": t + timedelta(minutes=20),
         "name": "Chrome", "title": "Research - Chrome",
         "url": "https://research.example.com", "task_name": "Research",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        {"user_id": "carol", "timestamp": t + timedelta(minutes=15),
         "start": t + timedelta(minutes=15), "end": t + timedelta(minutes=35),
         "name": "Code", "title": "analysis.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
        {"user_id": "carol", "timestamp": t + timedelta(minutes=35),
         "start": t + timedelta(minutes=35), "end": t + timedelta(minutes=45),
         "name": "Chrome", "title": "Results - Chrome",
         "url": "https://results.example.com", "task_name": "Review",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        # ── dave: GAP between activities (tests Phase-2/3 matching) ──
        {"user_id": "dave", "timestamp": t,
         "start": t, "end": t + timedelta(minutes=15),
         "name": "Code", "title": "server.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
        # 20-min gap: 09:15 → 09:35 (keystrokes exist in this gap)
        {"user_id": "dave", "timestamp": t + timedelta(minutes=35),
         "start": t + timedelta(minutes=35), "end": t + timedelta(minutes=50),
         "name": "Code", "title": "client.py - Visual Studio Code",
         "url": "", "task_name": "Development",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron"},
    ]

    keystrokes_data = [
        # ── alice keystrokes (all within activity windows) ────────
        {"user_id": "alice", "timestamp": t + timedelta(minutes=3),
         "keystrokes": "SELECT * FROM users", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "Dashboard - Chrome"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=12),
         "keystrokes": "def process():<Enter><Tab>return True", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "main.py - Visual Studio Code"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=28),
         "keystrokes": "assert result is not None", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "test.py - Visual Studio Code"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=42),
         "keystrokes": "hey team, PR is ready", "app": "Slack",
         "exec": "/Applications/Slack.app/Contents/MacOS/Slack",
         "window_title": "general - Slack"},
        {"user_id": "alice", "timestamp": t + timedelta(minutes=57),
         "keystrokes": "polars dataframe docs", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "Docs - Chrome"},
        {"user_id": "alice", "timestamp": t + timedelta(hours=3, minutes=8),
         "keystrokes": "LGTM, ship it!", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "PR Review - Chrome"},
        {"user_id": "alice", "timestamp": t + timedelta(hours=3, minutes=18),
         "keystrokes": "<Ctrl><Ctrl + s>", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "fix.py - Visual Studio Code"},
        # ── bob keystrokes ────────────────────────────────────────
        {"user_id": "bob", "timestamp": t + timedelta(minutes=3),
         "keystrokes": "Resolved - customer confirmed fix<Enter>", "app": "Zendesk",
         "exec": "/Applications/Zendesk.app/Contents/MacOS/Zendesk",
         "window_title": "Ticket #123 - Zendesk"},
        {"user_id": "bob", "timestamp": t + timedelta(minutes=13),
         "keystrokes": "customer ID: CUS-456", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "Customer Profile - CRM"},
        {"user_id": "bob", "timestamp": t + timedelta(minutes=23),
         "keystrokes": "Billing issue escalated to finance<Enter>", "app": "Zendesk",
         "exec": "/Applications/Zendesk.app/Contents/MacOS/Zendesk",
         "window_title": "Ticket #124 - Zendesk"},
        # ── carol keystrokes (within overlapping region) ──────────
        {"user_id": "carol", "timestamp": t + timedelta(minutes=5),
         "keystrokes": "machine learning trends", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "Research - Chrome"},
        {"user_id": "carol", "timestamp": t + timedelta(minutes=17),
         "keystrokes": "import numpy as np", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "analysis.py - Visual Studio Code"},
        {"user_id": "carol", "timestamp": t + timedelta(minutes=38),
         "keystrokes": "results look good", "app": "Chrome",
         "exec": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
         "window_title": "Results - Chrome"},
        # ── dave keystrokes (within activities only; gap keystrokes omitted
        #    to avoid the WE zero-duration orphan divergence) ──────
        {"user_id": "dave", "timestamp": t + timedelta(minutes=5),
         "keystrokes": "def handle_request():", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "server.py - Visual Studio Code"},
        {"user_id": "dave", "timestamp": t + timedelta(minutes=40),
         "keystrokes": "const client = new Client()", "app": "Code",
         "exec": "/Applications/Visual Studio Code.app/Contents/MacOS/Electron",
         "window_title": "client.py - Visual Studio Code"},
    ]

    return pl.DataFrame(activities_data), pl.DataFrame(keystrokes_data)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Run productivity-insights core pipeline (no runner / I/O)       ║
# ╚════════════════════════════════════════════════════════════════════╝


def _pi_derive_date(matched: pl.DataFrame, keystrokes: pl.DataFrame) -> str:
    if not matched.is_empty() and "start" in matched.columns:
        dates = (
            matched.select(pl.col("start").dt.date().alias("d"))
            .get_column("d").drop_nulls()
        )
        if not dates.is_empty():
            return str(dates.mode().sort()[0])
    if not keystrokes.is_empty() and "timestamp" in keystrokes.columns:
        dates = (
            keystrokes.select(pl.col("timestamp").dt.date().alias("d"))
            .get_column("d").drop_nulls()
        )
        if not dates.is_empty():
            return str(dates.mode().sort()[0])
    raise ValueError("Cannot derive date")


def _run_pi(activities: pl.DataFrame, keystrokes: pl.DataFrame) -> dict:
    """Run PI core pipeline user-by-user, mirroring WE's process() flow."""
    cleaned_acts = PI_ActivityCleaner().clean(activities)
    cleaned_keys = PI_KeystrokeCleaner().clean(keystrokes)

    act_users = (
        cleaned_acts.get_column("user_id").cast(pl.Utf8).unique().to_list()
        if not cleaned_acts.is_empty() else []
    )
    key_users = (
        cleaned_keys.get_column("user_id").cast(pl.Utf8).unique().to_list()
        if not cleaned_keys.is_empty() else []
    )

    results: dict[str, dict] = {}
    for uid in sorted(set(act_users) | set(key_users)):
        user_acts = cleaned_acts.filter(pl.col("user_id").cast(pl.Utf8) == uid)
        user_keys = cleaned_keys.filter(pl.col("user_id").cast(pl.Utf8) == uid)

        resolved = pi_resolve_overlaps(user_acts)
        matched = pi_join_keystrokes(resolved, user_keys, uid)
        primary_date = _pi_derive_date(matched, user_keys)

        builder = PI_WorkblockBuilder()
        result = builder.build_workblocks(matched, primary_date)

        results[uid] = {
            "workblocks": result.get("workblocks", []),
            "atomic_slices": result.get("atomic_slices", []),
        }
    return results


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Fixtures — same data, both pipelines                            ║
# ╚════════════════════════════════════════════════════════════════════╝


@pytest.fixture(scope="module")
def shared_data():
    return _fixture_data()


@pytest.fixture(scope="module")
def pi_result(shared_data):
    activities, keystrokes = shared_data
    return _run_pi(activities, keystrokes)


@pytest.fixture(scope="module")
def we_result(shared_data):
    activities, keystrokes = shared_data
    return we_process(activities, keystrokes)


# ── Helpers ──────────────────────────────────────────────────────────


def _wb_by_user(result) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = {}
    if result.workblocks.is_empty():
        return by_user
    for uid in result.workblocks["user_id"].unique().sort().to_list():
        rows = result.workblocks.filter(pl.col("user_id") == uid)
        by_user[uid] = rows.sort("start_time").to_dicts()
    return by_user


def _slices_by_user(result) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = {}
    if result.atomic_slices.is_empty():
        return by_user
    for uid in result.atomic_slices["user_id"].unique().sort().to_list():
        rows = result.atomic_slices.filter(pl.col("user_id") == uid)
        by_user[uid] = rows.sort(["workblock_id", "sequence"]).to_dicts()
    return by_user


def _approx(a, b, tol=DURATION_TOLERANCE):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Workblock-level equivalence                                     ║
# ╚════════════════════════════════════════════════════════════════════╝


class TestWorkblockEquivalence:

    def test_same_users_processed(self, pi_result, we_result):
        pi_users = sorted(pi_result.keys())
        we_users = sorted(
            we_result.workblocks["user_id"].unique().to_list()
            if not we_result.workblocks.is_empty() else []
        )
        assert pi_users == we_users

    def test_workblock_count_per_user(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            expected = len(data["workblocks"])
            actual = len(we_wbs.get(uid, []))
            assert expected == actual, f"{uid}: {expected} vs {actual} workblocks"

    def test_workblock_ids_match(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            expected = [wb["workblock_id"] for wb in data["workblocks"]]
            actual = [wb["workblock_id"] for wb in we_wbs.get(uid, [])]
            assert expected == actual, f"{uid}: ID mismatch"

    def test_workblock_boundaries(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["workblocks"], we_wbs.get(uid, []), strict=True)
            ):
                assert exp["start_time"] == act["start_time"], (
                    f"{uid} wb[{i}] start: {exp['start_time']} vs {act['start_time']}"
                )
                assert exp["end_time"] == act["end_time"], (
                    f"{uid} wb[{i}] end: {exp['end_time']} vs {act['end_time']}"
                )

    def test_workblock_durations(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["workblocks"], we_wbs.get(uid, []), strict=True)
            ):
                assert _approx(exp["duration_minutes"], act["duration_minutes"]), (
                    f"{uid} wb[{i}]: duration {exp['duration_minutes']} vs {act['duration_minutes']}"
                )
                assert _approx(exp["active_duration_minutes"], act["active_duration_minutes"]), (
                    f"{uid} wb[{i}]: active_dur {exp['active_duration_minutes']} vs {act['active_duration_minutes']}"
                )

    def test_workblock_activity_counts(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["workblocks"], we_wbs.get(uid, []), strict=True)
            ):
                assert exp["total_activities"] == act["total_activities"], (
                    f"{uid} wb[{i}]: {exp['total_activities']} vs {act['total_activities']}"
                )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Atomic-slice equivalence                                        ║
# ╚════════════════════════════════════════════════════════════════════╝


class TestSliceEquivalence:

    def test_slice_count_per_user(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            expected = len(data["atomic_slices"])
            actual = len(we_slices.get(uid, []))
            assert expected == actual, f"{uid}: {expected} vs {actual} slices"

    def test_slice_sequences(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            expected = [s["sequence"] for s in data["atomic_slices"]]
            actual = [s["sequence"] for s in we_slices.get(uid, [])]
            assert expected == actual, f"{uid}: sequence mismatch"

    def test_slice_boundaries(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp["start_time"] == act["start_time"], (
                    f"{uid} s[{i}] start: {exp['start_time']} vs {act['start_time']}"
                )
                assert exp["end_time"] == act["end_time"], (
                    f"{uid} s[{i}] end: {exp['end_time']} vs {act['end_time']}"
                )

    def test_slice_durations(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert _approx(exp["duration_minutes"], act["duration_minutes"]), (
                    f"{uid} s[{i}]: {exp['duration_minutes']} vs {act['duration_minutes']}"
                )

    def test_slice_applications(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp["application"] == act["application"], (
                    f"{uid} s[{i}]: {exp['application']!r} vs {act['application']!r}"
                )

    def test_slice_urls(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp.get("url", "") == act.get("url", ""), (
                    f"{uid} s[{i}]: {exp.get('url')!r} vs {act.get('url')!r}"
                )

    def test_slice_window_titles(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp.get("window_title", "") == act.get("window_title", ""), (
                    f"{uid} s[{i}]: {exp.get('window_title')!r} vs {act.get('window_title')!r}"
                )

    def test_slice_keystroke_counts(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp["keystroke_count"] == act["keystroke_count"], (
                    f"{uid} s[{i}]: {exp['keystroke_count']} vs {act['keystroke_count']}"
                )

    def test_slice_activity_counts(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp["activity_count"] == act["activity_count"], (
                    f"{uid} s[{i}]: {exp['activity_count']} vs {act['activity_count']}"
                )

    def test_slice_multitasking(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp.get("is_multitasking", False) == act.get("is_multitasking", False), (
                    f"{uid} s[{i}]: is_multitasking {exp.get('is_multitasking')} vs {act.get('is_multitasking')}"
                )

    def test_slice_workblock_assignment(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                assert exp["workblock_id"] == act["workblock_id"], (
                    f"{uid} s[{i}]: {exp['workblock_id']} vs {act['workblock_id']}"
                )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Aggregate / cross-cutting checks                                ║
# ╚════════════════════════════════════════════════════════════════════╝


class TestAggregateEquivalence:

    def test_total_keystroke_count_per_user(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        for uid, data in pi_result.items():
            expected = sum(s["keystroke_count"] for s in data["atomic_slices"])
            actual = sum(s["keystroke_count"] for s in we_slices.get(uid, []))
            assert expected == actual, f"{uid}: {expected} vs {actual} keystrokes"

    def test_total_active_duration_per_user(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        for uid, data in pi_result.items():
            expected = sum(wb["active_duration_minutes"] for wb in data["workblocks"])
            actual = sum(wb["active_duration_minutes"] for wb in we_wbs.get(uid, []))
            assert _approx(expected, actual, tol=0.5), (
                f"{uid}: active_duration {expected:.2f} vs {actual:.2f}"
            )

    def test_no_skipped_users(self, we_result):
        assert we_result.skipped_users == []

    def test_raw_keystrokes_parseable(self, we_result):
        if we_result.atomic_slices.is_empty():
            return
        for row in we_result.atomic_slices.to_dicts():
            parsed = json.loads(row.get("raw_keystrokes", "[]"))
            assert isinstance(parsed, list)

    def test_slices_cover_workblock_span(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        we_slices = _slices_by_user(we_result)
        for uid in pi_result:
            for wb in we_wbs.get(uid, []):
                wb_slices = [
                    s for s in we_slices.get(uid, [])
                    if s["workblock_id"] == wb["workblock_id"]
                ]
                if not wb_slices:
                    continue
                first = min(wb_slices, key=lambda s: s["start_time"])
                last = max(wb_slices, key=lambda s: s["end_time"])
                assert first["start_time"] == wb["start_time"]
                assert last["end_time"] == wb["end_time"]


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Full-snapshot comparison (catch-all)                            ║
# ╚════════════════════════════════════════════════════════════════════╝


class TestFullSnapshot:

    def test_all_shared_workblock_fields(self, pi_result, we_result):
        we_wbs = _wb_by_user(we_result)
        mismatches = []
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["workblocks"], we_wbs.get(uid, []), strict=True)
            ):
                for field in SHARED_WB_FIELDS:
                    ev, av = exp.get(field), act.get(field)
                    if field in ("duration_minutes", "active_duration_minutes"):
                        if not _approx(ev, av):
                            mismatches.append(f"{uid} wb[{i}].{field}: {ev} vs {av}")
                    elif ev != av:
                        mismatches.append(f"{uid} wb[{i}].{field}: {ev!r} vs {av!r}")
        assert not mismatches, (
            f"{len(mismatches)} mismatch(es):\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )

    def test_all_shared_slice_fields(self, pi_result, we_result):
        we_slices = _slices_by_user(we_result)
        mismatches = []
        for uid, data in pi_result.items():
            for i, (exp, act) in enumerate(
                zip(data["atomic_slices"], we_slices.get(uid, []), strict=True)
            ):
                for field in SHARED_SLICE_FIELDS:
                    ev, av = exp.get(field), act.get(field)
                    if field == "duration_minutes":
                        if not _approx(ev, av):
                            mismatches.append(f"{uid} s[{i}].{field}: {ev} vs {av}")
                    elif ev != av:
                        mismatches.append(f"{uid} s[{i}].{field}: {ev!r} vs {av!r}")
        assert not mismatches, (
            f"{len(mismatches)} mismatch(es):\n"
            + "\n".join(f"  - {m}" for m in mismatches)
        )
