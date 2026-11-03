import json
from datetime import datetime, timedelta, timezone
from itertools import permutations
import random

import polars as pl
import pytest

from workblock_engine import process
from workblock_engine.building.overlap import resolve_activity_overlaps
from workblock_engine.building.workblock_builder import WorkblockBuilder
from workblock_engine.cleaning.activity_cleaner import ActivityCleaner
from workblock_engine.cleaning.keystroke_cleaner import KeystrokeCleaner
from workblock_engine.matching.keystroke_matcher import join_keystrokes_to_activities


BASE = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def activity(identity, start=0, end=600, user_id="test-user"):
    return {
        "id": identity,
        "user_id": user_id,
        "timestamp": BASE + timedelta(seconds=start),
        "start": BASE + timedelta(seconds=start),
        "end": BASE + timedelta(seconds=end),
        "name": "editor",
        "exec": "editor",
        "title": identity,
        "url": "",
        "task_name": "",
    }


def keystroke(identity, text, second=5, user_id="test-user", url=""):
    return {
        "id": identity,
        "user_id": user_id,
        "timestamp": BASE + timedelta(seconds=second),
        "app": "editor",
        "keystrokes": text,
        "window_title": "editor",
        "url": url,
    }


def payload(result, user_id="test-user"):
    assert result.skipped_users == []
    return (
        result.workblocks.filter(pl.col("user_id") == user_id).to_dicts(),
        result.atomic_slices.filter(pl.col("user_id") == user_id).to_dicts(),
    )


def raw_keys(result):
    return [
        key
        for row in result.atomic_slices.to_dicts()
        for key in json.loads(row["raw_keystrokes"])
    ]


@pytest.mark.parametrize("with_ids", [False, True])
def test_equal_start_preserves_longest_end_for_every_input_order(with_ids):
    rows = [activity("long", end=14), activity("short", end=9), activity("tail", 25)]
    if not with_ids:
        rows = [{k: v for k, v in row.items() if k != "id"} for row in rows]
    expected = None
    for ordered in permutations(rows):
        resolved = resolve_activity_overlaps(pl.DataFrame(ordered))
        positive = resolved.filter(pl.col("end") > pl.col("start"))
        assert positive["title"].to_list() == ["long", "tail"]
        assert positive["end"].to_list() == [
            BASE + timedelta(seconds=14),
            BASE + timedelta(seconds=600),
        ]
        result = process(pl.DataFrame(ordered), pl.DataFrame())
        assert result.atomic_slices["end_time"][0] == "2026-01-01T09:00:14+00:00"
        actual = payload(result)
        if expected is None:
            expected = actual
        assert actual == expected


@pytest.mark.parametrize("with_ids", [False, True])
def test_equal_time_keystrokes_have_canonical_json_order(with_ids):
    keys = [keystroke("key-a", "alpha", url=None), keystroke("key-b", "beta")]
    if not with_ids:
        keys = [{k: v for k, v in row.items() if k != "id"} for row in keys]
    expected = None
    for repeat in range(20):
        ordered = keys if repeat % 2 else list(reversed(keys))
        result = process(pl.DataFrame([activity("editor")]), pl.DataFrame(ordered))
        emitted = raw_keys(result)
        assert len(emitted) == 2
        if with_ids:
            assert [key["id"] for key in emitted] == ["key-a", "key-b"]
        else:
            assert [str(key["id"]) for key in emitted] == sorted(
                str(key["id"]) for key in emitted
            )
        assert {key["text"]: key["url"] for key in emitted} == {
            "alpha": None,
            "beta": "",
        }
        actual = payload(result)
        if expected is None:
            expected = actual
        assert actual == expected


@pytest.mark.parametrize(
    "cleaner,rows,field,expected",
    [
        (
            ActivityCleaner,
            [activity("same", end=900), activity("same", end=600)],
            "end",
            BASE + timedelta(seconds=600),
        ),
        (
            ActivityCleaner,
            [
                dict(activity("same"), title="zulu"),
                dict(activity("same"), title="alpha"),
            ],
            "title",
            "alpha",
        ),
        (
            KeystrokeCleaner,
            [keystroke("same", "zulu"), keystroke("same", "alpha")],
            "keystrokes",
            "alpha",
        ),
        (
            KeystrokeCleaner,
            [
                keystroke("same", "alpha", url="https://example.test"),
                keystroke("same", "alpha", url=None),
            ],
            "url",
            None,
        ),
    ],
)
def test_conflicting_duplicate_ids_choose_canonical_record(
    cleaner, rows, field, expected
):
    for repeat in range(20):
        ordered = rows if repeat % 2 else list(reversed(rows))
        cleaned = cleaner().clean(pl.DataFrame(ordered))
        assert cleaned.height == 1
        assert cleaned[field][0] == expected


@pytest.mark.parametrize("second", [60, 75])
def test_matching_ties_choose_first_chronological_activity(second):
    rows = [
        activity("first", end=60),
        activity("second", start=60 if second == 60 else 90),
    ]
    keys = [keystroke("key", "typed", second=second)]
    for _ in range(20):
        matched = join_keystrokes_to_activities(
            pl.DataFrame(rows), pl.DataFrame(keys), "test-user"
        )
        counts = dict(zip(matched["title"], matched["keystroke_count"]))
        assert counts == {"first": 1, "second": 0}


@pytest.mark.parametrize("with_ids", [False, True])
def test_replays_and_mixed_user_batches_match_individual_results(with_ids):
    rows = [activity("long", end=14), activity("short", end=9), activity("tail", 25)]
    keys = [keystroke("key-a", "alpha", 30), keystroke("key-b", "beta", 30)]
    peer_rows = [dict(row, user_id="peer", id="peer-" + row["id"]) for row in rows]
    peer_keys = [dict(row, user_id="peer", id="peer-" + row["id"]) for row in keys]
    if not with_ids:
        rows, keys, peer_rows, peer_keys = [
            [{k: v for k, v in row.items() if k != "id"} for row in records]
            for records in (rows, keys, peer_rows, peer_keys)
        ]
    expected = payload(process(pl.DataFrame(rows), pl.DataFrame(keys)))
    peer_expected = payload(
        process(pl.DataFrame(peer_rows), pl.DataFrame(peer_keys)), "peer"
    )
    for repeat in range(20):
        mixed_rows, mixed_keys = rows + peer_rows, keys + peer_keys
        random.Random(repeat).shuffle(mixed_rows)
        random.Random(repeat + 100).shuffle(mixed_keys)
        result = process(pl.DataFrame(mixed_rows), pl.DataFrame(mixed_keys))
        assert payload(result) == expected
        assert payload(result, "peer") == peer_expected
        assert sorted(key["text"] for key in raw_keys(result)) == [
            "alpha",
            "alpha",
            "beta",
            "beta",
        ]


@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize("cross_midnight", [False, True])
def test_builder_orders_tied_activities_before_and_after_day_split(
    with_ids, cross_midnight
):
    rows = [activity("zulu"), activity("alpha")]
    if cross_midnight:
        offset = timedelta(hours=14, minutes=55)
        rows = [
            {
                key: value + offset if isinstance(value, datetime) else value
                for key, value in row.items()
            }
            for row in rows
        ]
    if not with_ids:
        rows = [{k: v for k, v in row.items() if k != "id"} for row in rows]
    expected = None
    for ordered in permutations(rows):
        cleaned = ActivityCleaner().clean(pl.DataFrame(ordered))
        matched = join_keystrokes_to_activities(cleaned, pl.DataFrame(), "test-user")
        result = WorkblockBuilder().build_workblocks(matched, "2026-01-01")
        assert len(result["workblocks"]) == (2 if cross_midnight else 1)
        assert {row["window_title"] for row in result["atomic_slices"]} == {"alpha"}
        actual = result["workblocks"], result["atomic_slices"]
        if expected is None:
            expected = actual
        assert actual == expected
