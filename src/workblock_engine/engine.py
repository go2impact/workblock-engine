"""Workblock engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging

import polars as pl

from workblock_engine.building.overlap import resolve_activity_overlaps
from workblock_engine.building.workblock_builder import WorkblockBuilder
from workblock_engine.cleaning.activity_cleaner import ActivityCleaner
from workblock_engine.cleaning.keystroke_cleaner import KeystrokeCleaner
from workblock_engine.matching.keystroke_matcher import join_keystrokes_to_activities

logger = logging.getLogger(__name__)

WORKBLOCK_SCHEMA: dict[str, pl.DataType] = {
    "workblock_id": pl.String,
    "user_id": pl.String,
    "date": pl.String,
    "start_time": pl.String,
    "end_time": pl.String,
    "duration_minutes": pl.Float64,
    "active_duration_minutes": pl.Float64,
    "total_activities": pl.Int64,
}

SLICE_SCHEMA: dict[str, pl.DataType] = {
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


@dataclass(frozen=True)
class WorkblockResult:
    workblocks: pl.DataFrame
    atomic_slices: pl.DataFrame
    skipped_users: list[str]


def process(
    activities: pl.DataFrame,
    keystrokes: pl.DataFrame,
    *,
    cross_day_gap_minutes: int = 180,
) -> WorkblockResult:
    """Process activities and keystrokes for all users into workblocks."""
    cleaned_activities = _clean_activities(activities)
    cleaned_keystrokes = _clean_keystrokes(keystrokes)

    activity_users = (
        cleaned_activities.get_column("user_id").cast(pl.Utf8).unique().to_list()
        if "user_id" in cleaned_activities.columns
        else []
    )
    keystroke_users = (
        cleaned_keystrokes.get_column("user_id").cast(pl.Utf8).unique().to_list()
        if "user_id" in cleaned_keystrokes.columns
        else []
    )
    user_ids = sorted(set(activity_users) | set(keystroke_users))

    all_workblocks: list[pl.DataFrame] = []
    all_slices: list[pl.DataFrame] = []
    skipped_users: list[str] = []

    for user_id in user_ids:
        try:
            user_activities = cleaned_activities.filter(
                pl.col("user_id").cast(pl.Utf8) == user_id
            )
            user_keystrokes = cleaned_keystrokes.filter(
                pl.col("user_id").cast(pl.Utf8) == user_id
            )

            resolved = _resolve_overlaps(user_activities)
            matched = _match_keystrokes(resolved, user_keystrokes, user_id)
            primary_date = _derive_primary_date(matched, user_keystrokes)
            workblocks_df, slices_df = _build_workblocks(
                matched,
                primary_date,
                cross_day_gap_minutes,
            )

            all_workblocks.append(_assemble_workblocks(workblocks_df, user_id))
            all_slices.append(_assemble_slices(slices_df, user_id))
        except Exception:
            logger.warning("Failed to process user %s, skipping", user_id, exc_info=True)
            skipped_users.append(user_id)

    return WorkblockResult(
        workblocks=pl.concat(all_workblocks) if all_workblocks else _empty_workblocks(),
        atomic_slices=pl.concat(all_slices) if all_slices else _empty_slices(),
        skipped_users=skipped_users,
    )


def _clean_activities(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in df.columns and df.schema["timestamp"] == pl.Utf8:
        df = df.with_columns(_coerce_datetime_column("timestamp"))
    if "start" in df.columns and df.schema["start"] == pl.Utf8:
        df = df.with_columns(_coerce_datetime_column("start"))
    if "end" in df.columns and df.schema["end"] == pl.Utf8:
        df = df.with_columns(_coerce_datetime_column("end"))
    if "task_name" not in df.columns:
        df = df.with_columns(pl.lit("").alias("task_name"))
    return ActivityCleaner().clean(df)


def _clean_keystrokes(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in df.columns and df.schema["timestamp"] == pl.Utf8:
        df = df.with_columns(_coerce_datetime_column("timestamp"))
    return KeystrokeCleaner().clean(df)


def _resolve_overlaps(activities_df: pl.DataFrame) -> pl.DataFrame:
    return resolve_activity_overlaps(activities_df)


def _match_keystrokes(
    activities_df: pl.DataFrame,
    keystrokes_df: pl.DataFrame,
    user_id: str,
) -> pl.DataFrame:
    return join_keystrokes_to_activities(activities_df, keystrokes_df, user_id)


def _build_workblocks(
    matched_activities: pl.DataFrame,
    primary_date: str,
    cross_day_gap_minutes: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    builder = WorkblockBuilder(cross_day_gap_minutes=cross_day_gap_minutes)
    result = builder.build_workblocks(matched_activities, primary_date)

    workblocks = result.get("workblocks", [])
    atomic_slices = result.get("atomic_slices", [])

    workblocks_df = (
        pl.DataFrame(workblocks) if workblocks else _empty_workblocks()
    )
    slices_df = pl.DataFrame(atomic_slices) if atomic_slices else _empty_slices()
    return workblocks_df, slices_df


def _derive_primary_date(
    matched_activities: pl.DataFrame,
    user_keystrokes: pl.DataFrame,
) -> str:
    if not matched_activities.is_empty() and "start" in matched_activities.columns:
        dates = (
            matched_activities.select(pl.col("start").dt.date().alias("date"))
            .get_column("date")
            .drop_nulls()
        )
        if not dates.is_empty():
            return str(dates.mode().sort()[0])

    if not user_keystrokes.is_empty() and "timestamp" in user_keystrokes.columns:
        dates = (
            user_keystrokes.select(pl.col("timestamp").dt.date().alias("date"))
            .get_column("date")
            .drop_nulls()
        )
        if not dates.is_empty():
            return str(dates.mode().sort()[0])

    raise ValueError("Cannot derive date: no activities or keystrokes")


def _assemble_workblocks(df: pl.DataFrame, user_id: str) -> pl.DataFrame:
    if df.is_empty():
        return _empty_workblocks()

    assembled = df.with_columns(
        [
            pl.lit(user_id).alias("user_id"),
            _date_from_start_time("start_time").alias("date"),
        ]
    )
    return _select_and_cast(assembled, WORKBLOCK_SCHEMA)


def _assemble_slices(df: pl.DataFrame, user_id: str) -> pl.DataFrame:
    if df.is_empty():
        return _empty_slices()

    assembled = df.with_columns(
        [
            pl.lit(user_id).alias("user_id"),
            _date_from_start_time("start_time").alias("date"),
            pl.col("raw_keystrokes")
            .map_elements(_serialize_raw_keystrokes, return_dtype=pl.Utf8)
            .alias("raw_keystrokes"),
            pl.col("is_synthetic")
            .fill_null(False)
            .alias("is_synthetic")
            if "is_synthetic" in df.columns
            else pl.lit(False).alias("is_synthetic"),
            pl.col("is_orphan")
            .fill_null(False)
            .alias("is_orphan")
            if "is_orphan" in df.columns
            else pl.lit(False).alias("is_orphan"),
            pl.col("synthetic_reason")
            .fill_null("")
            .alias("synthetic_reason")
            if "synthetic_reason" in df.columns
            else pl.lit("").alias("synthetic_reason"),
            pl.col("cross_app_keystrokes")
            .fill_null(0)
            .alias("cross_app_keystrokes")
            if "cross_app_keystrokes" in df.columns
            else pl.lit(0).alias("cross_app_keystrokes"),
        ]
    )
    return _select_and_cast(assembled, SLICE_SCHEMA)


def _date_from_start_time(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Utf8).str.slice(0, 10)


def _coerce_datetime_column(column_name: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column_name).str.to_datetime(time_zone="UTC", strict=False),
            pl.col(column_name).str.to_datetime(
                format="%+",
                time_zone="UTC",
                strict=False,
            ),
        ]
    ).alias(column_name)


def _serialize_raw_keystrokes(value: object) -> str:
    if value is None:
        return json.dumps([])
    if hasattr(value, "to_list"):
        value = value.to_list()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return json.dumps([])
        return json.dumps(parsed)
    if isinstance(value, list):
        return json.dumps(value)
    return json.dumps([])


def _select_and_cast(
    df: pl.DataFrame,
    schema: dict[str, pl.DataType],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column_name, data_type in schema.items():
        if column_name in df.columns:
            expressions.append(pl.col(column_name).cast(data_type).alias(column_name))
        else:
            expressions.append(_default_expr(column_name, data_type))
    return df.select(expressions)


def _default_expr(column_name: str, data_type: pl.DataType) -> pl.Expr:
    defaults: dict[str, object] = {
        "workblock_id": "",
        "user_id": "",
        "date": "",
        "start_time": "",
        "end_time": "",
        "duration_minutes": 0.0,
        "active_duration_minutes": 0.0,
        "total_activities": 0,
        "sequence": 0,
        "application": "",
        "exec": "",
        "url": "",
        "window_title": "",
        "keystroke_count": 0,
        "raw_keystrokes": json.dumps([]),
        "activity_count": 0,
        "is_multitasking": False,
        "concurrent_activities": 0,
        "concurrent_apps": [],
        "is_synthetic": False,
        "is_orphan": False,
        "synthetic_reason": "",
        "cross_app_keystrokes": 0,
    }
    return pl.lit(defaults[column_name], dtype=data_type).alias(column_name)


def _empty_workblocks() -> pl.DataFrame:
    return pl.DataFrame(schema=WORKBLOCK_SCHEMA)


def _empty_slices() -> pl.DataFrame:
    return pl.DataFrame(schema=SLICE_SCHEMA)
