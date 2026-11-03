"""
WorkblockBuilder - Simplified timeline generation based on temporal proximity.

This module creates workblocks (groups of activities within 1 hour of each other)
and atomic slices (continuous same-app/URL activities within workblocks).
No pattern detection or behavioral analysis - just temporal grouping.
"""

import hashlib
import json
import polars as pl
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Set, cast
from collections import defaultdict
import logging
import time
from workblock_engine.utils.time_utils import make_tz_aware
from workblock_engine.utils.id_utils import (
    generate_workblock_id,
)

logger = logging.getLogger(__name__)

class WorkblockBuilder:
    """
    Builds workblocks from user activities based on temporal proximity.

    Workblocks are groups of activities that occur within 1 hour of each other.
    Within workblocks, activities are further grouped into entries based on
    same app/URL continuity (also within 1 hour gaps).
    """

    def __init__(self, cross_day_gap_minutes: int = 180):
        """
        Initialize the workblock builder.

        Args:
            cross_day_gap_minutes: Idle-gap threshold (minutes) for forcing a
                split when activity crosses midnight. If the gap between the
                last pre-midnight activity and the next post-midnight activity
                is less than or equal to this value, a new workblock is
                started even when the 60-minute intra-day rule would keep them
                together. Lower values relax the split (e.g., ``0`` keeps any
                positive gap in one block). Defaults to 180 minutes (three
                hours) to align with PRD guidance.
        """
        self.cross_day_gap_minutes = cross_day_gap_minutes

    def _extract_url_from_title(self, title: str) -> str:
        """
        Extract URL from window titles like 'example.com - Google Chrome'.
        Returns empty string if no URL found.
        """
        if not title:
            return ""

        browser_suffixes = [
            " - Google Chrome",
            " - Mozilla Firefox",
            " - Microsoft Edge",
            " - Safari",
            " - Opera",
            " - Brave",
        ]

        clean_title = title
        for suffix in browser_suffixes:
            if title.endswith(suffix):
                clean_title = title[: -len(suffix)].strip()
                break

        if (
            clean_title.startswith(("/", "\\"))
            or (
                len(clean_title) > 2
                and clean_title[1] == ":"
                and clean_title[0].isalpha()
            )
            or "\\" in clean_title
        ):
            return ""

        if (" " not in clean_title and ("." in clean_title or clean_title.startswith("localhost"))):
            if not clean_title.startswith(("http://", "https://")):
                return f"https://{clean_title}"
            return clean_title

        return ""

    def _split_activities_by_utc_day(self, activities: pl.DataFrame) -> pl.DataFrame:
        """Split any activities that cross UTC midnight into per-day fragments."""
        if activities.is_empty():
            return activities

        def _parse_keystrokes(raw_value: Any) -> List[Dict[str, Any]]:
            if not raw_value:
                return []
            if isinstance(raw_value, str):
                try:
                    parsed = json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    return []
                if not isinstance(parsed, list):
                    return []
                return [dict(item) for item in parsed if isinstance(item, dict)]
            if isinstance(raw_value, list):
                return [dict(item) for item in raw_value if isinstance(item, dict)]
            return []

        def _render_keystrokes(values: List[Dict[str, Any]], original: Any) -> Any:
            if isinstance(original, str):
                return json.dumps(values)
            if isinstance(original, list):
                return values
            if original is None and not values:
                return None
            return json.dumps(values)

        def _expand_events(
            start_dt: datetime,
            end_dt: datetime,
            items: List[Dict[str, Any]],
        ) -> List[tuple[datetime, Dict[str, Any]]]:
            events: List[tuple[datetime, Dict[str, Any]]] = []
            if not items:
                return events
            tzinfo = make_tz_aware(start_dt).tzinfo
            start_dt = make_tz_aware(start_dt)
            end_dt = make_tz_aware(end_dt)
            base_date = start_dt.date()
            start_seconds = (
                start_dt.hour * 3600 + start_dt.minute * 60 + start_dt.second
            )
            crosses_midnight = end_dt.date() != start_dt.date()
            for item in items:
                time_str = item.get("time")
                if not time_str:
                    continue
                parts = time_str.split(":")
                if len(parts) < 2:
                    continue
                try:
                    hour = int(parts[0])
                    minute = int(parts[1])
                    second = int(parts[2]) if len(parts) > 2 else 0
                except ValueError:
                    continue
                total_seconds = hour * 3600 + minute * 60 + second
                event_date = base_date
                if crosses_midnight and total_seconds < start_seconds:
                    event_date = event_date + timedelta(days=1)
                event_dt = datetime(
                    event_date.year,
                    event_date.month,
                    event_date.day,
                    hour,
                    minute,
                    second,
                    tzinfo=tzinfo,
                )
                events.append((event_dt, dict(item)))
            events.sort(key=lambda entry: entry[0])
            return events

        def _start_of_next_day(dt: datetime) -> datetime:
            next_day = make_tz_aware(dt).date() + timedelta(days=1)
            next_start = datetime.combine(next_day, datetime.min.time())
            tzinfo = make_tz_aware(dt).tzinfo
            return next_start.replace(tzinfo=tzinfo)

        schema = activities.schema
        split_rows: List[Dict[str, Any]] = []
        activities_split = 0
        total_fragments = 0
        user_identifier: Optional[str] = None
        if "user_id" in activities.columns and len(activities) > 0:
            try:
                user_identifier = str(activities["user_id"][0])  # type: ignore[index]
            except Exception:
                user_identifier = None

        for row in activities.iter_rows(named=True):
            start = row.get("start")
            end = row.get("end")
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                split_rows.append(dict(row))
                continue

            start_dt = make_tz_aware(start)
            end_dt = make_tz_aware(end)
            if start_dt >= end_dt or start_dt.date() == end_dt.date():
                split_rows.append(dict(row))
                continue

            original_keystrokes = row.get("keystrokes")
            keystroke_events = _expand_events(
                start_dt,
                end_dt,
                _parse_keystrokes(original_keystrokes),
            )
            total_events = len(keystroke_events)
            event_index = 0
            fragment_start = start_dt

            fragments_for_row = 0

            while fragment_start < end_dt:
                fragment_end = min(_start_of_next_day(fragment_start), end_dt)
                if fragment_end <= fragment_start:
                    break

                fragment_keystrokes: List[Dict[str, Any]] = []
                while event_index < total_events:
                    event_dt, payload = keystroke_events[event_index]
                    if event_dt < fragment_start:
                        event_index += 1
                        continue
                    if event_dt < fragment_end or (
                        fragment_end == end_dt and event_dt <= fragment_end
                    ):
                        fragment_keystrokes.append(payload)
                        event_index += 1
                    else:
                        break

                fragment_row = dict(row)
                fragment_row["start"] = fragment_start
                fragment_row["end"] = fragment_end
                fragment_row["duration_minutes"] = round(
                    (fragment_end - fragment_start).total_seconds() / 60,
                    4,
                )
                if "keystrokes" in fragment_row:
                    fragment_row["keystrokes"] = _render_keystrokes(
                        fragment_keystrokes,
                        original_keystrokes,
                    )
                if "keystroke_count" in fragment_row:
                    fragment_row["keystroke_count"] = len(fragment_keystrokes)
                split_rows.append(fragment_row)
                fragments_for_row += 1
                fragment_start = fragment_end

            if event_index < total_events and split_rows:
                remaining = [payload for _, payload in keystroke_events[event_index:]]
                target_row = split_rows[-1]
                existing_value = (
                    target_row.get("keystrokes") if "keystrokes" in target_row else None
                )
                combined = _parse_keystrokes(existing_value)
                combined.extend(remaining)
                if "keystrokes" in target_row:
                    target_row["keystrokes"] = _render_keystrokes(
                        combined,
                        original_keystrokes,
                    )
                if "keystroke_count" in target_row:
                    target_row["keystroke_count"] = len(combined)

            if fragments_for_row > 1:
                activities_split += 1
                total_fragments += fragments_for_row

        try:
            result_df = pl.DataFrame(split_rows, schema=schema)
        except Exception:
            result_df = pl.DataFrame(split_rows)

        if activities_split > 0:
            user_detail = f" user_id={user_identifier}" if user_identifier else ""
            logger.info(
                "Split %s activities into %s fragments across UTC day boundaries%s",
                activities_split,
                total_fragments,
                user_detail,
            )

        return result_df

    def build_workblocks(self, activities: pl.DataFrame, date_str: str) -> Dict[str, Any]:
        """
        Build workblocks from activities.

        Args:
            activities: DataFrame with cleaned and enriched activities
                       Must have: user_id, timestamp/start, end, duration_minutes,
                       name (app), url, title, keystrokes
            date_str: Date string for the data (YYYY-MM-DD)

        Returns:
            Dictionary containing:
            - 'workblocks': List of workblock dictionaries
            - 'atomic_slices': List of atomic slice dictionaries
            - 'metadata': Processing metadata
        """
        build_start_time = time.time()
        if activities.is_empty():
            return {
                "workblocks": [],
                "atomic_slices": [],
                "metadata": {"date": date_str, "total_activities": 0},
            }

        if "start" not in activities.columns and "timestamp" in activities.columns:
            activities = activities.with_columns(pl.col("timestamp").alias("start"))

        if "end" not in activities.columns:
            activities = activities.with_columns(
                (
                    pl.col("start") + pl.duration(minutes=pl.col("duration_minutes"))
                ).alias("end")
            )

        activities = activities.sort([
            col for col in ["start", "end", "id", "name", "exec", "title", "url", "task_name"]
            if col in activities.columns
        ])
        activities = self._split_activities_by_utc_day(activities)
        activities = activities.sort([
            col for col in ["start", "end", "id", "name", "exec", "title", "url", "task_name"]
            if col in activities.columns
        ])

        user_id = activities["user_id"][0] if "user_id" in activities.columns else "unknown"

        workblocks = []
        atomic_slices = []
        current_workblock = None
        current_workblock_date = None
        workblock_sequence = 0

        entry_counter = 1
        assigned_keystrokes: Set[str] = set()

        keystrokes_by_minute = defaultdict(list)
        context_switches_by_minute = defaultdict(int)

        if len(activities) > 0:
            activities_with_span = activities.with_columns(
                [
                    pl.col("start").dt.truncate("1m").alias("start_minute"),
                    pl.col("end").dt.truncate("1m").alias("end_minute"),
                ]
            ).with_columns(
                [
                    (
                        (pl.col("end_minute") - pl.col("start_minute")).dt.total_seconds()
                        / 60
                    )
                    .cast(pl.Int32)
                    .alias("minute_span")
                ]
            )

            max_span_raw = activities_with_span["minute_span"].max()
            max_span_value = cast(Optional[int], max_span_raw)
            max_span = int(max_span_value) if max_span_value is not None else 0

            minute_offsets = pl.DataFrame({"minute_offset": range(max_span + 1)})

            expanded = activities_with_span.join(minute_offsets, how="cross")
            expanded = expanded.filter(pl.col("minute_offset") <= pl.col("minute_span"))
            expanded = expanded.with_columns(
                [
                    (
                        pl.col("start_minute")
                        + pl.duration(minutes=pl.col("minute_offset"))
                    ).alias("minute_timestamp")
                ]
            )
            expanded = expanded.filter(pl.col("minute_timestamp") <= pl.col("end"))

            context_switches_df = expanded.group_by("minute_timestamp").agg(
                pl.len().alias("context_switches")
            ).with_columns(
                [
                    pl.col("minute_timestamp")
                    .dt.strftime("%Y-%m-%d %H:%M")
                    .alias("minute_key")
                ]
            )

            for row in context_switches_df.iter_rows(named=True):
                context_switches_by_minute[row["minute_key"]] = row["context_switches"]

        for idx, row in enumerate(activities.iter_rows(named=True)):
            if "keystrokes" in row and row["keystrokes"]:
                keystrokes = row["keystrokes"]
                if isinstance(keystrokes, str):
                    try:
                        keystrokes = json.loads(keystrokes)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(
                            f"Warning: Failed to parse keystrokes JSON for activity {idx}: {e}"
                        )
                        keystrokes = []
                if isinstance(keystrokes, list):
                    for ks in keystrokes:
                        ks_time_str = ks.get("time", "")
                        if ks_time_str:
                            try:
                                ks_parts = ks_time_str.split(":")
                                if len(ks_parts) >= 2:
                                    ks_hour = int(ks_parts[0])
                                    ks_min = int(ks_parts[1])

                                    if not (0 <= ks_hour <= 23):
                                        print(
                                            f"Warning: Invalid hour {ks_hour} in keystroke time {ks_time_str}"
                                        )
                                        continue
                                    if not (0 <= ks_min <= 59):
                                        print(
                                            f"Warning: Invalid minute {ks_min} in keystroke time {ks_time_str}"
                                        )
                                        continue

                                    start_value = row.get("start")
                                    if isinstance(start_value, datetime):
                                        activity_date = start_value.date()
                                    elif isinstance(start_value, date):
                                        activity_date = start_value
                                    else:
                                        print(
                                            f"Warning: Unexpected start type {type(row['start'])} for activity {idx}"
                                        )
                                        continue

                                    # Parse seconds if available, default to 0
                                    ks_sec = int(ks_parts[2]) if len(ks_parts) >= 3 else 0
                                    if not (0 <= ks_sec <= 59):
                                        ks_sec = 0

                                    ks_datetime = datetime(
                                        activity_date.year,
                                        activity_date.month,
                                        activity_date.day,
                                        ks_hour,
                                        ks_min,
                                        ks_sec,
                                    )
                                    minute_key = ks_datetime.strftime("%Y-%m-%d %H:%M")

                                    is_ambiguous = (
                                        context_switches_by_minute.get(minute_key, 0) > 1
                                    )

                                    ks_with_ambiguity = ks.copy()
                                    ks_with_ambiguity["ambiguous"] = is_ambiguous
                                    ks_with_ambiguity["_datetime"] = make_tz_aware(ks_datetime)
                                    ks_with_ambiguity["_source_app"] = row.get("name", "")
                                    ks_with_ambiguity["_source_exec"] = row.get("exec", "")
                                    keystrokes_by_minute[minute_key].append(
                                        ks_with_ambiguity
                                    )
                            except (ValueError, AttributeError) as e:
                                print(
                                    f"Warning: Failed to parse keystroke time '{ks_time_str}' for activity {idx}: {e}"
                                )

        activities_list = list(activities.iter_rows(named=True))
        current_workblock_activities = []

        for row in activities_list:
            activity_start = row["start"]
            activity_end = row["end"]
            activity_date = make_tz_aware(activity_start).date()

            last_end_time = (
                current_workblock["end_time"] if current_workblock is not None else None
            )
            needs_new_workblock = self._should_start_new_workblock(
                last_end_time,
                activity_start,
            )

            if (
                not needs_new_workblock
                and current_workblock is not None
                and current_workblock_date is not None
                and activity_date != current_workblock_date
            ):
                gap_minutes = (
                    make_tz_aware(activity_start)
                    - make_tz_aware(current_workblock["end_time"])
                ).total_seconds() / 60
                if gap_minutes <= self.cross_day_gap_minutes:
                    needs_new_workblock = True

            if needs_new_workblock:
                if current_workblock is not None and current_workblock_activities:
                    max_end_time = max(act["end"] for act in current_workblock_activities)
                    current_workblock["end_time"] = max_end_time

                    segments = self._create_time_segments(
                        current_workblock_activities,
                        keystrokes_by_minute,
                    )
                    (
                        workblock_slices,
                        entry_counter,
                        assigned_keystrokes,
                    ) = self._create_atomic_slices_from_segments(
                        segments,
                        current_workblock,
                        entry_counter,
                        current_workblock["workblock_id"],
                        assigned_keystrokes,
                    )
                    atomic_slices.extend(workblock_slices)
                    workblocks.append(self._finalize_workblock(current_workblock))

                normalized_start = make_tz_aware(activity_start)
                workblock_id = generate_workblock_id(
                    user_id,
                    normalized_start,
                    sequence=workblock_sequence,
                )
                current_workblock = self._create_new_workblock(
                    workblock_id=workblock_id,
                    user_id=user_id,
                    start_time=activity_start,
                )
                workblock_sequence += 1
                current_workblock_date = activity_date
                entry_counter = 1
                current_workblock_activities = []

            if current_workblock is None:
                continue
            current_workblock["end_time"] = activity_end
            current_workblock["total_activities"] += 1
            current_workblock_activities.append(row)

        if current_workblock is not None and current_workblock_activities:
            max_end_time = max(act["end"] for act in current_workblock_activities)
            current_workblock["end_time"] = max_end_time

            segments = self._create_time_segments(
                current_workblock_activities,
                keystrokes_by_minute,
            )
            (
                workblock_slices,
                entry_counter,
                assigned_keystrokes,
            ) = self._create_atomic_slices_from_segments(
                segments,
                current_workblock,
                entry_counter,
                current_workblock["workblock_id"],
                assigned_keystrokes,
            )
            atomic_slices.extend(workblock_slices)
            workblocks.append(self._finalize_workblock(current_workblock))

        min_duration_minutes = 5.0
        filtered_workblocks = []
        filtered_entries = []

        for wb in workblocks:
            if wb["duration_minutes"] >= min_duration_minutes:
                filtered_workblocks.append(wb)
                for entry in atomic_slices:
                    if entry["workblock_id"] == wb["workblock_id"]:
                        filtered_entries.append(entry)

        renumbered_workblocks = []
        renumbered_entries = []

        for wb in filtered_workblocks:
            old_wb_id = wb["workblock_id"]
            renumbered_workblocks.append(wb)

            for entry in filtered_entries:
                if entry["workblock_id"] == old_wb_id:
                    renumbered_entries.append(entry)

        filtered_workblocks = renumbered_workblocks
        filtered_entries = renumbered_entries

        consolidated_entries = self._consolidate_entries(filtered_entries)
        entries_by_workblock = defaultdict(list)
        entries_by_date = defaultdict(list)

        for entry in consolidated_entries:
            workblock_id = entry.get("workblock_id")
            if workblock_id is not None:
                entries_by_workblock[workblock_id].append(entry)
            start_time = entry.get("start_time")
            if start_time:
                entry_date = start_time.split("T")[0]
                entries_by_date[entry_date].append(entry)

        for wb in filtered_workblocks:
            wb_entries = entries_by_workblock.get(wb["workblock_id"], [])
            active_duration = sum(entry.get("duration_minutes", 0) for entry in wb_entries)

            if active_duration > wb["duration_minutes"]:
                print(
                    f"Warning: Active duration ({active_duration}) exceeds total duration ({wb['duration_minutes']}) for workblock {wb['workblock_id']}. Capping at total duration."
                )
                active_duration = wb["duration_minutes"]

            wb["active_duration_minutes"] = round(active_duration, 1)
            wb["date"] = wb["start_time"].split("T")[0]

        build_time = time.time() - build_start_time
        if build_time > 0.1:
            print(f"        • workblock building took: {build_time:.3f}s")

        return {
            "workblocks": filtered_workblocks,
            "atomic_slices": consolidated_entries,
            "metadata": {
                "date": date_str,
                "primary_date": date_str,
                "dates": sorted(entries_by_date.keys()),
                "total_activities": len(activities),
                "total_workblocks": len(filtered_workblocks),
                "total_entries": len(consolidated_entries),
                "user_id": user_id,
            },
        }

    def _create_time_segments(
        self,
        activities_list: List[Dict],
        keystrokes_by_minute: Dict,
    ) -> List[Dict]:
        """
        Create non-overlapping time segments from all activity boundaries.
        This prevents double-counting when activities overlap.

        Args:
            activities_list: List of activity dictionaries with start/end times
            keystrokes_by_minute: Dictionary of keystrokes indexed by minute

        Returns:
            List of time segment dictionaries with attributed activities
        """
        if not activities_list:
            return []

        time_boundaries = set()
        for activity in activities_list:
            start_minute = activity["start"].replace(second=0, microsecond=0)
            end_minute = activity["end"].replace(second=0, microsecond=0)

            current = start_minute
            time_boundaries.add(activity["start"])

            while current < end_minute:
                next_minute = current + timedelta(minutes=1)
                time_boundaries.add(next_minute)
                current = next_minute

            time_boundaries.add(activity["end"])

        sorted_times = sorted(time_boundaries)
        time_segments = []

        for i in range(len(sorted_times) - 1):
            segment_start = sorted_times[i]
            segment_end = sorted_times[i + 1]

            if segment_start >= segment_end:
                continue

            segment = {
                "start": segment_start,
                "end": segment_end,
                "duration_minutes": (
                    segment_end - segment_start
                ).total_seconds()
                / 60,
                "activities": [],
                "keystrokes": [],
            }

            for activity in activities_list:
                if activity["start"] <= segment_start and activity["end"] >= segment_end:
                    segment["activities"].append(activity)

            # Add keystrokes whose timestamp falls within this segment
            current_minute = segment_start.replace(second=0, microsecond=0)
            end_minute = segment_end.replace(second=0, microsecond=0)
            segment_keystrokes = []
            while current_minute <= end_minute:
                mk = current_minute.strftime("%Y-%m-%d %H:%M")
                for ks in keystrokes_by_minute.get(mk, []):
                    ks_dt = ks.get("_datetime")
                    if ks_dt is None or (
                        make_tz_aware(segment_start) <= ks_dt < make_tz_aware(segment_end)
                    ):
                        segment_keystrokes.append(ks)
                current_minute += timedelta(minutes=1)
            segment["keystrokes"] = segment_keystrokes

            time_segments.append(segment)

        # Preserve zero-length synthetic/orphan activities as point segments so
        # they can still produce atomic slices with provenance.
        for activity in activities_list:
            if activity["start"] != activity["end"]:
                continue

            segment_minute = activity["start"].replace(second=0, microsecond=0)
            minute_key = segment_minute.strftime("%Y-%m-%d %H:%M")
            time_segments.append(
                {
                    "start": activity["start"],
                    "end": activity["end"],
                    "duration_minutes": 0.0,
                    "activities": [activity],
                    "keystrokes": keystrokes_by_minute.get(minute_key, []),
                }
            )

        return sorted(
            time_segments,
            key=lambda segment: (segment["start"], segment["end"]),
        )

    def _create_atomic_slices_from_segments(
        self,
        segments: List[Dict],
        current_workblock: Dict,
        entry_counter: int,
        workblock_id: str,
        assigned_keystrokes: set,
    ) -> tuple[List[Dict], int, set]:
        """
        Create atomic slices from time segments, ensuring no double-counting.

        Returns:
            Tuple of (atomic_slices, updated_entry_counter, updated_assigned_keystrokes).
        """
        atomic_slices = []

        for segment in segments:
            if not segment["activities"]:
                continue

            # Pick primary activity using keystroke app affinity when available
            primary_activity = self._select_primary_activity(
                segment["activities"], segment["keystrokes"]
            )

            segment_activity = primary_activity.copy()
            segment_activity["start"] = segment["start"]
            segment_activity["end"] = segment["end"]
            segment_activity["duration_minutes"] = segment["duration_minutes"]

            current_entry = self._create_new_entry(
                workblock_id=current_workblock["workblock_id"],
                sequence=entry_counter,
                activity=segment_activity,
            )
            entry_counter += 1

            current_entry["concurrent_activities"] = len(segment["activities"])
            current_entry["is_multitasking"] = len(segment["activities"]) > 1
            if len(segment["activities"]) > 1:
                current_entry["concurrent_apps"] = [
                    act.get("name", act.get("app_filename", ""))
                    for act in segment["activities"]
                ]
            else:
                current_entry["concurrent_apps"] = []

            minute_key = segment["start"].replace(second=0, microsecond=0).strftime(
                "%Y-%m-%d %H:%M"
            )

            for ks in segment["keystrokes"]:
                ks_key = self._keystroke_signature(ks, minute_key)

                if ks_key not in assigned_keystrokes:
                    # Strip internal metadata before storing in output
                    clean_ks = {k: v for k, v in ks.items() if not k.startswith("_")}
                    current_entry["raw_keystrokes"].append(clean_ks)
                    current_entry["keystroke_count"] += 1
                    assigned_keystrokes.add(ks_key)

            atomic_slices.append(self._finalize_entry(current_entry))

        return atomic_slices, entry_counter, assigned_keystrokes

    def _select_primary_activity(
        self, activities: List[Dict], keystrokes: List[Dict]
    ) -> Dict:
        """Select primary activity based on keystroke app affinity.

        When a segment has multiple overlapping activities and keystrokes,
        pick the activity whose app matches the majority of keystrokes'
        source app. Falls back to first activity when no keystrokes or
        no match.
        """
        if len(activities) <= 1 or not keystrokes:
            return activities[0]

        app_votes: Dict[str, int] = {}
        for ks in keystrokes:
            source = ks.get("_source_app", "")
            if source:
                app_votes[source] = app_votes.get(source, 0) + 1

        if not app_votes:
            return activities[0]

        best_app = max(app_votes, key=lambda k: app_votes[k])
        for act in activities:
            if act.get("name", "") == best_app:
                return act

        return activities[0]

    def _keystroke_signature(self, keystroke: Any, minute_key: str) -> str:
        """Create a stable, compact signature for keystroke deduplication."""
        if isinstance(keystroke, dict):
            ks_id = keystroke.get("id")
            if ks_id:
                return f"{minute_key}:{ks_id}"
            if "text" in keystroke and keystroke["text"] is not None:
                payload = str(keystroke["text"])
            else:
                payload = json.dumps(keystroke, sort_keys=True, ensure_ascii=False)
        else:
            payload = str(keystroke)

        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
        return f"{minute_key}:{digest}"

    def _should_start_new_workblock(
        self,
        last_end_time: Optional[datetime],
        current_start_time: datetime,
    ) -> bool:
        """Check if we should start a new workblock based on time gap."""
        if last_end_time is None:
            return True

        gap_minutes = (
            make_tz_aware(current_start_time) - make_tz_aware(last_end_time)
        ).total_seconds() / 60
        return gap_minutes > 60

    def _create_new_workblock(
        self,
        workblock_id: str,
        user_id: str,
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Create a new workblock structure."""
        return {
            "workblock_id": workblock_id,
            "user_id": user_id,
            "start_time": start_time,
            "end_time": start_time,
            "duration_minutes": 0,
            "active_duration_minutes": 0,
            "total_activities": 0,
        }

    def _create_new_entry(self, workblock_id: str, sequence: int, activity: Dict) -> Dict[str, Any]:
        """Create a new atomic slice structure."""
        url = activity.get("url", "")

        return {
            "workblock_id": workblock_id,
            "sequence": sequence,
            "start_time": activity["start"],
            "end_time": activity["end"],
            "duration_minutes": activity["duration_minutes"],
            "application": activity.get("app_filename") or activity.get("name") or "unknown",
            "exec": activity.get("exec", ""),
            "url": url,
            "window_title": activity.get("title", ""),
            "activity_count": 1,
            "keystroke_count": 0,
            "raw_keystrokes": [],
            "is_synthetic": activity.get("is_synthetic", False),
            "is_orphan": activity.get("is_orphan", False),
            "synthetic_reason": activity.get("synthetic_reason", ""),
            "cross_app_keystrokes": activity.get("cross_app_keystrokes", 0),
        }

    def _finalize_workblock(self, workblock: Dict) -> Dict[str, Any]:
        """Finalize a workblock by calculating duration."""
        duration = (
            make_tz_aware(workblock["end_time"]) - make_tz_aware(workblock["start_time"])
        ).total_seconds() / 60
        workblock["duration_minutes"] = round(duration, 1)

        workblock["start_time"] = workblock["start_time"].isoformat()
        workblock["end_time"] = workblock["end_time"].isoformat()

        return workblock

    def _finalize_entry(self, entry: Dict) -> Dict[str, Any]:
        """Finalize an atomic slice by calculating duration."""
        duration = (
            make_tz_aware(entry["end_time"]) - make_tz_aware(entry["start_time"])
        ).total_seconds() / 60
        entry["duration_minutes"] = max(0.1, round(duration, 1))

        entry["start_time"] = entry["start_time"].isoformat()
        entry["end_time"] = entry["end_time"].isoformat()

        for field in ["application", "exec", "url", "window_title"]:
            if entry.get(field) is None:
                entry[field] = ""

        if "concurrent_apps" not in entry:
            entry["concurrent_apps"] = []

        keystroke_url = None
        if entry.get("raw_keystrokes"):
            for ks in entry["raw_keystrokes"]:
                if isinstance(ks, dict) and ks.get("url"):
                    keystroke_url = ks["url"]
                    if not entry.get("window_title") or entry["window_title"] == "":
                        entry["window_title"] = keystroke_url
                    break

        if keystroke_url:
            entry["url"] = keystroke_url
        elif (not entry.get("url") or entry.get("url") == "") and entry.get("window_title"):
            extracted_url = self._extract_url_from_title(entry["window_title"])
            if extracted_url:
                entry["url"] = extracted_url

        if "is_multitasking" not in entry:
            entry["is_multitasking"] = False
        if "concurrent_activities" not in entry:
            entry["concurrent_activities"] = 1
        if "concurrent_apps" not in entry:
            entry["concurrent_apps"] = []

        return entry

    def _consolidate_entries(self, entries: List[Dict]) -> List[Dict]:
        """Consolidate consecutive entries with same app/URL and no keystrokes."""
        if not entries:
            return entries

        sorted_entries = sorted(
            entries,
            key=lambda x: (x["workblock_id"], x["start_time"]),
        )

        consolidated = []
        current_group = None

        for entry in sorted_entries:
            should_consolidate = False

            if current_group and entry["workblock_id"] == current_group["workblock_id"]:
                if (
                    entry["application"] == current_group["application"]
                    and entry.get("url", "") == current_group.get("url", "")
                    and len(entry.get("raw_keystrokes", [])) == 0
                    and len(current_group.get("raw_keystrokes", [])) == 0
                    and not entry.get("is_multitasking", False)
                    and not current_group.get("is_multitasking", False)
                    and entry.get("is_synthetic", False)
                    == current_group.get("is_synthetic", False)
                    and entry.get("is_orphan", False)
                    == current_group.get("is_orphan", False)
                    and entry.get("synthetic_reason", "")
                    == current_group.get("synthetic_reason", "")
                    and entry.get("cross_app_keystrokes", 0)
                    == current_group.get("cross_app_keystrokes", 0)
                ):
                    group_end = make_tz_aware(
                        datetime.fromisoformat(current_group["end_time"])
                    )
                    entry_start = make_tz_aware(datetime.fromisoformat(entry["start_time"]))

                    group_end_ms = int(group_end.timestamp() * 1000)
                    entry_start_ms = int(entry_start.timestamp() * 1000)
                    gap_ms = entry_start_ms - group_end_ms

                    if gap_ms <= 1000:
                        should_consolidate = True

            if should_consolidate:
                if current_group is None:
                    continue
                current_group["end_time"] = entry["end_time"]
                current_group["duration_minutes"] = round(
                    (
                        make_tz_aware(datetime.fromisoformat(current_group["end_time"]))
                        - make_tz_aware(datetime.fromisoformat(current_group["start_time"]))
                    ).total_seconds()
                    / 60,
                    1,
                )
                current_group["activity_count"] += entry.get("activity_count", 1)

                if "concurrent_apps" not in current_group:
                    current_group["concurrent_apps"] = entry.get("concurrent_apps", [])
            else:
                if current_group:
                    consolidated.append(current_group)
                current_group = entry.copy()
                if "concurrent_apps" not in current_group:
                    current_group["concurrent_apps"] = []

        if current_group:
            consolidated.append(current_group)

        wb_counters = {}
        for entry in consolidated:
            wb_id = entry["workblock_id"]
            if wb_id not in wb_counters:
                wb_counters[wb_id] = 1
            entry["sequence"] = wb_counters[wb_id]
            wb_counters[wb_id] += 1

        return consolidated
