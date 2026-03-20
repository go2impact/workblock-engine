"""Functions for matching keystrokes to activities."""
import polars as pl
from datetime import datetime
import json
import hashlib
import logging

from workblock_engine.utils.time_utils import make_tz_aware
from workblock_engine.utils.app_utils import normalize_app_name
from workblock_engine.matching.synthetic import (
    phase3_create_gap_activities,
    phase4_create_orphan_activities
)

logger = logging.getLogger(__name__)


def _generate_keystroke_id(ks: dict) -> int:
    """Generate deterministic ID for a keystroke based on its content."""
    ks_content = f"{ks.get('timestamp', '')}:{ks.get('app', '')}:{ks.get('exec', '')}:{ks.get('keystrokes', '')}:{ks.get('text', '')}"
    return int(hashlib.md5(ks_content.encode()).hexdigest()[:8], 16)


def _normalize_app_column(col: pl.Expr) -> pl.Expr:
    """Apply app normalization to a column."""
    return col.map_elements(
        lambda x: normalize_app_name(x) if x else "unknown",
        return_dtype=pl.Utf8
    )


def join_keystrokes_to_activities(activities_df: pl.DataFrame, keystrokes_df: pl.DataFrame, user_id: str) -> pl.DataFrame:
    """
    Enhanced keystroke matching with app-aware, multi-phase algorithm.

    Phases:
    1. Exact match (temporal + app)
    2. Nearest neighbor with app preference
    3. Gap filling (synthetic activities)
    4. Orphan activities (100% match rate)

    Args:
        activities_df: DataFrame with cleaned activities (must have start/end columns)
        keystrokes_df: DataFrame with cleaned keystrokes
        user_id: User ID for filtering

    Returns:
        Activities DataFrame with keystrokes column added, including synthetic activities
    """
    if keystrokes_df.is_empty():
        return activities_df.with_columns([
            pl.lit("[]").alias("keystrokes"),
            pl.lit(0).alias("keystroke_count"),
            pl.lit(0).alias("cross_app_keystrokes"),
            pl.lit(False).alias("is_synthetic"),
            pl.lit("").alias("synthetic_reason")
        ])

    # Add normalized app names to both datasets
    activities_df = activities_df.with_columns([
        _normalize_app_column(pl.col("name")).alias("app_normalized")
    ])

    # For activities, also try to get app from exec if available
    if "exec" in activities_df.columns:
        activities_df = activities_df.with_columns([
            pl.when(pl.col("app_normalized") == "unknown")
            .then(_normalize_app_column(pl.col("exec")))
            .otherwise(pl.col("app_normalized"))
            .alias("app_normalized")
        ])

    # Add normalized app names to keystrokes
    if "exec" in keystrokes_df.columns:
        keystrokes_df = keystrokes_df.with_columns([
            _normalize_app_column(pl.col("exec")).alias("app_normalized")
        ])

        # If keystrokes have app column, use it as fallback
        if "app" in keystrokes_df.columns:
            keystrokes_df = keystrokes_df.with_columns([
                pl.when(pl.col("app_normalized") == "unknown")
                .then(_normalize_app_column(pl.col("app")))
                .otherwise(pl.col("app_normalized"))
                .alias("app_normalized")
            ])
    elif "app" in keystrokes_df.columns:
        # No exec column, use app column directly
        keystrokes_df = keystrokes_df.with_columns([
            _normalize_app_column(pl.col("app")).alias("app_normalized")
        ])
    else:
        # Neither exec nor app column exists, create a default
        keystrokes_df = keystrokes_df.with_columns([
            pl.lit("unknown").alias("app_normalized")
        ])

    # Filter keystrokes for this user
    # Cast user_id to string for comparison to handle mixed types
    user_keystrokes_df = keystrokes_df.filter(pl.col("user_id").cast(pl.Utf8) == str(user_id))

    # Add activity index for tracking assignments
    activities_df = activities_df.with_row_index("activity_idx")

    # Add keystroke ID if not present
    if "id" not in user_keystrokes_df.columns:
        # Ensure all required columns exist with defaults for ID generation
        if "text" not in user_keystrokes_df.columns:
            user_keystrokes_df = user_keystrokes_df.with_columns(pl.lit("").alias("text"))
        if "app" not in user_keystrokes_df.columns:
            user_keystrokes_df = user_keystrokes_df.with_columns(pl.lit("").alias("app"))
        if "exec" not in user_keystrokes_df.columns:
            user_keystrokes_df = user_keystrokes_df.with_columns(pl.lit("").alias("exec"))

        # Generate IDs using Polars operations for better memory efficiency
        user_keystrokes_df = user_keystrokes_df.with_columns(
            pl.struct(["timestamp", "app", "exec", "keystrokes", "text"]).map_elements(
                lambda row: int(hashlib.md5(
                    f"{str(row['timestamp']) if row['timestamp'] else ''}:"
                    f"{row['app'] or ''}:"
                    f"{row['exec'] or ''}:"
                    f"{row['keystrokes'] or ''}:"
                    f"{row['text'] or ''}"
                    .encode()
                ).hexdigest()[:8], 16),
                return_dtype=pl.Int64
            ).alias("id")
        )

    # Track statistics
    total_keystrokes = len(user_keystrokes_df)
    stats = {
        "total_keystrokes": total_keystrokes,
        "phase1_matches": 0,
        "phase2_matches": 0,
        "phase3_matches": 0,
        "phase4_matches": 0,
        "cross_app_matches": 0,
        "synthetic_activities": 0
    }

    # Phase 1: Direct matching with app validation using Polars operations
    logger.debug(f"🔄 Phase 1: Direct temporal + app matching for user {user_id}...")

    # Perform temporal + app join
    phase1_matches = (
        user_keystrokes_df.join(
            activities_df,
            how="cross"
        )
        .filter(
            # Temporal overlap
            (pl.col("timestamp") >= pl.col("start")) &
            (pl.col("timestamp") <= pl.col("end")) &
            # App match
            (pl.col("app_normalized") == pl.col("app_normalized_right"))
        )
        # Take first match per keystroke (maintains original behavior)
        .group_by("id")
        .first()
    )

    # Extract matched keystroke IDs
    if not phase1_matches.is_empty():
        matched_keystroke_ids = set(phase1_matches["id"].to_list())
        stats["phase1_matches"] = len(matched_keystroke_ids)
    else:
        matched_keystroke_ids = set()

    # Convert to format needed for later processing
    # We'll need both the activities data and keystrokes data as dicts for synthetic activity creation
    activities_data = activities_df.drop("activity_idx").to_dicts()
    keystrokes_data = user_keystrokes_df.to_dicts()

    # Build keystroke assignments from phase 1 matches
    keystroke_assignments = {}
    if not phase1_matches.is_empty():
        for row in phase1_matches.iter_rows(named=True):
            activity_idx = int(row["activity_idx"])
            # Find the original keystroke from keystrokes_data by ID
            ks_id = row["id"]
            for ks in keystrokes_data:
                if ks.get("id", _generate_keystroke_id(ks)) == ks_id:
                    if activity_idx not in keystroke_assignments:
                        keystroke_assignments[activity_idx] = []
                    keystroke_assignments[activity_idx].append(ks)
                    break

    # Phase 2: Nearest neighbor with app preference
    logger.debug(f"🔄 Phase 2: Nearest neighbor matching...")
    # Phase 2: Identify unmatched keystrokes using Polars filter
    unmatched_df = user_keystrokes_df.filter(~pl.col("id").is_in(list(matched_keystroke_ids)))
    unmatched = unmatched_df.to_dicts()

    if not unmatched_df.is_empty():
        # Perform cross join between unmatched keystrokes and activities
        phase2_candidates = (
            unmatched_df.join(
                activities_df,
                how="cross"
            )
            # Calculate time distances vectorized
            .with_columns([
                # Convert to milliseconds for integer comparison
                (pl.col("timestamp").dt.epoch("ms")).alias("ks_ms"),
                (pl.col("start").dt.epoch("ms")).alias("start_ms"),
                (pl.col("end").dt.epoch("ms")).alias("end_ms"),
            ])
            # Calculate time distance
            .with_columns([
                pl.when(pl.col("ks_ms") < pl.col("start_ms"))
                .then((pl.col("start_ms") - pl.col("ks_ms")) / 1000.0)  # Before activity
                .when(pl.col("ks_ms") > pl.col("end_ms"))
                .then((pl.col("ks_ms") - pl.col("end_ms")) / 1000.0)  # After activity
                .otherwise(None)  # Should have been caught in phase 1
                .alias("time_dist")
            ])
            # Filter out overlaps (should have been caught in phase 1) and too far (>5 min)
            .filter(
                pl.col("time_dist").is_not_null() &
                (pl.col("time_dist") <= 300)
            )
            # Apply app penalty and calculate score
            .with_columns([
                (pl.col("app_normalized") != pl.col("app_normalized_right")).alias("is_cross_app"),
                (
                    pl.col("time_dist") +
                    pl.when(pl.col("app_normalized") != pl.col("app_normalized_right"))
                    .then(180)  # 3-minute penalty for cross-app
                    .otherwise(0)
                ).alias("score")
            ])
            # Get best match per keystroke
            .sort(["id", "score"])
            .group_by("id")
            .first()
        )

        # Process Phase 2 matches
        if not phase2_candidates.is_empty():
            for row in phase2_candidates.iter_rows(named=True):
                activity_idx = int(row["activity_idx"])
                ks_id = row["id"]

                # Find the original keystroke data
                for ks in unmatched:
                    if ks.get("id", _generate_keystroke_id(ks)) == ks_id:
                        if activity_idx not in keystroke_assignments:
                            keystroke_assignments[activity_idx] = []
                        keystroke_assignments[activity_idx].append(ks)
                        matched_keystroke_ids.add(ks_id)
                        stats["phase2_matches"] += 1
                        if row["is_cross_app"]:
                            stats["cross_app_matches"] += 1
                        break

    # Phase 3: Create synthetic activities for gaps
    logger.debug(f"🔄 Phase 3: Creating synthetic activities for gaps...")
    # Phase 3: Identify still unmatched keystrokes using Polars filter
    still_unmatched_df = user_keystrokes_df.filter(~pl.col("id").is_in(list(matched_keystroke_ids)))
    still_unmatched = still_unmatched_df.to_dicts()

    if still_unmatched and activities_data:
        gap_activities = phase3_create_gap_activities(activities_data, still_unmatched)

        for gap_activity in gap_activities:
            idx = len(activities_data)

            # Add to activities data
            activities_data.append({
                "user_id": user_id,
                "start": gap_activity["start"],
                "end": gap_activity["end"],
                "name": gap_activity["name"],
                "title": gap_activity.get("title", ""),  # Preserve title (often URL)
                "url": gap_activity.get("url", ""),     # Preserve URL field
                "exec": gap_activity.get("exec", ""),
                "app_filename": gap_activity.get("app_filename", ""),  # Add app_filename
                "app_normalized": gap_activity["app_normalized"],
                "duration_minutes": gap_activity["duration_minutes"],
                "is_synthetic": True,
                "synthetic_reason": gap_activity["synthetic_reason"]
            })

            # Assign keystrokes from the _keystrokes field
            if "_keystrokes" in gap_activity:
                keystroke_assignments[idx] = gap_activity["_keystrokes"]
                for ks in gap_activity["_keystrokes"]:
                    matched_keystroke_ids.add(ks.get("id", _generate_keystroke_id(ks)))
                stats["phase3_matches"] += len(gap_activity["_keystrokes"])
            stats["synthetic_activities"] += 1

    # Phase 4: Create synthetic activities for remaining orphans
    logger.debug(f"🔄 Phase 4: Creating synthetic activities for orphans...")
    # Phase 4: Identify final orphan keystrokes using Polars filter
    final_orphans_df = user_keystrokes_df.filter(~pl.col("id").is_in(list(matched_keystroke_ids)))
    final_orphans = final_orphans_df.to_dicts()

    if final_orphans:
        orphan_activities = phase4_create_orphan_activities(final_orphans)

        for orphan_activity in orphan_activities:
            idx = len(activities_data)

            # Add to activities data
            activities_data.append({
                "user_id": user_id,
                "start": orphan_activity["start"],
                "end": orphan_activity["end"],
                "name": orphan_activity["name"],
                "title": orphan_activity.get("title", ""),  # Preserve title (often URL)
                "url": orphan_activity.get("url", ""),     # Preserve URL field
                "exec": orphan_activity.get("exec", ""),
                "app_filename": orphan_activity.get("app_filename", ""),  # Add app_filename
                "app_normalized": orphan_activity["app_normalized"],
                "duration_minutes": orphan_activity["duration_minutes"],
                "is_synthetic": True,
                "is_orphan": True,
                "synthetic_reason": orphan_activity["synthetic_reason"]
            })

            # Assign keystrokes from the _keystrokes field
            if "_keystrokes" in orphan_activity:
                keystroke_assignments[idx] = orphan_activity["_keystrokes"]
                for ks in orphan_activity["_keystrokes"]:
                    matched_keystroke_ids.add(ks.get("id", _generate_keystroke_id(ks)))
                stats["phase4_matches"] += len(orphan_activity["_keystrokes"])
            stats["synthetic_activities"] += 1

    # Build final result
    enriched_rows = []

    for idx, activity in enumerate(activities_data):
        enriched_activity = activity.copy()

        # Add synthetic flags if not present
        if "is_synthetic" not in enriched_activity:
            enriched_activity["is_synthetic"] = False
        if "synthetic_reason" not in enriched_activity:
            enriched_activity["synthetic_reason"] = ""

        if idx in keystroke_assignments:
            assigned_keystrokes = keystroke_assignments[idx]

            # Extract keystroke field name
            keystroke_field = "keystrokes" if "keystrokes" in keystrokes_df.columns else "keys"

            # Format keystrokes
            raw_keystrokes = []
            cross_app_count = 0
            activity_app = activity.get("app_normalized", "unknown")

            for ks in assigned_keystrokes:
                if ks.get(keystroke_field):
                    ks_app = ks.get("app_normalized", "unknown")
                    raw_keystrokes.append({
                        "id": ks.get("id"),
                        "time": ks["timestamp"].strftime("%H:%M:%S"),
                        "text": ks[keystroke_field],
                        "app": ks_app,
                        "url": ks.get("url", "")  # Include URL field
                    })

                    # Count cross-app matches
                    if ks_app != activity_app and not activity.get("is_synthetic"):
                        cross_app_count += 1

            # Sort by time
            raw_keystrokes.sort(key=lambda x: x["time"])

            enriched_activity["keystrokes"] = json.dumps(raw_keystrokes)
            enriched_activity["keystroke_count"] = len(raw_keystrokes)
            enriched_activity["cross_app_keystrokes"] = cross_app_count
        else:
            enriched_activity["keystrokes"] = json.dumps([])
            enriched_activity["keystroke_count"] = 0
            enriched_activity["cross_app_keystrokes"] = 0

        enriched_rows.append(enriched_activity)

    # Ensure all datetime fields are timezone-aware before creating DataFrame
    for row in enriched_rows:
        for field in ['start', 'end', 'timestamp']:
            if field in row and isinstance(row[field], datetime):
                row[field] = make_tz_aware(row[field])

    # Log comprehensive statistics at DEBUG level
    if total_keystrokes > 0:
        total_matched = sum(stats[f"phase{i}_matches"] for i in range(1, 5))

        logger.debug(f"📊 Keystroke Matching Statistics for user {user_id}:")
        logger.debug(f"  Total keystrokes: {total_keystrokes}")
        logger.debug(f"  Phase 1 (exact match): {stats['phase1_matches']} ({stats['phase1_matches']/total_keystrokes*100:.1f}%)")
        logger.debug(f"  Phase 2 (nearest): {stats['phase2_matches']} ({stats['phase2_matches']/total_keystrokes*100:.1f}%)")
        logger.debug(f"  Phase 3 (gap synthetic): {stats['phase3_matches']} ({stats['phase3_matches']/total_keystrokes*100:.1f}%)")
        logger.debug(f"  Phase 4 (orphan synthetic): {stats['phase4_matches']} ({stats['phase4_matches']/total_keystrokes*100:.1f}%)")
        logger.debug(f"  Total matched: {total_matched} ({total_matched/total_keystrokes*100:.1f}%)")

        logger.debug(f"  Quality metrics:")
        logger.debug(f"    - Cross-app matches: {stats['cross_app_matches']}")
        logger.debug(f"    - Synthetic activities created: {stats['synthetic_activities']}")

        # Sanity check
        if total_matched != total_keystrokes:
            logger.warning(f"⚠️  WARNING: {total_keystrokes - total_matched} keystrokes still unmatched for user {user_id}!")

    return pl.DataFrame(enriched_rows)
