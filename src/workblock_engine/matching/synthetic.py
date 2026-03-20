"""Functions for creating synthetic activities from unmatched keystrokes."""
from typing import List, Dict
from decimal import Decimal, ROUND_HALF_UP

from workblock_engine.utils.time_utils import make_tz_aware
from workblock_engine.utils.app_utils import normalize_app_name


def normalize_timestamp(dt):
    """Remove microseconds from datetime for consistent precision."""
    if dt:
        return dt.replace(microsecond=0)
    return dt


def _calculate_duration_minutes(start_time, end_time) -> float:
    """
    Calculate duration in minutes with consistent precision.
    Same implementation as in WorkblockBuilder for consistency.
    """
    # Normalize timestamps to remove microseconds
    start_time = normalize_timestamp(start_time)
    end_time = normalize_timestamp(end_time)
    
    # Calculate total seconds with proper rounding
    total_seconds = round((end_time - start_time).total_seconds())
    
    # Convert to minutes using Decimal for consistent precision
    minutes = Decimal(str(total_seconds)) / Decimal('60')
    
    # Return as float with consistent precision
    return float(minutes)


def phase3_create_gap_activities(activities_data: List[Dict], unmatched_keystrokes: List[Dict]) -> List[Dict]:
    """
    Create synthetic activities for gaps between real activities where keystrokes exist.
    
    Uses optimized Polars implementation for larger datasets (50+ activities).
    
    Args:
        activities_data: List of activity dictionaries (sorted by time)
        unmatched_keystrokes: List of keystroke dictionaries not yet matched
        
    Returns:
        List of synthetic activity dictionaries
    """
    if not activities_data or not unmatched_keystrokes:
        return []
    
    # Use original implementation for small datasets where overhead isn't worth it
    if len(activities_data) < 50:
        return _phase3_create_gap_activities_original(activities_data, unmatched_keystrokes)
    
    # Use optimized implementation for larger datasets
    return _phase3_create_gap_activities_optimized(activities_data, unmatched_keystrokes)


def _phase3_create_gap_activities_original(activities_data: List[Dict], unmatched_keystrokes: List[Dict]) -> List[Dict]:
    """Original implementation using Python loops."""
    synthetic_activities = []
    used_keystroke_ids = set()  # Track which keystrokes we've already used
    
    # Find gaps between consecutive activities
    for i in range(len(activities_data) - 1):
        gap_start = make_tz_aware(activities_data[i]["end"])
        gap_end = make_tz_aware(activities_data[i + 1]["start"])
        gap_duration = (gap_end - gap_start).total_seconds()
        
        # Only consider significant gaps (> 5 minutes)
        if gap_duration > 300:
            # Find keystrokes in this gap that haven't been used yet
            gap_keystrokes = []
            for ks in unmatched_keystrokes:
                ks_id = ks.get("id", id(ks))
                if (ks_id not in used_keystroke_ids and 
                    gap_start <= make_tz_aware(ks["timestamp"]) <= gap_end):
                    gap_keystrokes.append(ks)
                    used_keystroke_ids.add(ks_id)
            
            if gap_keystrokes:
                # Group keystrokes by app within the gap
                app_groups = {}
                for ks in gap_keystrokes:
                    app = normalize_app_name(ks.get("exec", ""))
                    if app not in app_groups:
                        app_groups[app] = []
                    app_groups[app].append(ks)
                
                # Create synthetic activity for each app group
                for app, keystrokes in app_groups.items():
                    # Find time bounds for this app's keystrokes
                    ks_times = [make_tz_aware(ks["timestamp"]) for ks in keystrokes]
                    activity_start = min(ks_times)
                    activity_end = max(ks_times)
                    
                    # Extract filename from exec path for app_filename
                    exec_path = keystrokes[0].get("exec", "")
                    app_filename = exec_path.split("\\")[-1] if "\\" in exec_path else exec_path.split("/")[-1] if "/" in exec_path else exec_path
                    
                    # Get URL and title from keystrokes
                    url = ""
                    title = ""
                    for ks in keystrokes:
                        if ks.get("url"):
                            url = ks["url"]
                            title = url
                            break
                    
                    # If no URL found, try to get title from window_title
                    if not title:
                        for ks in keystrokes:
                            if ks.get("window_title"):
                                title = ks["window_title"]
                                break
                    
                    synthetic_activities.append({
                        "id": f"gap_{activity_start.isoformat()}_{app}",
                        "start": activity_start,
                        "end": activity_end,
                        "name": f"Gap Activity - {app.title()}",
                        "title": title,
                        "url": url,
                        "exec": exec_path,
                        "app_filename": app_filename,  # Add app_filename field
                        "app_normalized": app,
                        "duration_minutes": _calculate_duration_minutes(activity_start, activity_end),
                        "is_synthetic": True,
                        "synthetic_reason": "activity_gap",
                        "_keystrokes": keystrokes  # Prefix with _ to differentiate
                    })
    
    return synthetic_activities


def _phase3_create_gap_activities_optimized(activities_data: List[Dict], unmatched_keystrokes: List[Dict]) -> List[Dict]:
    """Optimized implementation using Polars for better performance on large datasets."""
    import polars as pl
    
    # Convert to DataFrames
    activities_df = pl.DataFrame(activities_data)
    keystrokes_df = pl.DataFrame(unmatched_keystrokes)
    
    # Ensure datetime columns are timezone-aware
    keystrokes_df = keystrokes_df.with_columns([
        pl.col("timestamp").map_elements(make_tz_aware, return_dtype=pl.Datetime(time_zone="UTC")).alias("ks_datetime")
    ])
    
    # Pre-process activities to get gaps
    activities_sorted = activities_df.sort("start")
    
    # Create gaps DataFrame
    gaps_data = []
    activities_rows = activities_sorted.rows(named=True)
    for i in range(len(activities_rows) - 1):
        gap_start = make_tz_aware(activities_rows[i]["end"])
        gap_end = make_tz_aware(activities_rows[i + 1]["start"])
        gap_duration = (gap_end - gap_start).total_seconds()
        
        if gap_duration > 300:  # Only significant gaps
            gaps_data.append({
                "gap_id": i,
                "gap_start": gap_start,
                "gap_end": gap_end
            })
    
    if not gaps_data:
        return []
    
    gaps_df = pl.DataFrame(gaps_data)
    
    # Use cross join and filter instead of nested loops
    result_df = gaps_df.join(keystrokes_df, how="cross")
    
    # Filter keystrokes that fall within gaps
    result_df = result_df.filter(
        (pl.col("ks_datetime") >= pl.col("gap_start")) &
        (pl.col("ks_datetime") <= pl.col("gap_end"))
    )
    
    if result_df.is_empty():
        return []
    
    # Extract app name from exec path
    result_df = result_df.with_columns([
        pl.col("exec").str.split("\\\\").list.last().str.split("/").list.last().str.replace(r"\.exe$", "").alias("app_normalized")
    ])
    
    # Build aggregation expressions based on available columns
    agg_exprs = [
        pl.col("ks_datetime").min().alias("activity_start"),
        pl.col("ks_datetime").max().alias("activity_end"),
        pl.col("exec").first(),
    ]
    
    # Add conditional aggregations for optional fields
    if "url" in result_df.columns:
        agg_exprs.append(pl.col("url").filter(pl.col("url") != "").first().alias("first_url"))
    else:
        agg_exprs.append(pl.lit("").alias("first_url"))
    
    if "window_title" in result_df.columns:
        agg_exprs.append(pl.col("window_title").filter(pl.col("window_title") != "").first().alias("first_title"))
    else:
        agg_exprs.append(pl.lit("").alias("first_title"))
    
    # Build struct for keystrokes data
    struct_fields = ["id", "timestamp", "exec", "app"]
    for field in ["url", "window_title", "keystrokes", "text"]:
        if field in result_df.columns:
            struct_fields.append(field)
    
    agg_exprs.append(pl.struct(struct_fields).alias("keystrokes_data"))
    
    # Group by gap and app to create synthetic activities
    grouped = result_df.group_by(["gap_id", "app_normalized"]).agg(agg_exprs)
    
    # Convert to synthetic activities
    synthetic_activities = []
    for row in grouped.iter_rows(named=True):
        app = row["app_normalized"]
        exec_path = row["exec"]
        app_filename = exec_path.split("\\")[-1] if "\\" in exec_path else exec_path.split("/")[-1] if "/" in exec_path else exec_path
        
        # Use URL as title if available, otherwise window title
        url = row["first_url"] or ""
        title = url if url else (row["first_title"] or "")
        
        # Use original times for start/end (with microseconds)
        start_time = row["activity_start"]
        end_time = row["activity_end"]
        
        # Calculate duration using the same method as original
        # Normalize timestamps to remove microseconds for duration calc only
        start_normalized = start_time.replace(microsecond=0) if start_time else start_time
        end_normalized = end_time.replace(microsecond=0) if end_time else end_time
        # Calculate total seconds with proper rounding
        total_seconds = round((end_normalized - start_normalized).total_seconds())
        # Convert to minutes using Decimal for consistent precision
        duration_minutes = float(Decimal(str(total_seconds)) / Decimal('60'))
        
        synthetic_activities.append({
            "id": f"gap_{start_time.isoformat()}_{app}",
            "start": start_time,  # Keep original with microseconds
            "end": end_time,      # Keep original with microseconds
            "name": f"Gap Activity - {app.title()}",
            "title": title,
            "url": url,
            "exec": exec_path,
            "app_filename": app_filename,
            "app_normalized": app,
            "duration_minutes": duration_minutes,
            "is_synthetic": True,
            "synthetic_reason": "activity_gap",
            "_keystrokes": row["keystrokes_data"]
        })
    
    return synthetic_activities


def phase4_create_orphan_activities(orphaned_keystrokes: List[Dict]) -> List[Dict]:
    """
    Create synthetic activities for truly orphaned keystrokes.
    Groups them by app and temporal proximity to ensure 100% match rate.
    
    Args:
        orphaned_keystrokes: List of keystroke dictionaries not matched in any phase
        
    Returns:
        List of synthetic activity dictionaries
    """
    if not orphaned_keystrokes:
        return []
    
    # Sort by app and time
    sorted_ks = sorted(
        orphaned_keystrokes,
        key=lambda x: (normalize_app_name(x.get("exec", "")), make_tz_aware(x["timestamp"]))
    )
    
    synthetic_activities = []
    current_activity = None
    
    for ks in sorted_ks:
        ks_app = normalize_app_name(ks.get("exec", ""))
        ks_time = make_tz_aware(ks["timestamp"])
        
        if not current_activity:
            # Start new synthetic activity
            exec_path = ks.get("exec", "")
            app_filename = exec_path.split("\\")[-1] if "\\" in exec_path else exec_path.split("/")[-1] if "/" in exec_path else exec_path
            
            # Extract URL and title
            url = ks.get("url", "")
            # Use URL as title if available, otherwise use window_title
            title = url if url else ks.get("window_title", ks.get("title", ""))
            
            current_activity = {
                "id": f"orphan_{ks_time.isoformat()}_{ks_app}",
                "start": ks_time,
                "end": ks_time,
                "name": f"Orphan Activity - {ks_app.title()}",
                "title": title,
                "url": url,
                "exec": exec_path,
                "app_filename": app_filename,
                "app_normalized": ks_app,
                "is_synthetic": True,
                "is_orphan": True,
                "synthetic_reason": "orphaned_keystrokes",
                "_keystrokes": [ks]
            }
        else:
            # Check if should continue current activity
            same_app = ks_app == current_activity["app_normalized"]
            time_gap = (ks_time - make_tz_aware(current_activity["end"])).total_seconds()
            
            if same_app and time_gap <= 300:  # 5 minutes
                # Extend current activity
                current_activity["end"] = ks_time
                current_activity["_keystrokes"].append(ks)
                # Update title if current keystroke has a URL and activity doesn't
                if ks.get("url") and not current_activity.get("title"):
                    current_activity["title"] = ks["url"]
                    current_activity["url"] = ks["url"]
            else:
                # Save current and start new
                current_activity["duration_minutes"] = _calculate_duration_minutes(
                    make_tz_aware(current_activity["start"]),
                    make_tz_aware(current_activity["end"])
                )
                synthetic_activities.append(current_activity)
                
                exec_path = ks.get("exec", "")
                app_filename = exec_path.split("\\")[-1] if "\\" in exec_path else exec_path.split("/")[-1] if "/" in exec_path else exec_path
                
                # Extract URL and title
                url = ks.get("url", "")
                # Use URL as title if available, otherwise use window_title
                title = url if url else ks.get("window_title", ks.get("title", ""))
                
                current_activity = {
                    "id": f"orphan_{ks_time.isoformat()}_{ks_app}",
                    "start": ks_time,
                    "end": ks_time,
                    "name": f"Orphan Activity - {ks_app.title()}",
                    "title": title,  # Use extracted title
                    "url": ks.get("url", ""),  # Preserve URL if available
                    "exec": exec_path,
                    "app_filename": app_filename,
                    "app_normalized": ks_app,
                    "is_synthetic": True,
                    "is_orphan": True,
                    "synthetic_reason": "orphaned_keystrokes",
                    "_keystrokes": [ks]
                }
    
    # Don't forget the last activity
    if current_activity:
        current_activity["duration_minutes"] = _calculate_duration_minutes(
            make_tz_aware(current_activity["start"]),
            make_tz_aware(current_activity["end"])
        )
        synthetic_activities.append(current_activity)
    
    return synthetic_activities
