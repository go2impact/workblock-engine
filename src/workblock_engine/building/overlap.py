"""Activity overlap resolution functions."""
import polars as pl
import logging

logger = logging.getLogger(__name__)


def resolve_activity_overlaps(activities_df: pl.DataFrame) -> pl.DataFrame:
    """
    Resolve overlapping activities by trimming end times.
    
    When activities overlap, we trim the end time of the earlier activity
    to match the start time of the next activity. This ensures no overlaps
    while preserving all activities.
    
    Args:
        activities_df: DataFrame with activities containing 'start' and 'end' columns
        
    Returns:
        DataFrame with overlaps resolved
    """
    if activities_df.is_empty():
        return activities_df
        
    # Sort by start time to process in chronological order
    activities_sorted = activities_df.sort("start")
    
    # Get the next activity's start time using shift
    # shift(-1) moves values up by 1 row (next row becomes current row)
    next_start = pl.col("start").shift(-1)
    
    # Identify overlaps: where current end > next start
    has_overlap = pl.col("end") > next_start
    
    # Create adjusted end time: use next start if overlap, else keep original end
    adjusted_end = pl.when(has_overlap).then(next_start).otherwise(pl.col("end"))
    
    # Apply the adjustment
    resolved_df = activities_sorted.with_columns(
        adjusted_end.alias("end")
    )
    
    # Count overlaps for logging (before adjustment)
    overlap_count = activities_sorted.filter(has_overlap).height
    
    # Recalculate duration if it exists
    if "duration_minutes" in resolved_df.columns:
        resolved_df = resolved_df.with_columns(
            (pl.col("end") - pl.col("start")).dt.total_minutes().alias("duration_minutes")
        )
    
    if overlap_count > 0:
        logger.debug(f"✅ Resolved {overlap_count} overlapping activities")
    
    return resolved_df