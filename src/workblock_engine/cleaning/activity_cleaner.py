"""
Activity data cleaning and token optimization for LLM processing.
Focuses solely on data cleaning and transformations.
Workblock creation is handled by WorkblockBuilder.
"""
import polars as pl


class ActivityCleaner:
    """Clean and optimize activity data for downstream processing."""

    def __init__(self):
        """
        Initialize the activity cleaner.
        """
        self.required_columns = ["user_id", "timestamp", "name", "title", "url", "task_name"]

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean activity data while preserving critical information.
        Reduces tokens for LLM processing.
        """
        # Handle empty dataframe
        if len(df) == 0 or df.is_empty():
            return pl.DataFrame({
                col: [] for col in self.required_columns + ["start", "end", "exec", "app_exe", "app_filename", "duration_minutes", "task_id"]
            })

        # Check for required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check if we need exec column
        if "exec" not in df.columns:
            df = df.with_columns(pl.col("name").alias("exec"))
        else:
            # Ensure exec column is of string type even with nulls
            df = df.with_columns(pl.col("exec").cast(pl.Utf8).alias("exec"))

        # Convert start/end to datetime if they're strings and add UTC timezone
        if "start" in df.columns and df.schema["start"] == pl.Utf8:
            df = df.with_columns(
                pl.col("start").str.to_datetime()
                .dt.replace_time_zone("UTC")
            )
        elif "start" not in df.columns:
            df = df.with_columns(
                pl.col("timestamp")
                .alias("start")
            )

        if "end" in df.columns and df.schema["end"] == pl.Utf8:
            df = df.with_columns(
                pl.col("end").str.to_datetime()
                .dt.replace_time_zone("UTC")
            )
        elif "end" not in df.columns:
            df = df.with_columns(
                (pl.col("timestamp") + pl.duration(minutes=10))
                .alias("end")
            )

        # Process the dataframe
        df = (
            df
            .with_columns([
                # Extract executable with parent directory: C:\Program Files\Microsoft\app.exe → Microsoft\app.exe
                pl.when(pl.col("exec").is_null())
                .then(pl.col("name"))  # Use name when exec is null
                .when(pl.col("exec").str.contains("\\\\"))  # Windows path
                .then(pl.col("exec").str.split("\\").list.slice(-2).list.join("\\"))
                .when(pl.col("exec").str.contains("/"))  # Unix path
                .then(pl.col("exec").str.split("/").list.slice(-2).list.join("/"))
                .otherwise(pl.col("exec"))  # No path separator, use as is
                .alias("app_exe"),

                # Extract just the executable filename: C:\...\chrome.exe → chrome.exe
                # Normalize to ensure consistency
                pl.when(pl.col("exec").is_null())
                .then(
                    # Normalize name when exec is null: ensure .exe extension and consistent case
                    pl.when(pl.col("name").str.to_lowercase().str.ends_with(".exe"))
                    .then(
                        # Already has .exe, just fix first letter capitalization
                        pl.col("name").str.slice(0, 1).str.to_uppercase() +
                        pl.col("name").str.slice(1).str.to_lowercase()
                    )
                    .otherwise(
                        # Add .exe with proper capitalization
                        pl.col("name").str.slice(0, 1).str.to_uppercase() +
                        pl.col("name").str.slice(1).str.to_lowercase() + ".exe"
                    )
                )
                .when(pl.col("exec").str.contains("\\\\"))  # Windows path
                .then(pl.col("exec").str.split("\\").list.last())
                .when(pl.col("exec").str.contains("/"))  # Unix path
                .then(pl.col("exec").str.split("/").list.last())
                .otherwise(pl.col("exec"))  # No path separator, use as is
                .alias("app_filename"),

                # Calculate duration in minutes
                pl.when(pl.col("end").is_not_null() & pl.col("start").is_not_null())
                .then((pl.col("end") - pl.col("start")).dt.total_minutes())
                .otherwise(pl.lit(0.0))
                .alias("duration_minutes"),

                # Ensure timestamp is datetime with UTC timezone
                pl.col("timestamp").str.to_datetime().dt.replace_time_zone("UTC") if df.schema["timestamp"] == pl.Utf8 else pl.col("timestamp").dt.replace_time_zone("UTC"),

                # Clean task names (remove extra whitespace)
                pl.col("task_name").str.strip_chars().alias("task_name")
            ])
            .drop([col for col in df.columns if col not in self.required_columns + ["app_exe", "app_filename", "duration_minutes", "task_id", "start", "end", "id", "exec"]])
        )

        # Ensure start and end columns have UTC timezone
        df = df.with_columns([
            pl.col("start").dt.replace_time_zone("UTC") if "start" in df.columns else pl.col("start"),
            pl.col("end").dt.replace_time_zone("UTC") if "end" in df.columns else pl.col("end")
        ])

        # Select columns in order, including ID if it exists
        select_cols = []
        if "id" in df.columns:
            select_cols.append("id")
        select_cols.extend([
            "user_id", "timestamp",
            "start",     # Preserve start time for keystroke joining
            "end",       # Preserve end time for keystroke joining
            "name",      # Full app name preserved
            "title",     # Full title preserved
            "url",       # Full URL preserved
            "task_name", # Full task name preserved
            "exec",      # Preserve exec path in cleaned output
            "app_exe", "app_filename", "duration_minutes"
        ])

        df = (
            df
            .select(select_cols + [
                pl.col("task_id").fill_null("unknown") if "task_id" in df.columns else pl.lit("unknown").alias("task_id")
            ])
            .sort("user_id", "timestamp")
        )

        # Deduplicate by ID if ID column exists
        if "id" in df.columns:
            original_count = len(df)
            df = df.unique(subset=["id"])
            deduped_count = len(df)
            if original_count > deduped_count:
                print(f"    Removed {original_count - deduped_count} duplicate activities (by ID)")

        return df
