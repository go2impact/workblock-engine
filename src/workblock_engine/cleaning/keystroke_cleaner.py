"""
Keystroke data cleaning and transformations.
Prepares keystroke data for workblock creation.

Keystroke compression is applied to reduce token count by replacing consecutive repeated keystrokes
with a count notation. For example:
- <Cmd><Cmd><Cmd> becomes <Cmd>{3}
- <Backspace><Backspace><Backspace> becomes <Backspace>{3}
- <Ctrl><Ctrl + a><Ctrl + c><Ctrl><Ctrl + a><Ctrl + c> becomes (<Ctrl><Ctrl + a><Ctrl + c>){2}
- <Cmd><Cmd + c><Cmd><Cmd + v> stays the same (no repeated pattern)

This compression only applies to patterns inside angle brackets (<>).
"""
import polars as pl

from workblock_engine.utils.keystroke_compression import compress_keystrokes


class KeystrokeCleaner:
    """Clean and prepare keystroke data for downstream processing."""

    def __init__(self):
        """
        Initialize the keystroke cleaner.
        """
        self.required_columns = ["user_id", "timestamp", "app", "keystrokes", "window_title"]
        # Note: 'id' is generated if not present, so it's not required

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean keystroke data.

        This method:
        1. Validates required columns
        2. Normalizes column names
        3. Adds missing columns
        4. Sorts data by user and timestamp

        Args:
            df: Raw keystroke DataFrame

        Returns:
            Cleaned DataFrame ready for pattern extraction
        """
        # Check if dataframe is empty
        if len(df) == 0 or df.is_empty():
            return pl.DataFrame({
                col: [] for col in self.required_columns + ["keystroke_count", "duration_seconds"]
            })

        # Normalize column names
        df = self._normalize_columns(df)

        # Check for required columns after normalization
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Basic cleaning
        df = (
            df
            .with_columns([
                # Ensure timestamp is proper type with UTC timezone
                pl.col("timestamp").str.to_datetime().dt.replace_time_zone("UTC") if df.schema["timestamp"] == pl.Utf8 else pl.col("timestamp").dt.replace_time_zone("UTC"),

                # Clean app names (remove paths, keep just app name)
                pl.col("app")
                .str.split("/")
                .list.last()
                .str.split("\\")
                .list.last()
                .alias("app_name"),

                # Calculate keystroke metrics (before compression for accurate count)
                pl.col("keystrokes").str.len_chars().alias("keystroke_count"),

                # Clean window titles
                pl.col("window_title").str.strip_chars().alias("window_title")
            ])
            .sort([
                col for col in [
                    "user_id", "timestamp", "id", "app", "exec", "keystrokes",
                    "window_title", "url",
                ] if col in df.columns
            ])
        )

        # Apply keystroke compression
        df = df.with_columns(
            pl.col("keystrokes").map_elements(
                self._compress_keystrokes,
                return_dtype=pl.Utf8
            ).alias("keystrokes")
        )

        # No ID generation - preserve existing IDs from source data

        # Add derived columns
        df = self._add_derived_columns(df)

        # Deduplicate by ID if ID column exists
        if "id" in df.columns:
            original_count = len(df)
            df = df.unique(subset=["id"], keep="first", maintain_order=True)
            deduped_count = len(df)
            if original_count > deduped_count:
                print(f"    Removed {original_count - deduped_count} duplicate keystrokes (by ID)")

        # Select final columns in order
        final_columns = []
        if "id" in df.columns:
            final_columns.append("id")
        final_columns.extend(["user_id", "timestamp", "app", "keystrokes", "window_title",
                             "keystroke_count", "duration_seconds"])
        # Add app_name if it exists
        if "app_name" in df.columns:
            app_name_pos = 4 if "id" not in df.columns else 5
            final_columns.insert(app_name_pos, "app_name")
        # Add exec if it exists (needed for 4-phase matching)
        if "exec" in df.columns:
            exec_pos = 4 if "id" not in df.columns else 5
            final_columns.insert(exec_pos, "exec")
        # Add url if it exists (needed for synthetic activities)
        if "url" in df.columns:
            url_pos = 6 if "id" not in df.columns else 7
            final_columns.insert(url_pos, "url")

        return df.select(final_columns)

    def _normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize column names and add missing columns.

        Handles variations in column naming from different data sources:
        - timestamp → datetime
        - exec → app (but preserve exec)
        - keystroke → keystrokes
        - etc.
        """
        # Column name mappings
        column_mappings = {
            # Remove timestamp->datetime mapping to keep consistency
            'application': 'app',
            'keystroke': 'keystrokes',
            'keys': 'keystrokes',
            'title': 'window_title'
        }

        # Special handling for exec column - preserve it if it exists
        if 'exec' in df.columns and 'app' not in df.columns:
            # Create app column from exec but keep exec
            df = df.with_columns(pl.col('exec').alias('app'))

        # Rename columns if they exist
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        # Add window_title if missing (derive from URL or use app name)
        if "window_title" not in df.columns:
            if "url" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("url").is_not_null() & (pl.col("url") != ""))
                    .then(pl.col("url").str.extract(r"https?://([^/]+)", group_index=1))
                    .when(pl.col("app").is_not_null())
                    .then(pl.col("app").str.split("/").list.last())
                    .otherwise(pl.lit("Unknown"))
                    .alias("window_title")
                )
            else:
                df = df.with_columns(
                    pl.when(pl.col("app").is_not_null())
                    .then(pl.col("app").str.split("/").list.last())
                    .otherwise(pl.lit("Unknown"))
                    .alias("window_title")
                )

        # Ensure keystrokes column exists
        if "keystrokes" not in df.columns:
            raise ValueError("No keystroke data found in DataFrame")

        return df

    def _add_derived_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add useful derived columns."""
        return df.with_columns([
            # Estimate duration based on keystroke activity
            # Assume average typing speed of 40 WPM = ~200 CPM = ~3.3 CPS
            (pl.col("keystroke_count") / 3.3).alias("duration_seconds"),
        ])

    def _compress_keystrokes(self, keystrokes: str) -> str:
        """Compress repeated keystroke tokens and multi-token patterns."""
        return compress_keystrokes(keystrokes)
