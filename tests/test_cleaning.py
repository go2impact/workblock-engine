"""
Comprehensive tests for ActivityCleaner and KeystrokeCleaner functionality.
Ported from productivity-insights with minimal changes:
- MissingActivityColumnsError → ValueError
- EmptyKeystrokeDataError → ValueError
- Import paths updated to workblock_engine
- exec column preserved in ActivityCleaner output (port change)
"""
import pytest
import polars as pl
from datetime import datetime, timedelta, timezone

from workblock_engine.cleaning.activity_cleaner import ActivityCleaner
from workblock_engine.cleaning.keystroke_cleaner import KeystrokeCleaner


# ---------------------------------------------------------------------------
# ActivityCleaner tests (ported from test_activity_cleaner.py)
# ---------------------------------------------------------------------------


class TestActivityCleaner:
    """Test ActivityCleaner specific functionality."""

    # Core functionality tests
    def test_empty_dataframe_handling(self):
        """Test that empty dataframe returns correct schema."""
        cleaner = ActivityCleaner()
        empty_df = pl.DataFrame()
        result = cleaner.clean(empty_df)

        # Check schema
        expected_columns = ["user_id", "timestamp", "name", "title", "url", "task_name",
                          "start", "end", "exec",
                          "app_exe", "app_filename", "duration_minutes", "task_id"]
        assert list(result.columns) == expected_columns
        assert len(result) == 0

    def test_missing_required_columns(self):
        """Test that missing required columns raise ValueError."""
        cleaner = ActivityCleaner()
        # Missing 'name' column
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "title": ["Test"],
            "url": ["https://test.com"],
            "task_name": ["Testing"]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            cleaner.clean(df)

    def test_data_preservation(self):
        """Test that all data is preserved without redaction."""
        cleaner = ActivityCleaner()
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Chrome"],
            "title": ["user@example.com - Ticket 12345"],
            "url": ["https://app.example.com/ticket/12345"],
            "task_name": ["Support"],
            "task_id": [1]
        })

        result = cleaner.clean(df)

        # All data should be preserved
        assert result["title"][0] == "user@example.com - Ticket 12345"
        assert result["url"][0] == "https://app.example.com/ticket/12345"

    def test_duration_calculation(self):
        """Test duration calculation from start/end times."""
        cleaner = ActivityCleaner()
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)

        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [start_time],
            "name": ["VSCode"],
            "title": ["main.py"],
            "url": [""],
            "task_name": ["Development"],
            "task_id": [1],
            "start": [start_time],
            "end": [end_time]
        })

        result = cleaner.clean(df)

        # Should calculate 30 minutes duration
        assert result["duration_minutes"][0] == 30.0

    def test_missing_exec_column(self):
        """Test handling when exec column is missing."""
        cleaner = ActivityCleaner()
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Chrome"],
            "title": ["Google"],
            "url": ["https://google.com"],
            "task_name": ["Browsing"],
            "task_id": [1]
            # No exec column
        })

        result = cleaner.clean(df)

        # Should use name as app_exe when exec is missing
        assert result["app_exe"][0] == "Chrome"

    def test_task_name_cleaning(self):
        """Test that task names are stripped of whitespace."""
        cleaner = ActivityCleaner()
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": [""],
            "task_name": ["  Development  "],  # Extra whitespace
            "task_id": [1]
        })

        result = cleaner.clean(df)

        # Task name should be trimmed
        assert result["task_name"][0] == "Development"

    def test_null_task_id_handling(self):
        """Test that null task_id is filled with 'unknown'."""
        cleaner = ActivityCleaner()
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": [""],
            "task_name": ["Browsing"],
            "task_id": [None]
        })

        result = cleaner.clean(df)

        # Null task_id should become "unknown"
        assert result["task_id"][0] == "unknown"

    def test_timestamp_string_conversion(self):
        """Test that string timestamps are converted to datetime."""
        cleaner = ActivityCleaner()
        timestamp_str = "2024-01-15 10:30:00"

        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [timestamp_str],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": [""],
            "task_name": ["Browsing"],
            "task_id": [1],
            "start": [timestamp_str],
            "end": ["2024-01-15 11:00:00"]
        })

        result = cleaner.clean(df)

        # Timestamps should be datetime objects
        assert isinstance(result["timestamp"][0], datetime)
        assert result["duration_minutes"][0] == 30.0

    def test_app_exe_extraction_windows_path(self):
        """Test app_exe extraction with Windows-style paths."""
        df = pl.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": [datetime.now()] * 3,
            "name": ["Chrome", "Teams", "VSCode"],
            "title": ["Google", "Meeting", "main.py"],
            "url": ["https://google.com", "", ""],
            "task_name": ["Browsing", "Meeting", "Development"],
            "task_id": [1, 2, 3],
            "exec": [
                r"C:\Program Files\Google\Chrome\chrome.exe",
                r"C:\Program Files\Microsoft\Teams\teams.exe",
                r"C:\Users\john\AppData\Local\Programs\Microsoft VS Code\Code.exe"
            ]
        })

        cleaner = ActivityCleaner()
        cleaned = cleaner.clean(df)

        # Verify app_exe extraction
        assert cleaned["app_exe"][0] == r"Chrome\chrome.exe"
        assert cleaned["app_exe"][1] == r"Teams\teams.exe"
        assert cleaned["app_exe"][2] == r"Microsoft VS Code\Code.exe"

    def test_app_exe_extraction_unix_path(self):
        """Test app_exe extraction with Unix-style paths."""
        df = pl.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": [datetime.now()] * 3,
            "name": ["Chrome", "Slack", "VSCode"],
            "title": ["Google", "General", "main.py"],
            "url": ["https://google.com", "", ""],
            "task_name": ["Browsing", "Communication", "Development"],
            "task_id": [1, 2, 3],
            "exec": [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Slack.app/Contents/MacOS/Slack",
                "/usr/local/bin/code"
            ]
        })

        cleaner = ActivityCleaner()
        cleaned = cleaner.clean(df)

        # For Unix paths, we extract parent/exe
        assert cleaned["app_exe"][0] == "MacOS/Google Chrome"
        assert cleaned["app_exe"][1] == "MacOS/Slack"
        assert cleaned["app_exe"][2] == "bin/code"

    def test_app_exe_null_exec(self):
        """Test app_exe extraction when exec is null."""
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Browser"],
            "title": ["Web Page"],
            "url": ["https://example.com"],
            "task_name": ["Browsing"],
            "task_id": [1],
            "exec": [None]
        })

        cleaner = ActivityCleaner()
        cleaned = cleaner.clean(df)

        # Should fall back to name when exec is null
        assert cleaned["app_exe"][0] == "Browser"

    def test_app_exe_mixed_paths(self):
        """Test app_exe extraction with mixed path styles."""
        df = pl.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": [datetime.now()] * 3,
            "name": ["Chrome", "Teams", "Slack"],
            "title": ["Google", "Meeting", "Chat"],
            "url": ["", "", ""],
            "task_name": ["Browsing", "Meeting", "Chat"],
            "task_id": [1, 2, 3],
            "exec": [
                r"C:\Program Files\Google\Chrome\chrome.exe",  # Windows
                "/Applications/Microsoft Teams.app",  # macOS
                None  # Missing
            ]
        })

        cleaner = ActivityCleaner()
        cleaned = cleaner.clean(df)

        assert dict(zip(cleaned["name"], cleaned["app_exe"])) == {
            "Chrome": r"Chrome\chrome.exe",
            "Teams": "Applications/Microsoft Teams.app",
            "Slack": "Slack",
        }

    def test_column_ordering(self):
        """Test that output columns are in the expected order."""
        cleaner = ActivityCleaner()
        df = pl.DataFrame({
            "extra_col": [1],  # Extra column that should be dropped
            "task_id": [1],
            "user_id": [1],
            "timestamp": [datetime.now()],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": ["https://test.com"],
            "task_name": ["Testing"],
            "exec": ["/usr/bin/chrome"]
        })

        result = cleaner.clean(df)

        # Check column order — port adds exec between task_name and app_exe
        expected_order = ["user_id", "timestamp", "start", "end", "name", "title", "url", "task_name",
                         "exec", "app_exe", "app_filename", "duration_minutes", "task_id"]
        assert list(result.columns) == expected_order
        # Extra column should be dropped
        assert "extra_col" not in result.columns

    def test_sorting_by_user_and_timestamp(self):
        """Test that data is sorted by user_id and timestamp."""
        cleaner = ActivityCleaner()
        base_time = datetime.now()

        df = pl.DataFrame({
            "user_id": [2, 1, 2, 1],
            "timestamp": [
                base_time + timedelta(minutes=30),
                base_time + timedelta(minutes=20),
                base_time + timedelta(minutes=10),
                base_time
            ],
            "name": ["Chrome"] * 4,
            "title": ["Page 1", "Page 2", "Page 3", "Page 4"],
            "url": [""] * 4,
            "task_name": ["Work"] * 4,
            "task_id": [1] * 4
        })

        result = cleaner.clean(df)

        # Check sorting
        user_ids = result["user_id"].to_list()
        assert user_ids == [1, 1, 2, 2]  # Sorted by user

        # Check timestamps are sorted within each user
        user1_times = result.filter(pl.col("user_id") == 1)["timestamp"].to_list()
        assert user1_times[0] < user1_times[1]

        user2_times = result.filter(pl.col("user_id") == 2)["timestamp"].to_list()
        assert user2_times[0] < user2_times[1]

    def test_id_deduplication(self):
        """Test that duplicate IDs are removed."""
        cleaner = ActivityCleaner()

        # Create data with duplicate IDs
        base_time = datetime.now()
        df = pl.DataFrame({
            "id": ["act_001", "act_001", "act_002", "act_002", "act_003"],
            "user_id": [1, 1, 1, 2, 2],
            "timestamp": [
                base_time,
                base_time,  # Duplicate with same ID
                base_time + timedelta(minutes=10),
                base_time + timedelta(minutes=10),  # Duplicate with same ID
                base_time + timedelta(minutes=20)
            ],
            "name": ["Chrome", "Firefox", "VSCode", "Slack", "Terminal"],
            "title": ["Page1", "Page2", "Editor", "Chat1", "Bash"],
            "url": ["url1", "url2", "", "", ""],
            "task_name": ["Task1", "Task1_dup", "Task2", "Task2_dup", "Task3"],
            "exec": ["/chrome", "/firefox", "/vscode", "/slack", "/terminal"]
        })

        result = cleaner.clean(df)

        # Check that ID column exists
        assert "id" in result.columns

        # Check duplicates were removed - should have 3 unique IDs
        assert len(result) == 3
        ids = result["id"].to_list()
        assert set(ids) == {"act_001", "act_002", "act_003"}

        # Verify first occurrence was kept
        act_001_row = result.filter(pl.col("id") == "act_001").row(0, named=True)
        assert act_001_row["name"] == "Chrome"  # First occurrence, not "Firefox"
        assert act_001_row["task_name"] == "Task1"  # First occurrence, not "Task1_dup"

        # Test with data without ID column (should not fail)
        df_no_id = pl.DataFrame({
            "user_id": [1, 1],
            "timestamp": [base_time, base_time + timedelta(minutes=10)],
            "name": ["Chrome", "VSCode"],
            "title": ["Page", "Editor"],
            "url": ["", ""],
            "task_name": ["Task1", "Task2"],
            "exec": ["/chrome", "/vscode"]
        })

        result_no_id = cleaner.clean(df_no_id)

        # Should not have ID column if not present in source
        assert "id" not in result_no_id.columns
        assert len(result_no_id) == 2

    def test_utc_timezone_handling(self):
        """Test that timestamps are properly converted to UTC timezone."""
        cleaner = ActivityCleaner()

        # Test with string timestamps
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": ["2025-05-01 12:00:00"],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": ["https://test.com"],
            "task_name": ["Testing"],
            "start": ["2025-05-01 12:00:00"],
            "end": ["2025-05-01 12:30:00"]
        })

        result = cleaner.clean(df)

        # Check that timestamp has UTC timezone
        assert result.schema["timestamp"].time_zone == "UTC"
        assert result.schema["start"].time_zone == "UTC"
        assert result.schema["end"].time_zone == "UTC"

        # Test with datetime objects (should also get UTC timezone)
        df_datetime = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime(2025, 5, 1, 12, 0, 0)],
            "name": ["Chrome"],
            "title": ["Test"],
            "url": ["https://test.com"],
            "task_name": ["Testing"]
        })

        result_datetime = cleaner.clean(df_datetime)

        # Check that timestamp has UTC timezone even when input was naive datetime
        assert result_datetime.schema["timestamp"].time_zone == "UTC"
        assert result_datetime.schema["start"].time_zone == "UTC"  # Created from timestamp
        assert result_datetime.schema["end"].time_zone == "UTC"    # Created from timestamp + duration

    # --- Port-specific test: exec preservation ---

    def test_clean_preserves_exec(self):
        """Port change: exec column must be preserved in output."""
        exec_path = r"C:\Program Files\chrome.exe"
        df = pl.DataFrame({
            "user_id": ["u1"],
            "timestamp": [datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)],
            "name": ["chrome.exe"],
            "title": ["Google - Chrome"],
            "url": ["https://google.com"],
            "task_name": [""],
            "exec": [exec_path],
        })
        cleaner = ActivityCleaner()
        result = cleaner.clean(df)
        assert "exec" in result.columns
        assert result["exec"][0] == exec_path

    def test_clean_adds_duration_column(self):
        """Test default duration when no start/end provided (timestamp + 10 min)."""
        df = pl.DataFrame({
            "user_id": ["u1"],
            "timestamp": [datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)],
            "name": ["chrome.exe"],
            "title": ["Google - Chrome"],
            "url": ["https://google.com"],
            "task_name": [""],
        })
        cleaner = ActivityCleaner()
        result = cleaner.clean(df)
        assert "duration_minutes" in result.columns
        assert result["duration_minutes"][0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# ActivityCleaner app_filename normalization tests
# (ported from test_activity_cleaner_normalization.py)
# ---------------------------------------------------------------------------


class TestAppFilenameNormalization:
    """Test that app_filename is normalized correctly when exec is null."""

    def test_normalize_app_names_without_exec(self):
        """Test that app names are normalized when exec is null."""
        cleaner = ActivityCleaner()

        df = pl.DataFrame({
            "user_id": ["1"] * 6,
            "timestamp": [datetime.now()] * 6,
            "name": [
                "Go2",           # Should become Go2.exe
                "Go2.exe",       # Should stay Go2.exe
                "go2.exe",       # Should become Go2.exe (case normalized)
                "Chrome",        # Should become Chrome.exe
                "chrome.exe",    # Should become Chrome.exe (case normalized)
                "CHROME.EXE",    # Should become Chrome.exe (case normalized)
            ],
            "title": ["Test"] * 6,
            "url": [""] * 6,
            "task_name": ["Work"] * 6,
            "exec": [None] * 6,  # No exec field
        })

        result = cleaner.clean(df)

        # Check app_filename normalization
        expected = {
            "Go2": "Go2.exe", "Go2.exe": "Go2.exe", "go2.exe": "Go2.exe",
            "Chrome": "Chrome.exe", "chrome.exe": "Chrome.exe", "CHROME.EXE": "Chrome.exe",
        }
        assert dict(zip(result["name"], result["app_filename"])) == expected

    def test_preserve_exec_when_available(self):
        """Test that exec-based app_filename is preserved when exec is available."""
        cleaner = ActivityCleaner()

        df = pl.DataFrame({
            "user_id": ["1"] * 3,
            "timestamp": [datetime.now()] * 3,
            "name": ["Chrome", "Firefox", "Code"],
            "title": ["Test"] * 3,
            "url": [""] * 3,
            "task_name": ["Work"] * 3,
            "exec": [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                "/usr/bin/firefox",
                r"C:\Users\User\AppData\Local\Programs\Microsoft VS Code\Code.exe",
            ],
        })

        result = cleaner.clean(df)

        # When exec is available, use the actual filename from path
        assert dict(zip(result["name"], result["app_filename"])) == {
            "Chrome": "chrome.exe", "Firefox": "firefox", "Code": "Code.exe",
        }

    def test_mixed_exec_and_name(self):
        """Test handling of mixed exec and name scenarios."""
        cleaner = ActivityCleaner()

        df = pl.DataFrame({
            "user_id": ["1"] * 4,
            "timestamp": [datetime.now()] * 4,
            "name": ["Chrome", "firefox", "Go2", "slack"],
            "title": ["Test"] * 4,
            "url": [""] * 4,
            "task_name": ["Work"] * 4,
            "exec": [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                None,  # Will use normalized name
                None,  # Will use normalized name
                "/Applications/Slack.app/Contents/MacOS/Slack",
            ],
        })

        result = cleaner.clean(df)

        # Mix of exec-based and name-based app_filename
        assert dict(zip(result["name"], result["app_filename"])) == {
            "Chrome": "chrome.exe",
            "firefox": "Firefox.exe",
            "Go2": "Go2.exe",
            "slack": "Slack",
        }


# ---------------------------------------------------------------------------
# KeystrokeCleaner tests (ported from test_keystroke_cleaner.py)
# ---------------------------------------------------------------------------


class TestKeystrokeCleaner:
    """Test KeystrokeCleaner functionality."""

    def test_empty_dataframe_handling(self):
        """Test that empty dataframe returns correct schema."""
        cleaner = KeystrokeCleaner()
        empty_df = pl.DataFrame()
        result = cleaner.clean(empty_df)

        # Check schema (ID not included for empty dataframe)
        expected_columns = ["user_id", "timestamp", "app", "keystrokes", "window_title",
                          "keystroke_count", "duration_seconds"]
        assert list(result.columns) == expected_columns
        assert len(result) == 0

    def test_missing_required_columns(self):
        """Test that missing keystrokes column raises ValueError."""
        cleaner = KeystrokeCleaner()
        # Missing 'keystrokes' column
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "app": ["Chrome"],
            "window_title": ["Test Page"]
        })

        with pytest.raises(ValueError, match="No keystroke data found in DataFrame"):
            cleaner.clean(df)

    def test_column_normalization(self):
        """Test that columns are normalized correctly."""
        cleaner = KeystrokeCleaner()

        # Test with alternative column names
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime.now()],
            "exec": ["Chrome.exe"],  # Should be renamed to app
            "keystroke": ["Hello world"],  # Should be renamed to keystrokes
            "title": ["Test Page"]  # Should be renamed to window_title
        })

        result = cleaner.clean(df)

        # Check column names are normalized
        assert "timestamp" in result.columns
        assert "app" in result.columns
        assert "keystrokes" in result.columns
        assert "window_title" in result.columns

    def test_data_preservation(self):
        """Test that all keystroke data is preserved without redaction."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": [datetime.now()] * 3,
            "app": ["Chrome"] * 3,
            "keystrokes": [
                "user@example.com password123",
                "4111-1111-1111-1111",
                "Normal typing data"
            ],
            "window_title": ["Login Page", "Payment", "Document"]
        })

        result = cleaner.clean(df)

        # All data should be preserved
        assert dict(zip(result["window_title"], result["keystrokes"])) == {
            "Login Page": "user@example.com password123",
            "Payment": "4111-1111-1111-1111",
            "Document": "Normal typing data",
        }

    def test_app_name_extraction(self):
        """Test that app names are extracted from paths."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": [datetime.now()] * 3,
            "app": [
                r"C:\Program Files\Chrome\chrome.exe",  # Windows
                "/Applications/Slack.app",  # macOS
                "firefox"  # No path
            ],
            "keystrokes": ["test"] * 3,
            "window_title": ["Page"] * 3
        })

        result = cleaner.clean(df)

        # Check app_name extraction
        assert dict(zip(result["app"], result["app_name"])) == {
            r"C:\Program Files\Chrome\chrome.exe": "chrome.exe",
            "/Applications/Slack.app": "Slack.app",
            "firefox": "firefox",
        }

    def test_keystroke_metrics(self):
        """Test keystroke count and duration calculations."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            "user_id": [1, 1],
            "timestamp": [datetime.now()] * 2,
            "app": ["VSCode"] * 2,
            "keystrokes": [
                "Hello world!",  # 12 characters
                "This is a much longer string with many characters"  # 49 characters
            ],
            "window_title": ["main.py"] * 2
        })

        result = cleaner.clean(df)

        # Check keystroke metrics
        assert result["keystroke_count"][0] == 12
        assert result["keystroke_count"][1] == 49

        # Duration estimate (3.3 chars per second)
        assert abs(result["duration_seconds"][0] - 12/3.3) < 0.1
        assert abs(result["duration_seconds"][1] - 49/3.3) < 0.1

    def test_window_title_derivation(self):
        """Test window title is derived from URL when missing."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            "user_id": [1, 1],
            "timestamp": [datetime.now()] * 2,
            "app": ["Chrome"] * 2,
            "keystrokes": ["search"] * 2,
            "url": ["https://github.com/repo", ""]
        })

        result = cleaner.clean(df)

        # Should derive window_title from URL domain
        assert dict(zip(result["url"], result["window_title"])) == {
            "https://github.com/repo": "github.com",
            "": "Chrome",
        }

    def test_string_timestamp_conversion(self):
        """Test that string timestamps are converted properly."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": ["2024-01-15 10:30:00"],
            "app": ["Chrome"],
            "keystrokes": ["test"],
            "window_title": ["Test"]
        })

        result = cleaner.clean(df)

        # Timestamp should be converted from string
        assert isinstance(result["timestamp"][0], datetime)

    def test_data_sorting(self):
        """Test that data is sorted by user and timestamp."""
        cleaner = KeystrokeCleaner()
        base_time = datetime.now()

        df = pl.DataFrame({
            "user_id": [2, 1, 2, 1],
            "timestamp": [
                base_time + timedelta(minutes=30),
                base_time + timedelta(minutes=20),
                base_time + timedelta(minutes=10),
                base_time
            ],
            "app": ["Chrome"] * 4,
            "keystrokes": ["test"] * 4,
            "window_title": ["Page"] * 4
        })

        result = cleaner.clean(df)

        # Check sorting
        user_ids = result["user_id"].to_list()
        assert user_ids == [1, 1, 2, 2]  # Sorted by user

        # Check timestamps are sorted within each user
        user1_times = result.filter(pl.col("user_id") == 1)["timestamp"].to_list()
        assert user1_times[0] < user1_times[1]

        user2_times = result.filter(pl.col("user_id") == 2)["timestamp"].to_list()
        assert user2_times[0] < user2_times[1]

    def test_id_deduplication(self):
        """Test that duplicate IDs are removed."""
        cleaner = KeystrokeCleaner()

        # Create data with duplicate IDs
        df = pl.DataFrame({
            "id": ["ks_001", "ks_001", "ks_002", "ks_002", "ks_003"],
            "user_id": [1, 1, 1, 2, 2],
            "timestamp": [
                datetime(2024, 1, 15, 10, 0),
                datetime(2024, 1, 15, 10, 0),  # Duplicate with same ID
                datetime(2024, 1, 15, 10, 30),
                datetime(2024, 1, 15, 10, 30),  # Duplicate with same ID
                datetime(2024, 1, 15, 11, 0)
            ],
            "app": ["Chrome", "Chrome", "VSCode", "VSCode", "Slack"],
            "keystrokes": ["hello", "hello_dup", "code", "code_dup", "message"],
            "window_title": ["Page1", "Page1", "Editor", "Editor", "Chat"]
        })

        result = cleaner.clean(df)

        # Check that ID column exists
        assert "id" in result.columns

        # Check duplicates were removed - should have 3 unique IDs
        assert len(result) == 3
        ids = result["id"].to_list()
        assert set(ids) == {"ks_001", "ks_002", "ks_003"}

        # Verify first occurrence was kept
        ks_001_row = result.filter(pl.col("id") == "ks_001").row(0, named=True)
        assert ks_001_row["keystrokes"] == "hello"  # First occurrence, not "hello_dup"

        # Test with data without ID column (should not fail)
        df_no_id = pl.DataFrame({
            "user_id": [1, 1],
            "timestamp": [datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 11, 0)],
            "app": ["Chrome", "VSCode"],
            "keystrokes": ["test1", "test2"],
            "window_title": ["Page", "Code"]
        })

        result_no_id = cleaner.clean(df_no_id)

        # Should not have ID column if not present in source
        assert "id" not in result_no_id.columns
        assert len(result_no_id) == 2

    def test_url_preservation(self):
        """Test that URL column is preserved when present."""
        cleaner = KeystrokeCleaner()

        df = pl.DataFrame({
            'user_id': ['user1', 'user1'],
            'timestamp': ['2025-01-15 09:00:00', '2025-01-15 09:30:00'],
            'app': ['Chrome', 'Chrome'],
            'keystrokes': ['test', 'search'],
            'window_title': ['Google', 'GitHub'],
            'url': ['https://google.com', 'https://github.com/project']
        })

        result = cleaner.clean(df)

        # URL column should be preserved
        assert 'url' in result.columns
        assert result['url'].to_list() == ['https://google.com', 'https://github.com/project']

    def test_utc_timezone_handling(self):
        """Test that timestamp is properly converted to UTC timezone."""
        cleaner = KeystrokeCleaner()

        # Test with string timestamp
        df = pl.DataFrame({
            "user_id": [1],
            "timestamp": ["2025-05-01 12:00:00"],
            "app": ["Chrome"],
            "keystrokes": ["Hello world"],
            "window_title": ["Test Page"]
        })

        result = cleaner.clean(df)

        # Check that timestamp has UTC timezone
        assert result.schema["timestamp"].time_zone == "UTC"

        # Test with datetime object (should also get UTC timezone)
        df_timestamp = pl.DataFrame({
            "user_id": [1],
            "timestamp": [datetime(2025, 5, 1, 12, 0, 0)],
            "app": ["Chrome"],
            "keystrokes": ["Hello world"],
            "window_title": ["Test Page"]
        })

        result_timestamp = cleaner.clean(df_timestamp)

        # Check that timestamp has UTC timezone even when input was naive datetime
        assert result_timestamp.schema["timestamp"].time_zone == "UTC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
