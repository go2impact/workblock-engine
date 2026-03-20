from datetime import datetime, timezone

from workblock_engine.utils.time_utils import make_tz_aware
from workblock_engine.utils.app_utils import normalize_app_name


class TestMakeTzAware:
    def test_naive_datetime_becomes_utc(self):
        dt = datetime(2026, 1, 1, 12, 0, 0)
        result = make_tz_aware(dt)
        assert result.tzinfo == timezone.utc

    def test_aware_datetime_unchanged(self):
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = make_tz_aware(dt)
        assert result == dt


class TestNormalizeAppName:
    def test_chrome_exe_path(self):
        assert normalize_app_name(r"C:\Program Files\Google\Chrome\Application\chrome.exe") == "chrome"

    def test_slack_unix_path(self):
        assert normalize_app_name("/Applications/Slack.app/Contents/MacOS/Slack") == "slack"

    def test_none_returns_unknown(self):
        assert normalize_app_name(None) == "unknown"

    def test_empty_returns_unknown(self):
        assert normalize_app_name("") == "unknown"

    def test_unknown_app_returns_lowercase(self):
        assert normalize_app_name("MyCustomApp.exe") == "mycustomapp"
