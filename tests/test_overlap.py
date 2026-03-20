import polars as pl
from datetime import datetime, timezone
from workblock_engine.building.overlap import resolve_activity_overlaps


class TestResolveOverlaps:
    def test_no_overlaps_unchanged(self):
        df = pl.DataFrame({
            "start": [
                datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc),
            ],
            "end": [
                datetime(2026, 1, 1, 9, 25, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            ],
        })
        result = resolve_activity_overlaps(df)
        assert result["end"].to_list() == df["end"].to_list()

    def test_overlap_trims_end(self):
        df = pl.DataFrame({
            "start": [
                datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 9, 20, tzinfo=timezone.utc),
            ],
            "end": [
                datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
            ],
            "duration_minutes": [30.0, 40.0],
        })
        result = resolve_activity_overlaps(df)
        assert result["end"][0] == datetime(2026, 1, 1, 9, 20, tzinfo=timezone.utc)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"start": [], "end": []}).cast({"start": pl.Datetime, "end": pl.Datetime})
        result = resolve_activity_overlaps(df)
        assert result.is_empty()
