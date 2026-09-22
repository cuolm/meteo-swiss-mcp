from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from swiss_weather_mcp.server import _parse_swiss_time

SWISS_TZ = ZoneInfo("Europe/Zurich")


def test_a_timestamp_without_a_zone_is_read_as_swiss_local_time():
    parsed = _parse_swiss_time("2026-09-23T14:00")
    assert parsed == datetime(2026, 9, 23, 14, tzinfo=SWISS_TZ)


def test_summer_time_is_two_hours_ahead_of_utc():
    # In September Switzerland is on CEST, so 14:00 local is 12:00 UTC
    parsed = _parse_swiss_time("2026-09-23T14:00")
    assert parsed.astimezone(timezone.utc).hour == 12


def test_winter_time_is_one_hour_ahead_of_utc():
    # In January Switzerland is on CET, so the same local hour is 13:00 UTC
    parsed = _parse_swiss_time("2026-01-23T14:00")
    assert parsed.astimezone(timezone.utc).hour == 13


def test_an_explicit_offset_is_honoured_rather_than_overwritten():
    parsed = _parse_swiss_time("2026-09-23T14:00+00:00")
    assert parsed.astimezone(timezone.utc).hour == 14


def test_a_date_on_its_own_is_read_as_that_day_at_midnight():
    parsed = _parse_swiss_time("2026-09-23")
    assert parsed == datetime(2026, 9, 23, 0, 0, tzinfo=SWISS_TZ)


def test_a_space_between_date_and_time_is_accepted():
    assert _parse_swiss_time("2026-09-23 14:00") == _parse_swiss_time("2026-09-23T14:00")


def test_an_unreadable_timestamp_says_what_a_good_one_looks_like():
    with pytest.raises(ValueError, match="2026-09-23T14:00"):
        _parse_swiss_time("tomorrow afternoon")
