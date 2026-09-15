from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from meteo_swiss_mcp.server import _lead_time_swiss_to_utc


@pytest.fixture
def freeze_swiss_now_fixture(mocker):
    """
    Return a function that pins datetime.now() inside server.py to a fixed Swiss local instant,
    so the conversion can be tested against a known calendar day.
    """
    def freeze(iso: str) -> None:
        frozen = datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo("Europe/Zurich"))
        datetime_mock = mocker.patch("meteo_swiss_mcp.server.datetime", wraps=datetime)
        datetime_mock.now.return_value = frozen

    return freeze


def test_lead_time_swiss_to_utc_winter(freeze_swiss_now_fixture):
    # In January Switzerland stays on CET all day, so the elapsed time from Swiss
    # midnight to the target hour equals the Swiss lead time itself.
    freeze_swiss_now_fixture("2026-01-15T09:30:00")

    assert _lead_time_swiss_to_utc(0) == 0
    assert _lead_time_swiss_to_utc(2) == 2
    assert _lead_time_swiss_to_utc(14) == 14
    assert _lead_time_swiss_to_utc(36) == 36


def test_lead_time_swiss_to_utc_summer(freeze_swiss_now_fixture):
    # In July Switzerland stays on CEST all day, so the lead time is again unchanged
    freeze_swiss_now_fixture("2026-07-15T09:30:00")

    assert _lead_time_swiss_to_utc(0) == 0
    assert _lead_time_swiss_to_utc(2) == 2
    assert _lead_time_swiss_to_utc(14) == 14
    assert _lead_time_swiss_to_utc(36) == 36


def test_lead_time_swiss_to_utc_dst_start(freeze_swiss_now_fixture):
    # DST starts 2026-03-29 02:00 Swiss local: the clock jumps straight to 03:00, making this
    # local day 23 hours long. Lead times reaching past the jump span one hour less real time.
    freeze_swiss_now_fixture("2026-03-29T00:00:00")

    assert _lead_time_swiss_to_utc(2) == 2  # before the transition, unchanged
    assert _lead_time_swiss_to_utc(12) == 11
    assert _lead_time_swiss_to_utc(36) == 35


def test_lead_time_swiss_to_utc_dst_end(freeze_swiss_now_fixture):
    # DST ends 2026-10-25 03:00 Swiss local: the clock falls back to 02:00, making this local
    # day 25 hours long. Lead times reaching past the fallback span one hour more real time.
    freeze_swiss_now_fixture("2026-10-25T00:00:00")

    assert _lead_time_swiss_to_utc(2) == 2  # before the transition, unchanged
    assert _lead_time_swiss_to_utc(12) == 13
    assert _lead_time_swiss_to_utc(36) == 37


def test_lead_time_swiss_to_utc_returns_int(freeze_swiss_now_fixture):
    freeze_swiss_now_fixture("2026-01-15T09:30:00")

    assert isinstance(_lead_time_swiss_to_utc(14), int)
